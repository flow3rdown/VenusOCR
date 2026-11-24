# Copyright (c) 2026 VenusOCR. All Rights Reserved.
# Portions Copyright (c) PaddlePaddle Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Document parsing pipeline: layout detection -> recognition -> assembly.

The compatibility path preserves the PaddleX 3.6.0 serial pipeline behavior.
Source attribution and the maintained compatibility map are documented in
``THIRD_PARTY_NOTICES.md`` and ``agent.md``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from venusocr.config import VenusOCRConfig
from venusocr.data.readers import ReadImage
from venusocr.data.sampler import ImageBatchSampler
from venusocr.models.layout import create_layout_detector
from venusocr.models.vlm import VLRecognizer
from venusocr.pipeline.crop import CropByBoxes, crop_margin, gather_imgs
from venusocr.pipeline.merge import filter_overlap_boxes, merge_blocks
from venusocr.pipeline.tables import (
    convert_otsl_to_html,
    tokenize_figure_of_table,
    untokenize_figure_of_table,
)
from venusocr.pipeline.text import truncate_repetitive_content
from venusocr.results.page import Block, PageResult
from venusocr.utils.logging import get_logger
from venusocr.utils.profiling import ParseMetrics

logger = get_logger(__name__)

IMAGE_LABELS = ["image", "header_image", "footer_image"]
UNSUPPORTED_VLM_LABELS = ["chart", "seal", "spotting"]

# Defaults pinned to the project's PaddleOCR/PaddleX 3.6.0 compatibility profile.
DEFAULT_PAGE_BATCH_SIZE = 64
DEFAULT_MARKDOWN_IGNORE_LABELS = [
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
]
# Default pixel constraints for VLM preprocessing (upstream
# ``get_layout_parsing_results``).
DEFAULT_MIN_PIXELS = 112896
DEFAULT_MAX_PIXELS = 1003520
# Default generation length cap (upstream ``_paddleocr_vl_run_vl_recognition_batches``).
DEFAULT_MAX_NEW_TOKENS = 4096


class DocumentParsingPipeline:
    """Parse document images/PDFs into structured page results.

    PP-DocLayoutV3 performs layout detection locally, and region-level
    text/table/formula recognition is delegated to a VenusOCR model served
    through an OpenAI-compatible (vLLM) endpoint.

    Args:
        config: A validated :class:`VenusOCRConfig`.
        layout_prep_cpu_workers: Number of threads for CPU-side page
            preparation (0 disables parallel preparation, same as upstream).
    """

    def __init__(self, config: VenusOCRConfig, layout_prep_cpu_workers: int = 0) -> None:
        self.config = config
        engine = config.engine
        if engine.vl_rec_backend != "vllm-server":
            raise ValueError(
                "VenusOCR only supports the OpenAI-compatible 'vllm-server' "
                f"recognition backend, got {engine.vl_rec_backend!r}."
            )
        logger.info(
            "│ 初始化引擎 · layout=%s · device=%s · server=%s · model=%s",
            engine.layout_model_name,
            engine.device,
            engine.vl_rec_server_url,
            engine.layout_model_dir or "auto",
        )
        self.layout_detector = create_layout_detector(
            backend=engine.layout_backend,
            model_name=engine.layout_model_name,
            model_dir=engine.layout_model_dir,
            device=engine.device,
            threshold=engine.layout_threshold,
            layout_nms=engine.layout_nms,
            layout_unclip_ratio=engine.layout_unclip_ratio,
            layout_merge_bboxes_mode=engine.layout_merge_bboxes_mode,
            batch_size=engine.layout_batch_size,
        )
        self.vl_recognizer = VLRecognizer(
            server_url=engine.vl_rec_server_url,
            model_name=engine.vl_rec_api_model_name,
            api_key=engine.vl_rec_api_key,
            max_concurrency=engine.vl_rec_max_concurrency,
        )
        self.batch_sampler = ImageBatchSampler(batch_size=DEFAULT_PAGE_BATCH_SIZE)
        self.img_reader = ReadImage(format="BGR")
        self.crop_by_boxes = CropByBoxes()
        self.layout_prep_cpu_workers = max(0, int(layout_prep_cpu_workers))
        self.last_metrics: Optional[ParseMetrics] = None
        self._active_metrics: Optional[ParseMetrics] = None

    # Thread-local metrics sink: ``predict`` runs single-threaded, while the
    # shared-model batch runner drives ``get_layout_parsing_results`` from
    # several worker threads concurrently. A plain attribute would mix
    # per-batch stage timings across threads. Lazily initialized so that
    # instances created without ``__init__`` (tests) still work.
    @property
    def _active_metrics(self) -> Optional[ParseMetrics]:
        tls = self.__dict__.get("_metrics_tls")
        return getattr(tls, "value", None) if tls is not None else None

    @_active_metrics.setter
    def _active_metrics(self, value: Optional[ParseMetrics]) -> None:
        tls = self.__dict__.get("_metrics_tls")
        if tls is None:
            tls = self.__dict__["_metrics_tls"] = threading.local()
        tls.value = value

    def close(self):
        """Release recognition client resources."""
        if hasattr(self, "layout_detector"):
            self.layout_detector.close()
        if hasattr(self, "vl_recognizer"):
            self.vl_recognizer.close()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def _prepare_settings(self):
        """Build the shared model/VLM settings from ``config.predict``.

        Returns:
            A tuple ``(model_settings, vlm_kwargs, prompt_label)`` used by
            both :meth:`predict` and the shared-model batch runner.
        """
        cfg = self.config.predict
        prompt_label = cfg.prompt_label
        if not cfg.use_layout_detection:
            prompt_label = prompt_label if prompt_label else "text"
            if prompt_label.lower() == "ocr":
                prompt_label = "text"
            if prompt_label.lower() not in ["text", "formula", "table"]:
                raise ValueError(
                    "Layout detection is disabled. predict.prompt_label must be "
                    "one of: text, formula, table"
                )

        model_settings = dict(
            use_doc_preprocessor=False,
            use_layout_detection=cfg.use_layout_detection,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_ocr_for_image_block=cfg.use_ocr_for_image_block,
            format_block_content=(
                cfg.format_block_content if cfg.format_block_content is not None else False
            ),
            merge_layout_blocks=(
                cfg.merge_layout_blocks if cfg.merge_layout_blocks is not None else True
            ),
            markdown_ignore_labels=(
                cfg.markdown_ignore_labels
                if cfg.markdown_ignore_labels is not None
                else DEFAULT_MARKDOWN_IGNORE_LABELS
            ),
            return_layout_polygon_points=(
                False if cfg.layout_shape_mode == "rect" else True
            ),
        )

        vlm_kwargs = {
            "repetition_penalty": cfg.repetition_penalty,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "min_pixels": cfg.min_pixels,
            "max_pixels": cfg.max_pixels,
            "max_new_tokens": cfg.max_new_tokens,
        }
        return model_settings, vlm_kwargs, prompt_label

    def predict(self, input, *, log_progress: bool = True) -> List[PageResult]:
        """Run document parsing on one image/PDF file (or numpy array).

        Args:
            input: Path to an image/PDF document, a numpy array (BGR), or
                a list of them.

        Returns:
            A list of :class:`PageResult` (one entry per page).
        """
        cfg = self.config.predict
        metrics = ParseMetrics(input_path=input if isinstance(input, str) else "")
        self._active_metrics = metrics
        pipeline_started = time.perf_counter()

        model_settings, vlm_kwargs, prompt_label = self._prepare_settings()

        results: List[PageResult] = []
        sampler = iter(self.batch_sampler(input))
        while True:
            load_started = time.perf_counter()
            try:
                batch_data = next(sampler)
            except StopIteration:
                break
            image_arrays = self.img_reader(batch_data.instances)
            metrics.add_stage("input_loading", time.perf_counter() - load_started)
            metrics.page_batches += 1
            metrics.page_count += len(image_arrays)

            document_page_count = next(
                (count for count in batch_data.page_counts if count is not None),
                metrics.page_count,
            )
            if log_progress and metrics.page_batches == 1:
                input_kind = "PDF" if any(
                    count is not None for count in batch_data.page_counts
                ) else "图片"
                logger.info(
                    "├─ 输入就绪 · %s · %d 页 · 首批 %d 页",
                    input_kind,
                    document_page_count,
                    len(image_arrays),
                )
            if log_progress and document_page_count > len(image_arrays):
                start_page = metrics.page_count - len(image_arrays) + 1
                logger.info(
                    "├─ 正在处理第 %d-%d / %d 页",
                    start_page,
                    metrics.page_count,
                    document_page_count,
                )

            if model_settings["use_layout_detection"]:
                layout_started = time.perf_counter()
                layout_det_results = self.layout_detector.predict(
                    image_arrays,
                    layout_shape_mode=cfg.layout_shape_mode,
                )
                metrics.add_stage(
                    "layout_detection", time.perf_counter() - layout_started
                )
                metrics.layout_regions += sum(
                    len(result["boxes"]) for result in layout_det_results
                )
                prep_started = time.perf_counter()
                imgs_in_doc = [
                    gather_imgs(doc_img, layout_det_res["boxes"])
                    for doc_img, layout_det_res in zip(image_arrays, layout_det_results)
                ]
                metrics.add_stage(
                    "region_preparation", time.perf_counter() - prep_started
                )
            else:
                layout_det_results = []
                for image in image_arrays:
                    layout_det_results.append(
                        {
                            "input_path": None,
                            "page_index": None,
                            "boxes": [
                                {
                                    "cls_id": 0,
                                    "label": prompt_label.lower(),
                                    "score": 1,
                                    "coordinate": [
                                        0,
                                        0,
                                        image.shape[1],
                                        image.shape[0],
                                    ],
                                }
                            ],
                        }
                    )
                imgs_in_doc = [[] for _ in layout_det_results]
                metrics.layout_regions += len(layout_det_results)

            page_results, _ = self._process_vlm(
                batch_data,
                image_arrays,
                layout_det_results,
                imgs_in_doc,
                model_settings,
                vlm_kwargs,
                cfg,
            )
            results.extend(page_results)
        metrics.output_blocks = sum(
            len(page_result.get("parsing_res_list", [])) for page_result in results
        )
        metrics.pipeline_seconds = time.perf_counter() - pipeline_started
        self.last_metrics = metrics
        self._active_metrics = None
        return results

    __call__ = predict

    # ------------------------------------------------------------------
    # page preparation (filter -> crop -> merge -> collect VLM inputs)
    # ------------------------------------------------------------------
    def _prepare_page(self, payload):
        """Filter -> crop -> merge -> build VLM inputs for one page."""
        i, image, layout_det_res, imgs_in_doc_for_img, layout_prep_cfg = payload
        layout_det_res = filter_overlap_boxes(
            layout_det_res, layout_prep_cfg["layout_shape_mode"]
        )
        boxes = layout_det_res["boxes"]
        blocks_for_img = self.crop_by_boxes(
            image, boxes, layout_prep_cfg["layout_shape_mode"]
        )
        if layout_prep_cfg["merge_layout_blocks"]:
            blocks_for_img = merge_blocks(
                blocks_for_img,
                non_merge_labels=layout_prep_cfg["image_labels"] + ["table"],
            )
        (
            page_vlm_entries,
            page_has_spotting,
            page_drop_figures,
        ) = self._collect_page_vlm_entries(
            i, blocks_for_img, imgs_in_doc_for_img, layout_prep_cfg
        )
        return (
            i,
            blocks_for_img,
            page_vlm_entries,
            page_has_spotting,
            page_drop_figures,
        )

    def _collect_page_vlm_entries(
        self,
        page_idx,
        blocks_for_img,
        imgs_in_doc_for_img,
        layout_prep_cfg,
    ):
        page_vlm_entries = []
        page_has_spotting = False
        page_drop_figures = set()

        for j, block in enumerate(blocks_for_img):
            block_img = block["img"]
            block_label = block["label"]
            if (
                block_label not in layout_prep_cfg["image_labels"]
                and block_img is not None
            ):
                figure_token_map = {}
                text_prompt = "Text Recognition:"
                blk_min_pixels = layout_prep_cfg["ocr_min_pixels"]
                blk_max_pixels = layout_prep_cfg["ocr_max_pixels"]
                drop_figures = []
                if block_label == "table":
                    text_prompt = "Table Recognition:"
                    block_img, figure_token_map, drop_figures = (
                        tokenize_figure_of_table(
                            block_img, block["box"], imgs_in_doc_for_img
                        )
                    )
                    blk_min_pixels = layout_prep_cfg["table_min_pixels"]
                    blk_max_pixels = layout_prep_cfg["table_max_pixels"]
                elif "formula" in block_label and block_label != "formula_number":
                    text_prompt = "Formula Recognition:"
                    crop_img = crop_margin(block_img)
                    w, h, _ = crop_img.shape
                    if w > 2 and h > 2:
                        block_img = crop_img
                    blk_min_pixels = layout_prep_cfg["formula_min_pixels"]
                    blk_max_pixels = layout_prep_cfg["formula_max_pixels"]
                page_vlm_entries.append(
                    (
                        page_idx,
                        j,
                        block_img,
                        text_prompt,
                        (blk_min_pixels, blk_max_pixels),
                        figure_token_map,
                    )
                )
                page_drop_figures.update(drop_figures)

        return page_vlm_entries, page_has_spotting, page_drop_figures

    # ------------------------------------------------------------------
    # VLM recognition + result assembly
    # ------------------------------------------------------------------
    def get_layout_parsing_results(
        self,
        images,
        layout_det_results,
        imgs_in_doc,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_ocr_for_image_block=False,
        vlm_kwargs=None,
        merge_layout_blocks=True,
        layout_shape_mode="auto",
    ):
        """Run region recognition over layout blocks and assemble page results.

        Returns:
            A tuple ``(parsing_res_lists, spotting_res_list, imgs_in_doc,
            request_counts)``; ``request_counts[i]`` is the number of VLM
            requests issued for page ``i``.
        """
        if vlm_kwargs is None:
            vlm_kwargs = {}
        else:
            # This method is called once per page batch. Never consume the
            # caller-owned dictionary, otherwise later batches silently lose
            # explicit pixel constraints.
            vlm_kwargs = dict(vlm_kwargs)

        min_pixels = vlm_kwargs.pop("min_pixels", None)
        default_min_pixels = min_pixels if min_pixels is not None else DEFAULT_MIN_PIXELS
        max_pixels = vlm_kwargs.pop("max_pixels", None)
        default_max_pixels = max_pixels if max_pixels is not None else DEFAULT_MAX_PIXELS

        vis_image_labels = IMAGE_LABELS + UNSUPPORTED_VLM_LABELS
        # VenusOCR never sends visual regions to the VLM. Keep the argument in
        # this vendored-compatible method signature, but enforce the model's
        # actual capability even when this method is called directly.
        image_labels = IMAGE_LABELS + UNSUPPORTED_VLM_LABELS
        ocr_min_pixels = vlm_kwargs.pop("ocr_min_pixels", default_min_pixels)
        ocr_max_pixels = vlm_kwargs.pop("ocr_max_pixels", default_max_pixels)
        table_min_pixels = vlm_kwargs.pop("table_min_pixels", default_min_pixels)
        table_max_pixels = vlm_kwargs.pop("table_max_pixels", default_max_pixels)
        chart_min_pixels = vlm_kwargs.pop("chart_min_pixels", default_min_pixels)
        chart_max_pixels = vlm_kwargs.pop("chart_max_pixels", default_max_pixels)
        formula_min_pixels = vlm_kwargs.pop("formula_min_pixels", default_min_pixels)
        formula_max_pixels = vlm_kwargs.pop("formula_max_pixels", default_max_pixels)
        seal_min_pixels = vlm_kwargs.pop("seal_min_pixels", default_min_pixels)
        seal_max_pixels = vlm_kwargs.pop("seal_max_pixels", default_max_pixels)

        layout_prep_cfg = {
            "layout_shape_mode": layout_shape_mode,
            "merge_layout_blocks": merge_layout_blocks,
            "image_labels": image_labels,
            "use_chart_recognition": use_chart_recognition,
            "use_seal_recognition": use_seal_recognition,
            "ocr_min_pixels": ocr_min_pixels,
            "ocr_max_pixels": ocr_max_pixels,
            "table_min_pixels": table_min_pixels,
            "table_max_pixels": table_max_pixels,
            "chart_min_pixels": chart_min_pixels,
            "chart_max_pixels": chart_max_pixels,
            "formula_min_pixels": formula_min_pixels,
            "formula_max_pixels": formula_max_pixels,
            "seal_min_pixels": seal_min_pixels,
            "seal_max_pixels": seal_max_pixels,
        }

        num_pages = len(images)
        page_payloads = [
            (
                i,
                images[i],
                layout_det_results[i],
                imgs_in_doc[i],
                layout_prep_cfg,
            )
            for i in range(num_pages)
        ]

        max_workers = (
            min(self.layout_prep_cpu_workers, num_pages) if num_pages > 0 else 0
        )
        prep_started = time.perf_counter()
        if num_pages > 1 and max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as cpu_pool:
                page_results = list(cpu_pool.map(self._prepare_page, page_payloads))
        else:
            page_results = [self._prepare_page(p) for p in page_payloads]
        # Per-page VLM request counts, used by the shared-model batch runner
        # for exact per-document metrics attribution.
        request_counts = [len(page_result[2]) for page_result in page_results]

        # --- aggregate VLM batches grouped by pixel constraints ---
        blocks = []
        has_spotting = False
        drop_figures_set = set()
        batch_dict_by_pixel = {}
        id2pixel_key_map = {}
        for (
            _,
            blocks_for_img,
            page_vlm_entries,
            page_has_spotting,
            page_drop_figures,
        ) in page_results:
            blocks.append(blocks_for_img)
            if page_has_spotting:
                has_spotting = True
            drop_figures_set.update(page_drop_figures)
            for (
                i,
                j,
                block_img,
                text_prompt,
                pixel_key,
                figure_token_map,
            ) in page_vlm_entries:
                if pixel_key not in batch_dict_by_pixel:
                    batch_dict_by_pixel[pixel_key] = {
                        "images": [],
                        "queries": [],
                        "figure_token_maps": [],
                        "vlm_block_ids": [],
                        "curr_vlm_block_idx": 0,
                    }
                batch_dict_by_pixel[pixel_key]["images"].append(block_img)
                batch_dict_by_pixel[pixel_key]["queries"].append(text_prompt)
                batch_dict_by_pixel[pixel_key]["figure_token_maps"].append(
                    figure_token_map
                )
                batch_dict_by_pixel[pixel_key]["vlm_block_ids"].append((i, j))
                id2pixel_key_map[(i, j)] = pixel_key

        active_metrics = getattr(self, "_active_metrics", None)
        if active_metrics is not None:
            active_metrics.add_stage(
                "region_preparation", time.perf_counter() - prep_started
            )
            active_metrics.recognition_requests += len(id2pixel_key_map)

        del images, layout_det_results, page_results

        # --- run VL recognition per pixel-constraint group ---
        if vlm_kwargs.get("max_new_tokens", None) is None:
            vlm_kwargs["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS

        recognition_started = time.perf_counter()
        for pixel_key in batch_dict_by_pixel:
            min_px, max_px = pixel_key
            pv_images = batch_dict_by_pixel[pixel_key]["images"]
            queries = batch_dict_by_pixel[pixel_key]["queries"]
            batch_results = self.vl_recognizer.recognize(
                [
                    {
                        "image": image,
                        "query": query,
                    }
                    for image, query in zip(pv_images, queries)
                ],
                skip_special_tokens=False if has_spotting else True,
                min_pixels=min_px,
                max_pixels=max_px,
                **vlm_kwargs,
            )
            del pv_images, queries
            batch_dict_by_pixel[pixel_key]["vlm_results"] = batch_results
        if active_metrics is not None:
            active_metrics.add_stage(
                "vlm_recognition", time.perf_counter() - recognition_started
            )

        assembly_started = time.perf_counter()
        (
            parsing_res_lists,
            spotting_res_list,
        ) = self._assemble_parsing_results(
            blocks,
            batch_dict_by_pixel,
            id2pixel_key_map,
            drop_figures_set,
            vis_image_labels,
        )
        if active_metrics is not None:
            active_metrics.add_stage(
                "result_assembly", time.perf_counter() - assembly_started
            )

        return parsing_res_lists, spotting_res_list, imgs_in_doc, request_counts

    def _assemble_parsing_results(
        self,
        blocks,
        batch_dict_by_pixel,
        id2pixel_key_map,
        drop_figures_set,
        vis_image_labels,
    ):
        parsing_res_lists = []
        spotting_res_list = []
        image_path_to_obj_map = {}
        table_blocks = []
        for i, blocks_for_img in enumerate(blocks):
            parsing_res_list = []
            spotting_res = {}
            for j, block in enumerate(blocks_for_img):
                block_img = block["img"]
                block_bbox = block["box"]
                block_label = block["label"]
                block_content = ""
                figure_token_map = {}
                if (i, j) in id2pixel_key_map:
                    pixel_key = id2pixel_key_map[(i, j)]
                    pixel_info = batch_dict_by_pixel[pixel_key]
                    curr_vlm_block_idx = pixel_info["curr_vlm_block_idx"]
                    if not (
                        curr_vlm_block_idx < len(pixel_info["vlm_block_ids"])
                        and pixel_info["vlm_block_ids"][curr_vlm_block_idx]
                        == (i, j)
                    ):
                        raise RuntimeError(
                            "Internal VLM result ordering mismatch while assembling "
                            f"page={i}, block={j}"
                        )
                    vl_rec_result = pixel_info["vlm_results"][curr_vlm_block_idx]
                    block_img4vl = pixel_info["images"][curr_vlm_block_idx]
                    figure_token_map = pixel_info["figure_token_maps"][
                        curr_vlm_block_idx
                    ]
                    curr_vlm_block_idx += 1
                    pixel_info["curr_vlm_block_idx"] = curr_vlm_block_idx
                    vl_rec_result["image"] = block_img4vl
                    result_str = vl_rec_result.get("result", "")
                    if result_str is None:
                        result_str = ""
                    min_count = 5000 if block_label == "table" else 50
                    result_str = truncate_repetitive_content(
                        result_str, min_count=min_count
                    )
                    if ("\\(" in result_str and "\\)" in result_str) or (
                        "\\[" in result_str and "\\]" in result_str
                    ):
                        result_str = result_str.replace("$", "")

                        result_str = (
                            result_str.replace("\\(", " $ ")
                            .replace("\\)", " $")
                            .replace("\\[\\[", "\\[")
                            .replace("\\]\\]", "\\]")
                            .replace("\\[", " $$ ")
                            .replace("\\]", " $$ ")
                        )
                        if block_label == "formula_number":
                            result_str = result_str.replace("$", "")
                    if block_label == "table":
                        html_str = convert_otsl_to_html(result_str)
                        if html_str != "":
                            result_str = html_str
                    block_content = result_str
                block_info = Block(
                    label=block_label,
                    bbox=block_bbox,
                    content=block_content,
                    group_id=block.get("group_id", None),
                    polygon_points=block.get("polygon_points", None),
                )
                if block_label == "table":
                    table_blocks.append(
                        {
                            "figure_token_map": figure_token_map,
                            "block": block_info,
                        }
                    )
                if block_label in vis_image_labels and block_img is not None:
                    from venusocr.pipeline.crop import construct_img_path

                    img_path = construct_img_path(block["label"], block["box"])
                    image_path_to_obj_map[img_path] = block_info
                    if img_path not in drop_figures_set:
                        import cv2

                        block_img = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)
                        from PIL import Image

                        block_info.image = {
                            "path": img_path,
                            "img": Image.fromarray(block_img),
                        }
                    else:
                        continue

                parsing_res_list.append(block_info)
                del block_info, block_img
            for blk_info in table_blocks:
                block = blk_info["block"]
                figure_token_map = blk_info["figure_token_map"]
                block.content = untokenize_figure_of_table(
                    block.content, figure_token_map, image_path_to_obj_map
                )
            parsing_res_lists.append(parsing_res_list)
            spotting_res_list.append(spotting_res)
            del parsing_res_list

        return parsing_res_lists, spotting_res_list

    # ------------------------------------------------------------------
    # per-batch orchestration
    # ------------------------------------------------------------------
    def _process_vlm(
        self,
        batch_data,
        image_arrays,
        layout_det_results,
        imgs_in_doc,
        model_settings,
        vlm_kwargs,
        cfg,
    ):
        """Recognize one page batch and build per-page results.

        Returns:
            A tuple ``(page_results, request_counts)`` where
            ``request_counts[i]`` is the number of VLM requests issued for
            page ``i``.
        """
        (
            parsing_res_lists,
            spotting_res_list,
            imgs_in_doc,
            request_counts,
        ) = self.get_layout_parsing_results(
            images=image_arrays,
            layout_det_results=layout_det_results,
            imgs_in_doc=imgs_in_doc,
            use_chart_recognition=model_settings["use_chart_recognition"],
            use_seal_recognition=model_settings["use_seal_recognition"],
            use_ocr_for_image_block=model_settings["use_ocr_for_image_block"],
            vlm_kwargs=vlm_kwargs,
            merge_layout_blocks=model_settings["merge_layout_blocks"],
            layout_shape_mode=cfg.layout_shape_mode,
        )

        page_results = []
        for (
            input_path,
            page_index,
            page_count,
            image,
            layout_det_res,
            parsing_res_list,
            spotting_res,
            imgs_in_doc_for_img,
        ) in zip(
            batch_data.input_paths,
            batch_data.page_indexes,
            batch_data.page_counts,
            image_arrays,
            layout_det_results,
            parsing_res_lists,
            spotting_res_list,
            imgs_in_doc,
        ):
            single_img_res = {
                "input_path": input_path,
                "page_index": page_index,
                "page_count": page_count,
                "width": image.shape[1],
                "height": image.shape[0],
                "doc_preprocessor_res": {"output_img": image},
                "layout_det_res": layout_det_res,
                "table_res_list": [],
                "parsing_res_list": parsing_res_list,
                "spotting_res": spotting_res,
                "imgs_in_doc": imgs_in_doc_for_img,
                "model_settings": model_settings,
            }
            page_results.append(PageResult(single_img_res))
        return page_results, request_counts
