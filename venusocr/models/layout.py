# Copyright (c) 2026 VenusOCR. All Rights Reserved.
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

"""Selectable PP-DocLayoutV3 inference backends.

Both backends run locally and produce the same result schema. PaddleX is the
PaddleOCR 3.6.0 compatibility backend; Transformers is an optional backend
for safetensors checkpoints and does not import Paddle or PaddleX.

The post-processing defaults below are pinned to the PP-DocLayoutV3 settings
used by the project's PaddleOCR 3.6.0 evaluation script.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from venusocr.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_LAYOUT_MODEL_NAME = "PP-DocLayoutV3"
DEFAULT_LAYOUT_BATCH_SIZE = 8

# Canonical class order from the native PP-DocLayoutV3 ``inference.yml``.
# Converted Hugging Face checkpoints may collapse duplicate semantic names
# (for example display/inline formula into ``formula``), but the class ids are
# still distinct and drive downstream prompting, merging, and Markdown output.
PP_DOCLAYOUT_V3_ID2LABEL = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "display_formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer_image",
    9: "footer",
    10: "footnote",
    11: "formula_number",
    12: "header_image",
    13: "header",
    14: "image",
    15: "inline_formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "vertical_text",
    24: "vision_footnote",
}

# PaddleOCR-VL 1.6 uses one score threshold for every PP-DocLayoutV3 class.
DEFAULT_LAYOUT_THRESHOLD = 0.3

# Per-class overlapping-box filtering mode (cls_id -> "union" / "large").
DEFAULT_LAYOUT_MERGE_BBOXES_MODE = {
    0: "union",   # abstract
    1: "union",   # algorithm
    2: "union",   # aside_text
    3: "large",   # chart
    4: "union",   # content
    5: "large",   # display_formula
    6: "large",   # doc_title
    7: "union",   # figure_title
    8: "union",   # footer
    9: "union",   # footer
    10: "union",  # footnote
    11: "union",  # formula_number
    12: "union",  # header
    13: "union",  # header
    14: "union",  # image
    15: "large",  # inline_formula
    16: "union",  # number
    17: "large",  # paragraph_title
    18: "union",  # reference
    19: "union",  # reference_content
    20: "union",  # seal
    21: "union",  # table
    22: "union",  # text
    23: "union",  # text
    24: "union",  # vision_footnote
}

DEFAULT_LAYOUT_NMS = True
DEFAULT_LAYOUT_UNCLIP_RATIO = [1.0, 1.0]


class PaddleXLayoutDetector:
    """PP-DocLayoutV3 layout region detector with reading order.

    A thin adapter over the paddlex model runtime
    (``paddlex.inference.models.create_predictor``). The default
    post-processing parameters are identical to the official PaddleOCR-VL
    pipeline configuration.

    Args:
        model_name: Layout model name (``PP-DocLayoutV3`` by default).
        model_dir: Local model directory. ``None`` downloads the official
            weights via the paddlex runtime.
        device: Device for local inference, e.g. ``"gpu:0"`` or ``"cpu"``.
        threshold: Score threshold (float or per-class dict). ``None`` uses
            the PaddleOCR-VL 1.6 default of 0.3.
        layout_nms: Whether to use layout-aware NMS. ``None`` -> ``True``.
        layout_unclip_ratio: Box expansion ratio. ``None`` -> ``[1.0, 1.0]``.
        layout_merge_bboxes_mode: Overlapping-box filtering mode (str or
            per-class dict). ``None`` uses the official per-class defaults.
        batch_size: Batch size of the underlying predictor (8 by default,
            same as the official pipeline config).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LAYOUT_MODEL_NAME,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        *,
        threshold: Optional[Union[float, Dict[int, float]]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, List[float]]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]] = None,
        batch_size: int = DEFAULT_LAYOUT_BATCH_SIZE,
    ) -> None:
        try:
            from paddlex.inference.models import create_predictor
        except ImportError as exc:
            raise ImportError(
                "The paddlex layout backend requires PaddleX 3.6.0 and a "
                "matching PaddlePaddle installation. Install it with "
                "`pip install venusocr[paddlex]`."
            ) from exc

        self.model_name = model_name
        logger.info(
            "Creating layout detector: %s (dir=%s, device=%s)",
            model_name,
            model_dir or "<auto-download>",
            device,
        )
        self._predictor = create_predictor(
            model_name=model_name,
            model_dir=model_dir,
            device=device,
            batch_size=batch_size,
            threshold=threshold if threshold is not None else DEFAULT_LAYOUT_THRESHOLD,
            layout_nms=layout_nms if layout_nms is not None else DEFAULT_LAYOUT_NMS,
            layout_unclip_ratio=(
                layout_unclip_ratio
                if layout_unclip_ratio is not None
                else DEFAULT_LAYOUT_UNCLIP_RATIO
            ),
            layout_merge_bboxes_mode=(
                layout_merge_bboxes_mode
                if layout_merge_bboxes_mode is not None
                else DEFAULT_LAYOUT_MERGE_BBOXES_MODE
            ),
        )

    def predict(
        self,
        images,
        *,
        threshold: Optional[Union[float, Dict[int, float]]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, List[float]]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]] = None,
        layout_shape_mode: str = "auto",
    ) -> List[Any]:
        """Run layout detection on a list of BGR page images.

        Per-call parameters default to ``None``, in which case the
        construction-time (official) defaults apply.

        Returns:
            A list of layout detection results (dict-like objects with a
            ``"boxes"`` entry per page).
        """
        return list(
            self._predictor(
                images,
                threshold=threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                layout_shape_mode=layout_shape_mode,
                filter_overlap_boxes=False,
            )
        )

    def close(self) -> None:
        """Release references held by the PaddleX predictor."""
        self._predictor = None


class _TransformersLayoutResult(dict):
    """Dict-compatible result with the visualization interface used by PageResult."""

    def __init__(self, boxes, image):
        super().__init__({"boxes": boxes})
        self._image = image

    @property
    def img(self):
        import cv2

        canvas = self._image.copy()
        for item in self["boxes"]:
            x1, y1, x2, y2 = map(int, item["coordinate"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(
                canvas,
                item["label"],
                (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 120, 0),
                1,
                cv2.LINE_AA,
            )
        return {"res": canvas}


class TransformersLayoutDetector:
    """PP-DocLayoutV3 backend implemented with Hugging Face Transformers.

    This backend removes the Paddle/PaddleX runtime dependency. Its output
    schema matches :class:`PaddleXLayoutDetector`, while exact numeric parity
    remains dependent on checkpoint conversion and backend kernels.
    """

    DEFAULT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"

    def __init__(
        self,
        model_name: str = DEFAULT_LAYOUT_MODEL_NAME,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        *,
        threshold: Optional[Union[float, Dict[int, float]]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, List[float]]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]] = None,
        batch_size: int = DEFAULT_LAYOUT_BATCH_SIZE,
    ) -> None:
        try:
            import torch
            from transformers import (
                PPDocLayoutV3ForObjectDetection,
                PPDocLayoutV3ImageProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "The transformers layout backend requires torch and transformers. "
                "Install them with `pip install venusocr[transformers]`."
            ) from exc

        source = model_dir or self.DEFAULT_MODEL_ID
        self.model_name = model_name
        self.batch_size = batch_size
        self.threshold = threshold if threshold is not None else DEFAULT_LAYOUT_THRESHOLD
        self.layout_nms = layout_nms if layout_nms is not None else DEFAULT_LAYOUT_NMS
        self.layout_unclip_ratio = (
            layout_unclip_ratio
            if layout_unclip_ratio is not None
            else DEFAULT_LAYOUT_UNCLIP_RATIO
        )
        self.layout_merge_bboxes_mode = (
            layout_merge_bboxes_mode
            if layout_merge_bboxes_mode is not None
            else DEFAULT_LAYOUT_MERGE_BBOXES_MODE
        )
        self._torch = torch
        self._device = self._normalize_device(device, torch.cuda.is_available())
        logger.info("Creating Transformers layout detector: %s on %s", source, self._device)
        self._processor = PPDocLayoutV3ImageProcessor.from_pretrained(source)
        self._model = PPDocLayoutV3ForObjectDetection.from_pretrained(source)
        self._model.eval().to(self._device)
        self._id2label = PP_DOCLAYOUT_V3_ID2LABEL.copy()

    @staticmethod
    def _normalize_device(device: Optional[str], cuda_available: bool) -> str:
        if device is None:
            return "cuda:0" if cuda_available else "cpu"
        if device == "gpu":
            return "cuda"
        if device.startswith("gpu:"):
            return "cuda:" + device.split(":", 1)[1]
        return device

    def predict(
        self,
        images,
        *,
        threshold: Optional[Union[float, Dict[int, float]]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, List[float]]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]] = None,
        layout_shape_mode: str = "auto",
    ) -> List[Any]:
        from PIL import Image

        from venusocr.models.layout_postprocess import format_layout_results

        pil_images = [Image.fromarray(image[:, :, ::-1]).convert("RGB") for image in images]
        results = []
        effective_threshold = threshold if threshold is not None else self.threshold
        threshold_map = effective_threshold if isinstance(effective_threshold, dict) else None
        # Match PaddleX's Transformers detector: a scalar is applied by the HF
        # processor, while a category map first retains all candidates and is
        # applied locally per class.
        pre_threshold = 0.0 if threshold_map else float(effective_threshold)
        for start in range(0, len(pil_images), self.batch_size):
            chunk = pil_images[start : start + self.batch_size]
            inputs = self._processor(images=chunk, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            target_sizes = self._torch.tensor(
                [image.size[::-1] for image in chunk], device=self._device
            )
            raw = self._processor.post_process_object_detection(
                outputs, threshold=pre_threshold, target_sizes=target_sizes
            )
            for item in raw:
                if threshold_map:
                    thresholds = self._torch.tensor(
                        [
                            threshold_map.get(int(label), 0.5)
                            for label in item["labels"]
                        ],
                        device=item["scores"].device,
                    )
                else:
                    thresholds = self._torch.full_like(
                        item["scores"], float(effective_threshold)
                    )
                keep = (item["scores"] > thresholds) & (item["labels"] > -1)
                keep_list = keep.detach().cpu().tolist()
                for key in ("scores", "labels", "boxes", "order_seq"):
                    if key in item:
                        item[key] = item[key][keep]
                if "polygon_points" in item:
                    item["polygon_points"] = [
                        polygon
                        for polygon, selected in zip(item["polygon_points"], keep_list)
                        if selected
                    ]
            formatted = format_layout_results(
                raw,
                self._id2label,
                [image.size for image in chunk],
                layout_nms=self.layout_nms if layout_nms is None else layout_nms,
                layout_unclip_ratio=(
                    self.layout_unclip_ratio
                    if layout_unclip_ratio is None
                    else layout_unclip_ratio
                ),
                layout_merge_bboxes_mode=(
                    self.layout_merge_bboxes_mode
                    if layout_merge_bboxes_mode is None
                    else layout_merge_bboxes_mode
                ),
                layout_shape_mode=layout_shape_mode,
            )
            for boxes, original in zip(formatted, images[start : start + len(chunk)]):
                results.append(_TransformersLayoutResult(boxes, original))
        return results

    def close(self) -> None:
        """Release model memory."""
        if getattr(self, "_model", None) is not None:
            self._model = None
            self._processor = None
            if self._device.startswith("cuda"):
                self._torch.cuda.empty_cache()


def create_layout_detector(*, backend: str = "paddlex", **kwargs):
    """Create a layout detector without exposing backend-specific classes."""
    if backend == "paddlex":
        return PaddleXLayoutDetector(**kwargs)
    if backend == "transformers":
        return TransformersLayoutDetector(**kwargs)
    raise ValueError(f"Unsupported layout backend: {backend!r}")


# Backward-compatible internal alias. New code should use create_layout_detector().
LayoutDetector = PaddleXLayoutDetector
