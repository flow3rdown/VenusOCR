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

"""Page-level parsing result, serialization and artifact persistence."""

from __future__ import annotations

import copy
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from venusocr.results.markdown import (
    MarkdownConverter,
    build_handle_funcs_dict,
    format_centered_by_html,
    format_chart2html_table,
    format_image_plain,
    format_image_scaled_by_html,
    format_text_plain,
    format_table_center,
    simplify_table,
)
from venusocr.results.writers import write_image, write_json, write_markdown

SKIP_ORDER_LABELS = [
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
]

_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}


def _format_data(obj):
    """Format data into a JSON-serializable structure (vendored from paddlex)."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [_format_data(item) for item in obj.tolist()]
    elif isinstance(obj, Path):
        return obj.as_posix()
    elif isinstance(obj, dict):
        return dict({k: _format_data(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return [_format_data(i) for i in obj]
    else:
        return obj


def _is_json_file(file_path) -> bool:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type is not None and mime_type == "application/json"


def _is_image_file(file_path) -> bool:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type is not None and mime_type.startswith("image/")


def _is_markdown_file(file_path) -> bool:
    _, ext = os.path.splitext(str(file_path))
    if ext.lower() in _MARKDOWN_EXTENSIONS:
        return True
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type == "text/markdown"


class Block:
    """A parsed content block (label + bbox + recognized content)."""

    def __init__(
        self,
        label,
        bbox,
        content="",
        group_id=None,
        polygon_points=None,
        global_block_id=None,
        global_group_id=None,
    ) -> None:
        """Initialize a Block object.

        Args:
            label (str): Label assigned to the block.
            bbox (list): Bounding box coordinates of the block.
            content (str, optional): Content of the block. Defaults to an empty string.
        """
        self.label = label
        self.bbox = list(map(int, bbox))
        self.content = content
        self.image = None
        self.polygon_points = polygon_points
        self.group_id = group_id
        self.global_block_id = global_block_id
        self.global_group_id = global_group_id

    def __str__(self) -> str:
        return (
            f"\n\n#################\nlabel:\t{self.label}\nbbox:\t{self.bbox}\n"
            f"content:\t{self.content}\n#################"
        )

    def __repr__(self) -> str:
        return self.__str__()


class PageResult(dict):
    """Parsing result of a single page (dict-backed, like the upstream result).

    Keys follow the upstream ``PaddleOCRVLResult`` payload: ``input_path``,
    ``page_index``, ``page_count``, ``width``, ``height``,
    ``doc_preprocessor_res``, ``layout_det_res``, ``table_res_list``,
    ``parsing_res_list``, ``spotting_res``, ``imgs_in_doc``, ``model_settings``.
    """

    MARKDOWN_SAVE_KEYS = ["markdown_texts"]

    def __init__(self, data: dict) -> None:
        super().__init__(data)
        markdown_ignore_labels = self["model_settings"].get("markdown_ignore_labels", [])
        self.skip_order_labels = [
            label for label in SKIP_ORDER_LABELS.copy() + markdown_ignore_labels
        ]
        self._rand_fn: Optional[str] = None

    # ------------------------------------------------------------------
    # naming (vendored from BaseResult / BaseCVResult)
    # ------------------------------------------------------------------
    def _get_input_fn(self) -> str:
        if self.get("input_path", None) is None:
            if self._rand_fn:
                return self._rand_fn
            timestamp = int(time.time())
            random_number = random.randint(1000, 9999)
            self._rand_fn = Path(f"{timestamp}_{random_number}").name
            return self._rand_fn
        if isinstance(self["input_path"], list):
            input_path = self["input_path"][0]
        else:
            input_path = self["input_path"]
        fn = Path(input_path).name
        if (page_idx := self.get("page_index", None)) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            if isinstance(page_idx, list):
                page_idx = f"{page_idx[0]}-{page_idx[-1]}"
            fn = f"{stem}_{page_idx}{suffix}"
        return fn

    def _page_image_width(self) -> int:
        """Return the page image width, unwrapping list if necessary."""
        w = self["width"]
        return w[0] if isinstance(w, list) else w

    # ------------------------------------------------------------------
    # representations
    # ------------------------------------------------------------------
    @property
    def json(self) -> Dict[str, Dict[str, Any]]:
        """JSON representation of the result."""
        return self._to_json()

    @property
    def markdown(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """Markdown representation of the result."""
        return self._to_markdown()

    @property
    def img(self) -> Dict[str, np.ndarray]:
        """Image (visualization) representation of the result."""
        return self._to_img()

    @property
    def str(self) -> Dict[str, str]:
        """String representation of the result."""
        return self._to_str()

    def _to_img(self) -> Dict[str, np.ndarray]:
        """Convert the parsing result to visualization images."""
        res_img_dict = {}
        model_settings = self["model_settings"]
        if model_settings["use_layout_detection"]:
            layout_det_res = self["layout_det_res"]
            if hasattr(layout_det_res, "img"):
                res_img_dict["layout_det_res"] = layout_det_res.img["res"]
            elif isinstance(layout_det_res, list):
                for idx, layout_res in enumerate(layout_det_res):
                    if hasattr(layout_res, "img"):
                        res_img_dict[f"layout_det_res_{idx}"] = layout_res.img["res"]
        return res_img_dict

    def _to_str(self, *args, **kwargs) -> Dict[str, str]:
        """Convert the result to a printable dictionary."""
        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        data["page_count"] = self["page_count"]
        data["width"] = self["width"]
        data["height"] = self["height"]
        data["model_settings"] = self["model_settings"]
        if self["model_settings"]["use_layout_detection"]:
            layout_det_res = self["layout_det_res"]
            if hasattr(layout_det_res, "str"):
                data["layout_det_res"] = layout_det_res.str["res"]
            else:
                data["layout_det_res"] = layout_det_res
        data["parsing_res_list"] = [
            {
                "block_label": block.label,
                "block_content": block.content,
                "block_bbox": block.bbox,
            }
            for block in self["parsing_res_list"]
        ]
        return {"res": data}

    def _build_handle_funcs_dict(self, pretty=True):
        """Build label-to-handler mapping for content formatting."""
        use_ocr_for_image_block = self["model_settings"].get(
            "use_ocr_for_image_block", False
        )
        use_seal_recognition = self["model_settings"].get("use_seal_recognition", False)
        original_image_width = self._page_image_width()

        if pretty:
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=use_ocr_for_image_block,
                ),
                collapse_newlines=not use_ocr_for_image_block,
            )
            format_seal_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=use_seal_recognition,
                ),
                collapse_newlines=not use_seal_recognition,
            )
        else:
            format_text_func = lambda block: block.content
            format_image_func = lambda block: format_image_plain(
                block, show_ocr_content=use_ocr_for_image_block
            )
            format_seal_func = lambda block: format_image_plain(
                block, show_ocr_content=use_seal_recognition
            )

        format_chart_func = (
            format_chart2html_table
            if self["model_settings"].get("use_chart_recognition", False)
            else format_image_func
        )

        if not self["model_settings"].get("use_layout_detection", False):
            format_seal_func = format_text_func

        if pretty:
            format_table_func = lambda block: "\n" + format_table_center(block)
        else:
            format_table_func = lambda block: simplify_table("\n" + block.content)

        format_formula_func = lambda block: block.content

        handle_funcs_dict = build_handle_funcs_dict(
            text_func=format_text_func,
            image_func=format_image_func,
            chart_func=format_chart_func,
            table_func=format_table_func,
            formula_func=format_formula_func,
            seal_func=format_seal_func,
        )
        for label in self["model_settings"].get("markdown_ignore_labels", []):
            handle_funcs_dict.pop(label, None)
        return handle_funcs_dict

    def _to_json(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Convert the result to a JSON-serializable dictionary."""
        _keep_img = kwargs.pop("keep_img", False)

        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        data["page_count"] = self["page_count"]
        data["width"] = self["width"]
        data["height"] = self["height"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        if self["model_settings"].get("format_block_content", False):
            handle_funcs_dict = self._build_handle_funcs_dict(pretty=True)

        parsing_res_list = self["parsing_res_list"]
        parsing_res_list_json = []
        order_index = 1
        for idx, parsing_res in enumerate(parsing_res_list):
            label = parsing_res.label
            if label not in self.skip_order_labels:
                order = order_index
                order_index += 1
            else:
                order = None
            res_dict = {
                "block_label": parsing_res.label,
                "block_content": parsing_res.content,
                "block_bbox": parsing_res.bbox,
                "block_id": idx,
                "block_order": order,
                "group_id": (
                    parsing_res.group_id if parsing_res.group_id is not None else idx
                ),
            }
            if (
                hasattr(parsing_res, "global_block_id")
                and parsing_res.global_block_id is not None
            ):
                res_dict["global_block_id"] = parsing_res.global_block_id
            if (
                hasattr(parsing_res, "global_group_id")
                and parsing_res.global_group_id is not None
            ):
                res_dict["global_group_id"] = parsing_res.global_group_id
            if parsing_res.polygon_points is not None:
                res_dict["block_polygon_points"] = parsing_res.polygon_points

            if _keep_img and parsing_res.image is not None:
                res_dict["image"] = parsing_res.image

            if self["model_settings"].get("format_block_content", False):
                if handle_funcs_dict.get(parsing_res.label):
                    res_dict["block_content"] = handle_funcs_dict[parsing_res.label](
                        parsing_res
                    )
                else:
                    res_dict["block_content"] = parsing_res.content

            parsing_res_list_json.append(res_dict)
        data["parsing_res_list"] = parsing_res_list_json
        if self.get("spotting_res"):
            if isinstance(self["spotting_res"], list):
                data["spotting_res"] = [res for res in self["spotting_res"]]
            else:
                data["spotting_res"] = self["spotting_res"]
        if self["model_settings"]["use_layout_detection"]:
            layout_det_res = self["layout_det_res"]
            if hasattr(layout_det_res, "json"):
                data["layout_det_res"] = layout_det_res.json["res"]
            elif isinstance(layout_det_res, list):
                data["layout_det_res"] = [
                    res.json["res"] if hasattr(res, "json") else res
                    for res in layout_det_res
                ]
            else:
                data["layout_det_res"] = layout_det_res

        return {"res": _format_data(copy.deepcopy(data))}

    def _to_markdown(self, pretty=True, show_formula_number=False) -> dict:
        """Convert the parsing result to markdown text + images.

        Args:
            pretty (Optional[bool]): whether to pretty markdown by HTML, default by True.
            show_formula_number (bool): whether to show formula numbers.

        Returns:
            dict: Markdown information with text and images.
        """
        handle_funcs_dict = self._build_handle_funcs_dict(pretty=pretty)

        result = MarkdownConverter.convert(
            self["parsing_res_list"],
            handle_funcs_dict=handle_funcs_dict,
            show_formula_number=show_formula_number,
            imgs_in_doc=self["imgs_in_doc"],
        )
        result["page_index"] = self["page_index"]
        result["input_path"] = self["input_path"]
        return result

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save_to_json(
        self,
        save_path: str,
        indent: int = 4,
        ensure_ascii: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Save the JSON representation to ``save_path`` (a directory or file)."""
        json_data = self._to_json()
        if not _is_json_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in json_data:
                write_json(
                    base_save_path / f"{stem}_{key}.json",
                    json_data[key],
                    indent=indent,
                    ensure_ascii=ensure_ascii,
                )
        else:
            write_json(
                save_path,
                json_data[list(json_data.keys())[0]],
                indent=indent,
                ensure_ascii=ensure_ascii,
            )

    def save_to_img(self, save_path: str, *args, **kwargs) -> None:
        """Save the visualization images to ``save_path`` (a directory or file)."""
        img = self._to_img()
        if not _is_image_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_image_file(fn) else ".png"
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in img:
                write_image(base_save_path / f"{stem}_{key}{suffix}", img[key])
        else:
            write_image(save_path, img[list(img.keys())[0]])

    def save_to_markdown(
        self, save_path, pretty=True, show_formula_number=False, *args, **kwargs
    ) -> None:
        """Save the markdown text (and referenced images) to ``save_path``."""
        if not _is_markdown_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_markdown_file(fn) else ".md"
            stem = fn.stem
            base_save_path = Path(save_path)
            save_path = base_save_path / f"{stem}{suffix}"
        data = self._to_markdown(pretty=pretty, show_formula_number=show_formula_number)
        save_path = Path(save_path)
        for key, value in data.items():
            if key in self.MARKDOWN_SAVE_KEYS:
                write_markdown(save_path, value)
            if isinstance(value, dict):
                for img_path, img_data in value.items():
                    if img_data:
                        write_image(save_path.parent / img_path, img_data)
