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

"""Region cropping utilities for the parsing pipeline.

Vendored from ``paddlex.inference.pipelines.components``
(``CropByBoxes``), ``paddlex.inference.pipelines.layout_parsing.utils``
(``gather_imgs`` / ``construct_img_path``) and
``paddlex.inference.pipelines.paddleocr_vl.uilts`` (``crop_margin``).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from PIL import Image

# Visual-only labels collected as document images. This extends PaddleX's
# image-label set because VenusOCR deliberately does not recognize chart,
# seal, or spotting regions with its text/formula/table VLM.
IMAGE_BLOCK_LABELS = [
    "image",
    "figure",
    "header_image",
    "footer_image",
    "chart",
    "seal",
    "spotting",
]


class CropByBoxes:
    """Crop image regions by layout boxes."""

    def __call__(
        self, img: np.ndarray, boxes: List[dict], layout_shape_mode="auto"
    ) -> List[dict]:
        """Crop ``img`` by ``boxes``.

        Args:
            img: The input image as a numpy array (BGR).
            boxes: A list of dicts with ``cls_id`` / ``coordinate`` /
                ``label`` (and optionally ``polygon_points``).
            layout_shape_mode: When not ``"rect"``, polygon regions outside
                the box polygon are whitened.

        Returns:
            A list of dicts with ``img`` / ``box`` / ``label`` (and
            optionally ``polygon_points``).
        """
        import cv2

        output_list = []
        for box_info in boxes:
            label_id = box_info["cls_id"]
            box = box_info["coordinate"]
            label = box_info.get("label", label_id)
            xmin, ymin, xmax, ymax = [int(i) for i in box]
            img_crop = img[ymin:ymax, xmin:xmax].copy()
            out_info = {"img": img_crop, "box": box, "label": label}
            if layout_shape_mode != "rect" and "polygon_points" in box_info:
                mask = np.zeros(img_crop.shape[:2], dtype=np.int32)
                polygon = np.array(box_info["polygon_points"], dtype=np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                if polygon is not None and len(polygon) > 0:
                    polygon = polygon - np.array([xmin, ymin])
                cv2.fillPoly(mask, [polygon], 1)
                mask = mask.astype(bool)
                img_crop[~mask] = 255
                out_info["img"] = img_crop
                out_info["polygon_points"] = box_info["polygon_points"]

            output_list.append(out_info)
        return output_list


def construct_img_path(label, box):
    """Construct the relative artifact path for a cropped region image."""
    x_min, y_min, x_max, y_max = list(map(int, box))
    return f"imgs/img_in_{label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"


def _crop_image_region_for_pil(original_img, x_min, y_min, x_max, y_max):
    """Crop `original_img` to [y_min:y_max, x_min:x_max], clamped to image bounds."""
    crop = original_img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    if crop.ndim == 2:
        return np.ascontiguousarray(crop)
    if crop.shape[-1] >= 3:
        return np.ascontiguousarray(crop[..., :3][..., ::-1])
    return np.ascontiguousarray(crop)


def gather_imgs(original_img, layout_det_objs):
    """Collect image-label regions of a page as PIL images.

    Args:
        original_img: The page image (BGR numpy array).
        layout_det_objs: Layout detection boxes of the page.

    Returns:
        A list of ``{"path", "img", "label", "coordinate", "score"}`` dicts.
    """
    imgs_in_doc = []
    h, w = int(original_img.shape[0]), int(original_img.shape[1])
    for det_obj in layout_det_objs:
        if det_obj["label"] in IMAGE_BLOCK_LABELS:
            label = det_obj["label"]
            x_min, y_min, x_max, y_max = list(map(int, det_obj["coordinate"]))
            x_min = max(0, min(x_min, w))
            x_max = max(0, min(x_max, w))
            y_min = max(0, min(y_min, h))
            y_max = max(0, min(y_max, h))
            if x_max <= x_min or y_max <= y_min:
                continue
            img_path = construct_img_path(label, det_obj["coordinate"])
            arr_for_pil = _crop_image_region_for_pil(
                original_img, x_min, y_min, x_max, y_max
            )
            if arr_for_pil is None:
                continue
            img = Image.fromarray(arr_for_pil)
            imgs_in_doc.append(
                {
                    "path": img_path,
                    "img": img,
                    "label": label,
                    "coordinate": (x_min, y_min, x_max, y_max),
                    "score": det_obj["score"],
                }
            )
    return imgs_in_doc


def crop_margin(img):
    """Crop the blank margin around the content of an image."""
    import cv2

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    max_val = int(gray.max())
    min_val = int(gray.min())

    if max_val == min_val:
        return img

    # Use a LUT (lookup table) to normalize: avoids intermediate float64 array.
    lut = np.zeros(256, dtype=np.uint8)
    for v in range(min_val, max_val + 1):
        lut[v] = int((v - min_val) / (max_val - min_val) * 255)
    data = cv2.LUT(gray, lut)

    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y : y + h, x : x + w]

    return cropped
