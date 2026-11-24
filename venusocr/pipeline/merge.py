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

"""Layout block filtering and merging.

Vendored from ``paddlex.inference.pipelines.paddleocr_vl.uilts``
(``filter_overlap_boxes`` / ``merge_blocks`` / ``merge_images``) and
``paddlex.inference.pipelines.layout_parsing.utils`` (overlap-ratio
helpers). The numeric logic is identical to upstream.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image


def calculate_bbox_area(bbox):
    """Calculate bounding box area."""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area


def calculate_overlap_ratio(
    bbox1: Union[np.ndarray, list, tuple],
    bbox2: Union[np.ndarray, list, tuple],
    mode="union",
) -> float:
    """Calculate the overlap ratio between two bounding boxes.

    Args:
        bbox1: The first bounding box, format [x_min, y_min, x_max, y_max].
        bbox2: The second bounding box, format [x_min, y_min, x_max, y_max].
        mode: The mode of calculation: 'union', 'small', or 'large'.

    Returns:
        The overlap ratio value between the two bounding boxes.
    """
    bbox1 = np.array(bbox1, dtype=np.float64)
    bbox2 = np.array(bbox2, dtype=np.float64)

    x_min_inter = np.maximum(bbox1[0], bbox2[0])
    y_min_inter = np.maximum(bbox1[1], bbox2[1])
    x_max_inter = np.minimum(bbox1[2], bbox2[2])
    y_max_inter = np.minimum(bbox1[3], bbox2[3])

    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)

    inter_area = np.multiply(inter_width, inter_height, dtype=np.float64)

    bbox1_area = calculate_bbox_area(bbox1)
    bbox2_area = calculate_bbox_area(bbox2)

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = np.minimum(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = np.maximum(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area


def calculate_projection_overlap_ratio(
    bbox1: List[float],
    bbox2: List[float],
    direction: str = "horizontal",
    mode="union",
) -> float:
    """Calculate the IoU of lines between two bounding boxes.

    Args:
        bbox1: First bounding box [x_min, y_min, x_max, y_max].
        bbox2: Second bounding box [x_min, y_min, x_max, y_max].
        direction: direction of the projection, "horizontal" or "vertical".

    Returns:
        Line overlap ratio. Returns 0 if there is no overlap.
    """
    start_index, end_index = 1, 3
    if direction == "horizontal":
        start_index, end_index = 0, 2

    intersection_start = max(bbox1[start_index], bbox2[start_index])
    intersection_end = min(bbox1[end_index], bbox2[end_index])
    overlap = intersection_end - intersection_start
    if overlap <= 0:
        return 0

    if mode == "union":
        ref_width = max(bbox1[end_index], bbox2[end_index]) - min(
            bbox1[start_index], bbox2[start_index]
        )
    elif mode == "small":
        ref_width = min(
            bbox1[end_index] - bbox1[start_index], bbox2[end_index] - bbox2[start_index]
        )
    elif mode == "large":
        ref_width = max(
            bbox1[end_index] - bbox1[start_index], bbox2[end_index] - bbox2[start_index]
        )
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    return overlap / ref_width if ref_width > 0 else 0.0


def _make_valid(poly):
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def calculate_polygon_overlap_ratio(
    polygon1: List[Tuple[int, int]],
    polygon2: List[Tuple[int, int]],
    mode: str = "union",
) -> float:
    """Calculate the overlap ratio between two polygons."""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("Please install Shapely library.")
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    poly1 = _make_valid(poly1)
    poly2 = _make_valid(poly2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if mode == "union":
        return intersection / union
    elif mode == "small":
        small_area = min(poly1.area, poly2.area)
        return intersection / small_area
    elif mode == "large":
        large_area = max(poly1.area, poly2.area)
        return intersection / large_area
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _compute_pairwise_overlap_small(
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise overlap ratio (mode="small") and bbox areas.

    This only replaces the numeric overlap/area calculation. The caller keeps
    the original sequential dropping order because it affects inline formulas.
    """
    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 2]
    y2 = coords[:, 3]

    areas = np.abs((x2 - x1) * (y2 - y1))

    inter_x1 = np.maximum(x1[:, None], x1[None, :])
    inter_y1 = np.maximum(y1[:, None], y1[None, :])
    inter_x2 = np.minimum(x2[:, None], x2[None, :])
    inter_y2 = np.minimum(y2[:, None], y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    small_area = np.minimum(areas[:, None], areas[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.where(small_area > 0, inter_area / small_area, 0.0)
    return overlap, areas


def filter_overlap_boxes(
    layout_det_res: Dict[str, List[Dict]], layout_shape_mode: str
) -> Dict[str, List[Dict]]:
    """Remove overlapping boxes from layout detection results.

    Args:
        layout_det_res: Layout detection result dict containing a 'boxes' list.
        layout_shape_mode: Layout shape mode ("rect" / "auto" / ...).

    Returns:
        Filtered dict with overlapping boxes removed.
    """
    layout_det_res_filtered = deepcopy(layout_det_res)
    boxes = [
        box for box in layout_det_res_filtered["boxes"] if box["label"] != "reference"
    ]
    if not boxes:
        layout_det_res_filtered["boxes"] = boxes
        return layout_det_res_filtered

    coords = np.array([box["coordinate"] for box in boxes], dtype=np.float64)
    widths = coords[:, 2] - coords[:, 0]
    heights = coords[:, 3] - coords[:, 1]
    overlap_matrix, areas = _compute_pairwise_overlap_small(coords)
    dropped_indexes = set()

    for i in range(len(boxes)):
        if widths[i] < 6 or heights[i] < 6:
            dropped_indexes.add(i)
        for j in range(i + 1, len(boxes)):
            if i in dropped_indexes or j in dropped_indexes:
                continue
            overlap_ratio = overlap_matrix[i, j]
            if (
                boxes[i]["label"] == "inline_formula"
                or boxes[j]["label"] == "inline_formula"
            ):
                if overlap_ratio > 0.5:
                    if boxes[i]["label"] == "inline_formula":
                        dropped_indexes.add(i)
                    if boxes[j]["label"] == "inline_formula":
                        dropped_indexes.add(j)
                    continue
            if overlap_ratio > 0.7:
                if layout_shape_mode != "rect" and "polygon_points" in boxes[i]:
                    poly_overlap_ratio = calculate_polygon_overlap_ratio(
                        boxes[i]["polygon_points"], boxes[j]["polygon_points"], "small"
                    )
                    if poly_overlap_ratio < 0.7:
                        continue
                labels = {boxes[i]["label"], boxes[j]["label"]}
                if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                    if "table" not in labels or labels <= {
                        "table",
                        "image",
                        "seal",
                        "chart",
                    }:
                        continue
                if areas[i] >= areas[j]:
                    dropped_indexes.add(j)
                else:
                    dropped_indexes.add(i)
    layout_det_res_filtered["boxes"] = [
        box for idx, box in enumerate(boxes) if idx not in dropped_indexes
    ]
    return layout_det_res_filtered


def _to_pil_image(img):
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)


def _to_np_array(img):
    if isinstance(img, Image.Image):
        return np.array(img)
    return img


def calc_merged_wh(images):
    """Calculate width (max of all) and height (sum) for a vertical merge."""
    widths = [_to_pil_image(img).width for img in images]
    heights = [_to_pil_image(img).height for img in images]
    w = max(widths)
    h = sum(heights)
    return w, h


def merge_images(images, aligns="center", layout_shape_mode="auto"):
    """Merge images vertically with given alignment.

    Args:
        images: List of images (PIL images or numpy arrays) to merge.
        aligns: Alignment(s) for each merge step ('center', 'right', 'left').

    Returns:
        Merged image as a numpy array.
    """
    if not images:
        return None
    if len(images) == 1:
        return _to_np_array(images[0])
    if isinstance(aligns, str):
        aligns = [aligns] * (len(images) - 1)
    if len(aligns) != len(images) - 1:
        raise ValueError("The length of aligns must be len(images) - 1")

    # Convert all images to PIL once upfront (avoids repeated conversion)
    pil_images = [_to_pil_image(img) for img in images]

    # Simulate the iterative merge to compute each image's final x-offset,
    # without allocating intermediate canvases.
    x_offsets = [0] * len(pil_images)
    merged_w = pil_images[0].width

    for i in range(1, len(pil_images)):
        img2_w = pil_images[i].width
        step_w = max(merged_w, img2_w)
        align = aligns[i - 1]

        if align == "center":
            x1 = (step_w - merged_w) // 2
            x2 = (step_w - img2_w) // 2
        elif align == "right":
            x1 = step_w - merged_w
            x2 = step_w - img2_w
        else:  # left
            x1 = 0
            x2 = 0

        # All previously placed images shift by x1
        for k in range(i):
            x_offsets[k] += x1
        x_offsets[i] = x2
        merged_w = step_w

    # Single canvas allocation
    total_h = sum(img.height for img in pil_images)
    canvas = Image.new("RGB", (merged_w, total_h), (255, 255, 255))
    y_offset = 0
    for i, img in enumerate(pil_images):
        canvas.paste(img, (x_offsets[i], y_offset))
        y_offset += img.height

    return _to_np_array(canvas)


def merge_blocks(blocks, non_merge_labels, layout_shape_mode="auto"):
    """Merge adjacent text blocks, except those with labels in non_merge_labels.

    Args:
        blocks: List of block dicts (``img`` / ``box`` / ``label``).
        non_merge_labels: Block labels that should not be merged.

    Returns:
        List of processed (and possibly merged) blocks.
    """
    blocks_to_merge = []
    non_merge_blocks = {}
    for idx, block in enumerate(blocks):
        if block["label"] in non_merge_labels:
            non_merge_blocks[idx] = block
        else:
            blocks_to_merge.append((idx, block))

    merged_groups = []
    current_group = []
    current_indices = []
    current_aligns = []

    def is_aligned(a1, a2):
        return abs(a1 - a2) <= 5

    def get_alignment(block_bbox, prev_bbox):
        if is_aligned(block_bbox[0], prev_bbox[0]):
            return "left"
        elif is_aligned(block_bbox[2], prev_bbox[2]):
            return "right"
        else:
            return "center"

    def overlapwith_other_box(block_idx, prev_idx, blocks):
        prev_bbox = blocks[prev_idx]["box"]
        block_bbox = blocks[block_idx]["box"]
        x1 = min(prev_bbox[0], block_bbox[0])
        y1 = min(prev_bbox[1], block_bbox[1])
        x2 = max(prev_bbox[2], block_bbox[2])
        y2 = max(prev_bbox[3], block_bbox[3])
        min_box = [x1, y1, x2, y2]
        for idx, other_block in enumerate(blocks):
            if (
                idx in [block_idx, prev_idx]
                or other_block["label"] not in non_merge_labels
            ):
                continue
            other_bbox = other_block["box"]
            if calculate_overlap_ratio(min_box, other_bbox) > 0:
                return True
        return False

    for i, (idx, block) in enumerate(blocks_to_merge):
        if not current_group:
            current_group = [block]
            current_indices = [idx]
            current_aligns = []
            continue

        prev_idx, prev_block = blocks_to_merge[i - 1]
        prev_bbox = prev_block["box"]
        prev_label = prev_block["label"]
        block_bbox = block["box"]
        block_label = block["label"]

        iou_h = calculate_projection_overlap_ratio(block_bbox, prev_bbox, "horizontal")
        is_cross = (
            iou_h == 0
            and block_label == "text"
            and block_label == prev_label
            and block_bbox[0] > prev_bbox[2]
            and block_bbox[1] < prev_bbox[3]
            and block_bbox[0] - prev_bbox[2]
            < max(prev_bbox[2] - prev_bbox[0], block_bbox[2] - block_bbox[0]) * 0.3
        )
        is_updown_align = (
            iou_h > 0
            and block_label in ["text"]
            and block_label == prev_label
            and block_bbox[3] >= prev_bbox[1]
            and abs(block_bbox[1] - prev_bbox[3])
            < max(prev_bbox[3] - prev_bbox[1], block_bbox[3] - block_bbox[1]) * 0.5
            and (
                is_aligned(block_bbox[0], prev_bbox[0])
                ^ is_aligned(block_bbox[2], prev_bbox[2])
            )
            and overlapwith_other_box(idx, prev_idx, blocks)
        )
        if is_cross:
            align_mode = "center"
        elif is_updown_align:
            align_mode = get_alignment(block_bbox, prev_bbox)
        else:
            align_mode = None

        if is_cross or is_updown_align:
            current_group.append(block)
            current_indices.append(idx)
            current_aligns.append(align_mode)
        else:
            merged_groups.append((current_indices, current_aligns))
            current_group = [block]
            current_indices = [idx]
            current_aligns = []
    if current_group:
        merged_groups.append((current_indices, current_aligns))

    group_ranges = []
    for group_indices, aligns in merged_groups:
        start, end = min(group_indices), max(group_indices)
        group_ranges.append((start, end, group_indices, aligns))

    result_blocks = []
    used_indices = set()
    idx = 0
    while idx < len(blocks):
        group_found = False
        for start, end, group_indices, aligns in group_ranges:
            if idx == start and all(i not in used_indices for i in group_indices):
                group_found = True
                imgs = [blocks[i]["img"] for i in group_indices]
                merge_aligns = aligns if aligns else []
                w, h = calc_merged_wh(imgs)
                aspect_ratio = h / w if w != 0 else float("inf")
                if aspect_ratio >= 3:
                    for j, block_idx in enumerate(group_indices):
                        block = blocks[block_idx].copy()
                        block["img"] = blocks[block_idx]["img"]
                        block["merge_aligns"] = None
                        result_blocks.append(block)
                        used_indices.add(block_idx)
                else:
                    merged_img = merge_images(imgs, merge_aligns, layout_shape_mode)
                    for j, block_idx in enumerate(group_indices):
                        block = blocks[block_idx].copy()
                        block["img"] = merged_img if j == 0 else None
                        block["merge_aligns"] = merge_aligns if j == 0 else None
                        block["group_id"] = group_indices[0]
                        result_blocks.append(block)
                        used_indices.add(block_idx)
                insert_list = []
                for n_idx in range(start + 1, end):
                    if n_idx in non_merge_blocks:
                        insert_list.append(n_idx)
                for n_idx in insert_list:
                    result_blocks.append(non_merge_blocks[n_idx])
                    used_indices.add(n_idx)
                idx = end + 1
                break
        if group_found:
            continue
        if idx in non_merge_blocks and idx not in used_indices:
            result_blocks.append(non_merge_blocks[idx])
            used_indices.add(idx)
        idx += 1
    return result_blocks
