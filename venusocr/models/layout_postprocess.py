# Copyright (c) 2026 VenusOCR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Backend-neutral post-processing for PP-DocLayoutV3 detections."""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np

from venusocr.pipeline.merge import calculate_polygon_overlap_ratio


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1 + 1) * max(0.0, y2 - y1 + 1)
    area_a = max(0.0, box_a[2] - box_a[0] + 1) * max(0.0, box_a[3] - box_a[1] + 1)
    area_b = max(0.0, box_b[2] - box_b[0] + 1) * max(0.0, box_b[3] - box_b[1] + 1)
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _layout_nms(
    rows: np.ndarray,
    same_class_iou: float = 0.6,
    cross_class_iou: float = 0.98,
):
    remaining = list(np.argsort(rows[:, 1])[::-1])
    selected = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [
            idx
            for idx in remaining
            if _iou(rows[current, 2:6], rows[idx, 2:6])
            < (same_class_iou if rows[current, 0] == rows[idx, 0] else cross_class_iou)
        ]
    return selected


def _is_contained(inner: np.ndarray, outer: np.ndarray) -> bool:
    inner_area = max(0.0, inner[4] - inner[2]) * max(0.0, inner[5] - inner[3])
    if inner_area <= 0:
        return False
    x1, y1 = max(inner[2], outer[2]), max(inner[3], outer[3])
    x2, y2 = min(inner[4], outer[4]), min(inner[5], outer[5])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / inner_area >= 0.9


def _containment_flags(
    rows: np.ndarray,
    preserved_class: int | None = None,
    category_id: int | None = None,
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror PaddleX's containment bookkeeping for layout post-processing."""
    contains_other = np.zeros(len(rows), dtype=int)
    contained_by_other = np.zeros(len(rows), dtype=int)
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i == j or (
                preserved_class is not None and rows[i][0] == preserved_class
            ):
                continue
            if category_id is not None and mode is not None:
                if (
                    mode == "large"
                    and rows[j][0] == category_id
                    and _is_contained(rows[i], rows[j])
                ):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
                if (
                    mode == "small"
                    and rows[i][0] == category_id
                    and _is_contained(rows[i], rows[j])
                ):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
            elif _is_contained(rows[i], rows[j]):
                contained_by_other[i] = 1
                contains_other[j] = 1
    return contains_other, contained_by_other


def _filter_containment(
    rows: np.ndarray,
    modes: Union[str, Mapping[int, str]],
    preserved_class: int | None,
) -> np.ndarray:
    if isinstance(modes, str):
        if modes == "union":
            return rows
        contains, contained = _containment_flags(rows, preserved_class)
        if modes == "large":
            return rows[contained == 0]
        return rows[(contains == 0) | (contained == 1)]

    keep = np.ones(len(rows), dtype=bool)
    for category_id, mode in modes.items():
        if mode == "union":
            continue
        contains, contained = _containment_flags(
            rows, preserved_class, int(category_id), mode
        )
        if mode == "large":
            keep &= contained == 0
        elif mode == "small":
            keep &= (contains == 0) | (contained == 1)
    return rows[keep]


def _unclip(rows: np.ndarray, ratio) -> np.ndarray:
    output = rows.copy()
    for row in output:
        value = ratio.get(int(row[0])) if isinstance(ratio, dict) else ratio
        if value is None:
            continue
        if isinstance(value, (int, float)):
            width_ratio = height_ratio = float(value)
        else:
            width_ratio, height_ratio = map(float, value)
        width, height = row[4] - row[2], row[5] - row[3]
        center_x, center_y = row[2] + width / 2, row[3] + height / 2
        row[2], row[4] = center_x - width * width_ratio / 2, center_x + width * width_ratio / 2
        row[3], row[5] = center_y - height * height_ratio / 2, center_y + height * height_ratio / 2
    return output


def _rect_from_box(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box).astype(np.int32)
    return np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )


def _polygon_to_quad(polygon: np.ndarray):
    if polygon is None or len(polygon) < 3:
        return None
    import cv2

    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    quad = cv2.boxPoints(cv2.minAreaRect(points))
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angles)]
    return np.roll(quad, -int(np.argmin(quad[:, 0] + quad[:, 1])), axis=0)


def _normalize_polygon(box, polygon, layout_shape_mode, previous_polygon=None):
    """Match PaddleX LayoutAnalysisProcess polygon normalization."""
    rect = _rect_from_box(box)
    if polygon is None:
        return rect
    polygon = np.asarray(polygon, dtype=np.float32)
    if polygon.ndim == 1:
        polygon = polygon.reshape(-1, 2)
    if len(polygon) < 4 or layout_shape_mode == "rect":
        return rect
    if layout_shape_mode == "poly":
        return polygon

    quad = _polygon_to_quad(polygon)
    if layout_shape_mode == "quad":
        return quad if quad is not None else rect
    if layout_shape_mode == "auto":
        if quad is not None:
            if calculate_polygon_overlap_ratio(rect, quad, mode="union") >= 0.95:
                return rect
            if (
                calculate_polygon_overlap_ratio(polygon, quad, mode="union") >= 0.8
                and (
                    previous_polygon is None
                    or calculate_polygon_overlap_ratio(
                        previous_polygon, rect, mode="small"
                    )
                    < 0.01
                )
            ):
                return quad
        return polygon
    raise ValueError("layout_shape_mode must be one of: rect, poly, quad, auto")


def format_layout_results(
    raw_results: Sequence[dict],
    id2label: Mapping[int, str],
    image_sizes: Sequence[tuple[int, int]],
    *,
    layout_nms: bool,
    layout_unclip_ratio,
    layout_merge_bboxes_mode,
    layout_shape_mode: str = "auto",
) -> list[list[dict]]:
    """Convert Transformers detections to the layout result schema used by VenusOCR."""
    labels = {int(key): value for key, value in id2label.items()}
    label2id = {value: key for key, value in labels.items()}
    results = []
    for raw, (width, height) in zip(raw_results, image_sizes):
        scores = raw["scores"].detach().cpu().numpy()
        class_ids = raw["labels"].detach().cpu().numpy()
        boxes = np.round(raw["boxes"].detach().cpu().numpy()).astype(int)
        order_tensor = raw.get("order_seq")
        orders = (
            order_tensor.detach().cpu().numpy()
            if order_tensor is not None
            else np.arange(1, len(scores) + 1)
        )
        if len(scores) == 0:
            results.append([])
            continue

        original_indices = np.arange(len(scores))
        rows = np.column_stack(
            (class_ids, scores, boxes, orders, original_indices)
        ).astype(float)
        if layout_nms:
            rows = rows[_layout_nms(rows)]

        image_class = label2id.get("image")
        if image_class is not None and len(rows) > 1:
            unfiltered_rows = rows
            max_ratio = 0.82 if width > height else 0.93
            page_area = width * height
            rows = np.asarray(
                [
                    row
                    for row in rows
                    if int(row[0]) != image_class
                    or max(0, min(width, row[4]) - max(0, row[2]))
                    * max(0, min(height, row[5]) - max(0, row[3]))
                    <= max_ratio * page_area
                ]
            )
            if rows.size == 0:
                rows = unfiltered_rows

        if layout_merge_bboxes_mode:
            rows = _filter_containment(
                rows, layout_merge_bboxes_mode, label2id.get("formula")
            )
        if len(rows) == 0:
            results.append([])
            continue
        # Hugging Face exposes the 7-column ordered form
        # [class, score, box(4), order_seq]. PaddleX applies argsort to that
        # form; its 8-column tie-break path additionally needs order_votes,
        # which the Transformers processor does not expose.
        rows = rows[np.argsort(rows[:, 6])]
        raw_polygons = raw.get("polygon_points") or []
        normalized_polygons = []
        for row in rows:
            original_index = int(row[7])
            polygon = (
                raw_polygons[original_index]
                if original_index < len(raw_polygons)
                else None
            )
            if polygon is not None:
                polygon = np.asarray(polygon, dtype=np.float32)
                polygon[:, 0] = np.clip(polygon[:, 0], 0, width)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, height)
            normalized_polygons.append(
                _normalize_polygon(
                    row[2:6],
                    polygon,
                    layout_shape_mode,
                    normalized_polygons[-1] if normalized_polygons else None,
                )
            )
        if layout_unclip_ratio:
            rows = _unclip(rows, layout_unclip_ratio)

        page_result = []
        for output_index, (row, polygon) in enumerate(
            zip(rows, normalized_polygons), start=1
        ):
            class_id = int(row[0])
            x1 = int(max(0, min(width, row[2])))
            y1 = int(max(0, min(height, row[3])))
            x2 = int(max(0, min(width, row[4])))
            y2 = int(max(0, min(height, row[5])))
            if x1 >= x2 or y1 >= y2:
                continue
            page_result.append(
                {
                    "cls_id": class_id,
                    "label": labels.get(class_id, f"class_{class_id}"),
                    "score": float(row[1]),
                    "coordinate": [x1, y1, x2, y2],
                    "order": output_index,
                    "polygon_points": polygon,
                }
            )
        results.append(page_result)
    return results
