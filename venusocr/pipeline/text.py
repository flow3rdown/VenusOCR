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

"""Text-level post-processing: repetition truncation and spotting handling.

Vendored from ``paddlex.inference.pipelines.paddleocr_vl.uilts``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image


def find_shortest_repeating_substring(s: str) -> Union[str, None]:
    """Find the shortest substring that repeats to form the entire string."""
    n = len(s)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            substring = s[:i]
            if substring * (n // i) == s:
                return substring
    return None


def find_repeating_suffix(
    s: str, min_len: int = 8, min_repeats: int = 5
) -> Union[Tuple[str, str, int], None]:
    """Detect if string ends with a repeating phrase."""
    for i in range(len(s) // (min_repeats), min_len - 1, -1):
        unit = s[-i:]
        if s.endswith(unit * min_repeats):
            count = 0
            temp_s = s
            while temp_s.endswith(unit):
                temp_s = temp_s[:-i]
                count += 1
            start_index = len(s) - (count * i)
            return s[:start_index], unit, count
    return None


def truncate_repetitive_content(
    content: str,
    line_threshold: int = 10,
    char_threshold: int = 10,
    min_len: int = 10,
    min_count: int = 3000,
) -> str:
    """Detect and truncate character-, phrase-, or line-level repetition."""
    if len(content) < min_count:
        return content

    stripped_content = content.strip()
    if not stripped_content:
        return content

    # Priority 1: Phrase-level suffix repetition in long single lines.
    if "\n" not in stripped_content and len(stripped_content) > 100:
        suffix_match = find_repeating_suffix(stripped_content, min_len=8, min_repeats=5)
        if suffix_match:
            prefix, repeating_unit, count = suffix_match
            if len(repeating_unit) * count > len(stripped_content) * 0.5:
                return prefix

    # Priority 2: Full-string character-level repetition (e.g., 'ababab')
    if "\n" not in stripped_content and len(stripped_content) > min_len:
        repeating_unit = find_shortest_repeating_substring(stripped_content)
        if repeating_unit:
            count = len(stripped_content) // len(repeating_unit)
            if count >= char_threshold:
                return repeating_unit

    # Priority 3: Line-level repetition (e.g., same line repeated many times)
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if not lines:
        return content
    total_lines = len(lines)
    if total_lines < line_threshold:
        return content
    line_counts = Counter(lines)
    most_common_line, count = line_counts.most_common(1)[0]
    if count >= line_threshold and (count / total_lines) >= 0.8:
        return most_common_line

    return content


ANNOT_TEXT_RE = re.compile(r"<\|TEXT_START\|>(.*?)<\|TEXT_END\|>", re.S)
LOC_BLOCK_RE = re.compile(r"<\|LOC_BEGIN\|>(.*?)<\|LOC_END\|>", re.S)
LOC_ITEM_RE = re.compile(r"<\|LOC_(\d+)\|>")
LOC_TOKEN_RE = re.compile(r"<\|LOC_(\d+)\|>")


def _to_pil_image(img):
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)


def _to_np_array(img):
    if isinstance(img, Image.Image):
        return np.array(img)
    return img


def pre_process_for_spotting(image: np.ndarray) -> np.ndarray:
    """Up-scale small images for the spotting task."""
    h, w = image.shape[:2]

    if w < 1500 and h < 1500:
        pil_img = _to_pil_image(image)
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        pil_img = pil_img.resize((w * 2, h * 2), resample_filter)
        return _to_np_array(pil_img)
    return image


def post_process_for_spotting(
    input_str: str, w: int, h: int
) -> Tuple[str, Dict[str, List]]:
    """Extract text and location blocks from a spotting result string."""
    if not isinstance(input_str, str):
        raise TypeError(f"input_str must be str, got {type(input_str).__name__}")

    # Extract text and location blocks
    texts = ANNOT_TEXT_RE.findall(input_str)
    loc_blocks = LOC_BLOCK_RE.findall(input_str)

    rec_polys = []
    rec_texts = []

    # Process the extracted text and location blocks
    n = min(len(texts), len(loc_blocks))
    for i in range(n):
        txt = texts[i].strip()
        loc_items = LOC_ITEM_RE.findall(loc_blocks[i])
        if len(loc_items) < 8:
            continue
        # Take the first 8 items (4 points)
        vals = list(map(int, loc_items[:8]))
        pts = [(vals[j], vals[j + 1]) for j in range(0, 8, 2)]
        pts = [(p[0] / 1000.0 * w, p[1] / 1000.0 * h) for p in pts]
        rec_polys.append(pts)
        rec_texts.append(txt)

    # If no polys or texts are extracted, try an alternative parsing method
    if not rec_polys or not rec_texts:
        matches = list(LOC_TOKEN_RE.finditer(input_str))
        last_end = 0
        i = 0
        while i + 7 < len(matches):
            group = matches[i : i + 8]
            vals = [int(m.group(1)) for m in group]
            pts = [(vals[j], vals[j + 1]) for j in range(0, 8, 2)]
            pts = [(p[0] / 1000.0 * w, p[1] / 1000.0 * h) for p in pts]
            text_span = input_str[last_end : group[0].start()]
            txt = text_span.strip()
            rec_texts.append(txt)
            rec_polys.append(pts)
            last_end = group[-1].end()
            i += 8

    # Join the extracted texts into a single string separated by newlines
    result_str = "\n\n".join(rec_texts)
    spotting_res = {"rec_polys": rec_polys, "rec_texts": rec_texts}

    return result_str, spotting_res
