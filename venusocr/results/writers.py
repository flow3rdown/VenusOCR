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

"""Low-level artifact writers for VenusOCR results.

These functions are vendored from ``paddlex.inference.utils.io.writers``
(JSON / image / markdown backends) and stripped down to exactly what
VenusOCR persists. Parent directories are created automatically, matching
upstream behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image


def write_json(
    out_path: Union[str, Path],
    obj: Any,
    *,
    indent: int = 4,
    ensure_ascii: bool = False,
) -> None:
    """Write a JSON-serializable object to ``out_path``.

    Args:
        out_path: Target file path (parent directories are created).
        obj: JSON-serializable object.
        indent: Indentation for pretty printing.
        ensure_ascii: Whether to escape non-ASCII characters.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def write_image(out_path: Union[str, Path], obj: Union[Image.Image, np.ndarray]) -> None:
    """Write an image (PIL image or numpy array) to ``out_path``.

    Args:
        out_path: Target file path (parent directories are created).
        obj: Image data. 4-channel images are always saved as PNG,
            mirroring the upstream PIL writer backend.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, Image.Image):
        img = obj
    elif isinstance(obj, np.ndarray):
        img = Image.fromarray(obj)
    else:
        raise TypeError(f"Unsupported image object type: {type(obj)}")
    format_ = "PNG" if len(img.getbands()) == 4 else None
    img.save(out_path.as_posix(), format=format_)


def write_markdown(out_path: Union[str, Path], text: str) -> None:
    """Write markdown text to ``out_path``.

    Args:
        out_path: Target file path (parent directories are created).
        text: Markdown content.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode="w", encoding="utf-8", errors="replace") as f:
        f.write(text)
