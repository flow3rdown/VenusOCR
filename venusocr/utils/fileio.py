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

"""Input file discovery for VenusOCR."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".pdf"}


def collect_input_files(input_path: str) -> Tuple[List[str], Optional[str]]:
    """Collect supported document files from a file or directory.

    When ``input_path`` is a directory the search is recursive and the
    returned root can be used to mirror the directory structure in the
    output directory.

    Args:
        input_path: A single document file or a directory.

    Returns:
        A tuple ``(file_paths, input_root)``. ``input_root`` is the absolute
        directory path when the input is a directory, otherwise ``None``.

    Raises:
        ValueError: If the path does not exist, has an unsupported suffix,
            or the directory contains no supported files.
    """
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type: {path.suffix!r}. "
                f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
            )
        return [str(path.absolute())], None

    if path.is_dir():
        seen = set()
        file_paths: List[str] = []
        for suffix in sorted(SUPPORTED_SUFFIXES):
            for pattern in (f"*{suffix}", f"*{suffix.upper()}"):
                for p in path.rglob(pattern):
                    abs_p = str(p.absolute())
                    if abs_p not in seen:
                        seen.add(abs_p)
                        file_paths.append(abs_p)
        file_paths.sort()
        if not file_paths:
            raise ValueError(f"No supported document files found in: {input_path}")
        return file_paths, str(path.absolute())

    raise ValueError(f"Path does not exist: {input_path}")


def output_name_for(file_path: str) -> str:
    """Retain the suffix when sibling documents share an output stem."""
    path = Path(file_path)
    if path.parent.is_dir():
        for sibling in path.parent.iterdir():
            if (
                sibling.name != path.name
                and (sibling.stem == path.stem or sibling.name == path.stem)
                and sibling.suffix.lower() in SUPPORTED_SUFFIXES
                and sibling.is_file()
            ):
                return path.name
    return path.stem


def output_path_for(input_root: Optional[str], file_path: str, output_dir: str) -> str:
    """Compute the output directory for one file, mirroring the input tree.

    Args:
        input_root: Root directory of a batch input (``None`` for single files).
        file_path: Absolute path of the input file.
        output_dir: Base output directory.

    Returns:
        The directory into which this file's results should be written.
    """
    if input_root is None:
        return output_dir
    rel = Path(file_path).parent.relative_to(input_root)
    if str(rel) == ".":
        return output_dir
    return str(Path(output_dir) / rel)
