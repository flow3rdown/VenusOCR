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

"""Batch sampling over image / PDF inputs.

Vendored from ``paddlex.inference.common.batch_sampler`` (``BaseBatchSampler``
and ``ImageBatchSampler``). A PDF is expanded into one instance per page
(rendered via :class:`~venusocr.data.readers.PDFReader`); images and numpy
arrays are passed through as single instances.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import numpy as np

from venusocr.data.readers import PDFReader, get_pdf_render_scale
_CACHE_DIR = Path(os.environ.get("VENUSOCR_CACHE_DIR", Path.home() / ".cache" / "venusocr"))


class ImgBatch:
    """A batch of page/image instances with their provenance metadata."""

    def __init__(self):
        self.instances = []
        self.input_paths = []
        self.page_indexes = []
        self.page_counts = []

    def append(self, instance, input_path, page_index, page_count):
        """Append one instance to the batch."""
        self.instances.append(instance)
        self.input_paths.append(input_path)
        self.page_indexes.append(page_index)
        self.page_counts.append(page_count)

    def __len__(self):
        return len(self.instances)


class ImageBatchSampler:
    """Yield :class:`ImgBatch` objects from file paths / numpy arrays.

    Args:
        batch_size (int): Number of page/image instances per batch.
    """

    IMG_SUFFIX = [
        "bmp", "dib",
        "jpeg", "jpg",
        "png",
        "webp",
        "pbm", "pgm", "ppm", "pnm",
        "sr", "ras",
        "tiff", "tif",
    ]
    PDF_SUFFIX = ["pdf"]

    def __init__(self, batch_size: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.batch_size = batch_size
        self.pdf_reader = PDFReader(zoom=get_pdf_render_scale())

    def __call__(self, input: Any) -> Iterator[ImgBatch]:
        yield from self.sample(input)

    def _download_from_url(self, in_path: str) -> str:
        file_name = Path(urlparse(in_path).path).name
        if not file_name:
            raise ValueError(f"Cannot determine a file name from URL: {in_path}")
        save_path = _CACHE_DIR / "predict_input" / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(in_path, save_path.as_posix())
        return save_path.as_posix()

    def _get_files_list(self, fp):
        if fp is None or not os.path.exists(fp):
            raise FileNotFoundError(f"No files found at path: {fp}")
        if os.path.isfile(fp):
            return [fp]

        file_list = []
        if os.path.isdir(fp):
            for root, _, files in os.walk(fp):
                for single_file in files:
                    if (
                        single_file.split(".")[-1].lower()
                        in self.IMG_SUFFIX + self.PDF_SUFFIX
                    ):
                        file_list.append(os.path.join(root, single_file))
        if len(file_list) == 0:
            raise ValueError(f"No supported PDF or image files found in: {fp}")
        file_list = sorted(file_list)
        return file_list

    def sample(self, inputs) -> Iterator[ImgBatch]:
        """Sample inputs into batches of page/image instances."""
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = ImgBatch()
        for item in inputs:
            if isinstance(item, np.ndarray):
                batch.append(item, None, None, None)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = ImgBatch()
            elif isinstance(item, str):
                is_url = item.startswith(("http://", "https://"))
                suffix_source = urlparse(item).path if is_url else item
                suffix = Path(suffix_source).suffix.lower().lstrip(".")
                if suffix in self.PDF_SUFFIX:
                    file_path = (
                        self._download_from_url(item) if is_url else item
                    )
                    doc = self.pdf_reader.load(file_path)
                    page_count = len(doc)
                    for page_idx, page_img in enumerate(self.pdf_reader.read(doc)):
                        batch.append(page_img, file_path, page_idx, page_count)
                        if len(batch) == self.batch_size:
                            yield batch
                            batch = ImgBatch()
                elif suffix in self.IMG_SUFFIX:
                    file_path = (
                        self._download_from_url(item) if is_url else item
                    )
                    batch.append(file_path, file_path, None, None)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = ImgBatch()
                elif not is_url and Path(item).is_dir():
                    file_list = self._get_files_list(item)
                    yield from self.sample(file_list)
                else:
                    raise ValueError(
                        "Unsupported input file type. Expected PDF or an image "
                        f"ending in {self.IMG_SUFFIX + self.PDF_SUFFIX}, got: {item}"
                    )
            else:
                raise TypeError(
                    "ImageBatchSampler only supports str and numpy.ndarray, "
                    f"got {type(item).__name__}"
                )
        if len(batch) > 0:
            yield batch
