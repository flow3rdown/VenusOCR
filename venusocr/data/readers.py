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

"""Input readers: image decoding (OpenCV) and PDF rendering (pypdfium2).

Vendored from ``paddlex.inference.common.reader.image_reader`` and
``paddlex.inference.utils.io.readers``. The PDF render scale defaults to
2.0 (same as upstream ``PADDLE_PDX_PDF_RENDER_SCALE``) and can be overridden
via the ``VENUSOCR_PDF_RENDER_SCALE`` environment variable.
"""

from __future__ import annotations

import os
import threading
from typing import Iterator

import numpy as np

_DEFAULT_PDF_RENDER_SCALE = 2.0

# pypdfium2 is not thread-safe; serialize all pdfium operations
# (same as upstream ``paddlex.inference.utils.pdfium_lock``).
pdfium_lock = threading.Lock()


def get_pdf_render_scale() -> float:
    """Return the PDF render scale (env ``VENUSOCR_PDF_RENDER_SCALE``, default 2.0)."""
    raw = os.environ.get("VENUSOCR_PDF_RENDER_SCALE", None)
    if raw is None:
        return _DEFAULT_PDF_RENDER_SCALE
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_PDF_RENDER_SCALE


class ReadImage:
    """Load images from file paths (or pass through numpy arrays)."""

    def __init__(self, format: str = "BGR"):
        """Initialize the reader.

        Args:
            format (str, optional): Target color format. Choices are 'BGR',
                'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__()
        self.format = format

    def __call__(self, imgs):
        """Read a list of images."""
        return [self.read(img) for img in imgs]

    def read(self, img):
        """Read one image (path string or numpy array)."""
        import cv2

        if isinstance(img, np.ndarray):
            if self.format == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif isinstance(img, str):
            flags = {
                "BGR": cv2.IMREAD_COLOR,
                "RGB": cv2.IMREAD_COLOR,
                "GRAY": cv2.IMREAD_GRAYSCALE,
            }[self.format]
            with open(img, "rb") as f:
                img_array = np.frombuffer(f.read(), np.uint8)
            blob = cv2.imdecode(img_array, flags=flags)
            if blob is None:
                raise Exception(f"Image read Error: {img}")

            if self.format == "RGB":
                if blob.ndim != 3:
                    raise RuntimeError("Array is not 3-dimensional.")
                # BGR to RGB
                blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            return blob
        else:
            raise TypeError(
                f"ReadImage only supports the following types:\n"
                f"1. str, indicating a image file path or a directory containing image files.\n"
                f"2. numpy.ndarray.\n"
                f"However, got type: {type(img).__name__}."
            )


class PDFReader:
    """Render PDF pages to RGB numpy arrays with pypdfium2."""

    def __init__(self, rotate: int = 0, zoom: float = _DEFAULT_PDF_RENDER_SCALE):
        self._rotation = rotate
        self._scale = zoom

    def load(self, in_path: str):
        """Load a PDF document (closed automatically by :meth:`read`)."""
        import pypdfium2 as pdfium

        with pdfium_lock:
            doc = pdfium.PdfDocument(str(in_path))
            doc.init_forms()
            return doc

    def read(self, doc) -> Iterator[np.ndarray]:
        """Yield rendered page images (numpy arrays) from a loaded document."""
        with pdfium_lock:
            try:
                for page in doc:
                    image = page.render(
                        scale=self._scale, rotation=self._rotation
                    ).to_numpy()
                    page.close()
                    yield image
            finally:
                doc.close()
