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

"""Layout visualization drawn on the original document pages, saved as PDF.

Instead of saving standalone visualization images, the parsed layout blocks
(bbox / polygon from ``parsing_res_list``) are drawn as translucent filled
boxes directly on top of the original PDF pages (or the input image), and
all pages are written into a single PDF file.
"""

from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from venusocr.data.readers import get_pdf_render_scale
from venusocr.utils.logging import get_logger

logger = get_logger(__name__)

_FILL_ALPHA = 60
_IMAGE_RESOLUTION = 150.0


def _color_for_label(label: str) -> Tuple[int, int, int]:
    """Map a block label to a deterministic, vivid RGB color."""
    hue = (int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _unwrap(value):
    """Unwrap single-element list values (page_result width/height/page_index)."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a scalable default font, falling back to the bitmap default."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no ``size`` argument
        return ImageFont.load_default()


def _render_original_pages(input_path: str) -> Tuple[List[Image.Image], float]:
    """Render the original document pages as RGB images.

    PDFs are re-rendered with the same scale used by the parsing pipeline
    (``VENUSOCR_PDF_RENDER_SCALE``, default 2.0) so page coordinates line up
    with the pipeline results; other inputs are loaded as a single image.

    Returns:
        ``(pages, resolution)`` where ``resolution`` is the DPI recorded in
        the output PDF so page physical sizes match the original document.
    """
    if Path(input_path).suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        from venusocr.data.readers import pdfium_lock

        scale = get_pdf_render_scale()
        pages: List[Image.Image] = []
        with pdfium_lock:
            doc = pdfium.PdfDocument(str(input_path))
            try:
                for page in doc:
                    array = page.render(scale=scale).to_numpy()
                    page.close()
                    pages.append(Image.fromarray(array).convert("RGB"))
            finally:
                doc.close()
        return pages, 72.0 * scale
    return [Image.open(input_path).convert("RGB")], _IMAGE_RESOLUTION


def _polygon_points(block) -> Optional[List[Tuple[float, float]]]:
    """Return polygon points as ``[(x, y), ...]`` or ``None``."""
    points = block.polygon_points
    if points is None:
        return None
    try:
        import numpy as np

        if isinstance(points, np.ndarray):
            points = points.reshape(-1).tolist()
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    flat = []
    for item in points:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    if len(flat) < 6 or len(flat) % 2:
        return None
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]


def _draw_blocks(page_img: Image.Image, page_result) -> Image.Image:
    """Draw translucent layout boxes of one page result on a page image."""
    base = page_img.convert("RGBA")
    width, height = base.size
    src_w = _unwrap(page_result["width"]) or width
    src_h = _unwrap(page_result["height"]) or height
    scale_x, scale_y = width / src_w, height / src_h

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(width / 800))
    font = _load_font(max(12, round(width / 70)))

    for block in page_result["parsing_res_list"]:
        color = _color_for_label(block.label)
        fill = color + (_FILL_ALPHA,)
        outline = color + (255,)
        polygon = _polygon_points(block)
        if polygon is not None:
            scaled = [(x * scale_x, y * scale_y) for x, y in polygon]
            draw.polygon(scaled, fill=fill, outline=outline, width=line_width)
            anchor_x = min(p[0] for p in scaled)
            anchor_y = min(p[1] for p in scaled)
        else:
            x1, y1, x2, y2 = block.bbox
            anchor_x, anchor_y = x1 * scale_x, y1 * scale_y
            draw.rectangle(
                [anchor_x, anchor_y, x2 * scale_x, y2 * scale_y],
                fill=fill,
                outline=outline,
                width=line_width,
            )
        draw.text((anchor_x + line_width, anchor_y + line_width), block.label, font=font, fill=outline)

    return Image.alpha_composite(base, overlay).convert("RGB")


def _index_page_results(page_results) -> Dict[int, object]:
    """Map 0-based page indexes to page results (positional fallback)."""
    indexed = {}
    for position, res in enumerate(page_results):
        page_index = _unwrap(res.get("page_index"))
        indexed[page_index if isinstance(page_index, int) else position] = res
    return indexed


def render_layout_images(input_path: str, page_results) -> List[Image.Image]:
    """Render original pages with translucent layout boxes drawn on top.

    Args:
        input_path: Path of the parsed document (PDF or image).
        page_results: Page-level parsing results holding ``parsing_res_list``.

    Returns:
        One RGB image per document page.
    """
    pages, _ = _render_original_pages(input_path)
    indexed = _index_page_results(page_results)
    drawn = []
    for position, page_img in enumerate(pages):
        page_result = indexed.get(position)
        if page_result is None:
            drawn.append(page_img)
        else:
            drawn.append(_draw_blocks(page_img, page_result))
    return drawn


def save_layout_pdf(input_path: str, page_results, out_path) -> None:
    """Draw layout boxes on the original pages and save them as one PDF.

    Args:
        input_path: Path of the parsed document (PDF or image).
        page_results: Page-level parsing results holding ``parsing_res_list``.
        out_path: Target PDF path (parent directories are created).
    """
    pages, resolution = _render_original_pages(input_path)
    if not pages:
        raise ValueError(f"No pages rendered from: {input_path}")
    indexed = _index_page_results(page_results)
    drawn = [
        _draw_blocks(page_img, indexed[position]) if position in indexed else page_img
        for position, page_img in enumerate(pages)
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drawn[0].save(
        out_path,
        "PDF",
        save_all=True,
        append_images=drawn[1:],
        resolution=resolution,
    )
