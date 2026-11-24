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

"""Document-level parsing result for VenusOCR."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from venusocr.config import OutputConfig
from venusocr.results.layout_pdf import save_layout_pdf
from venusocr.results.page import PageResult
from venusocr.results.writers import write_image, write_markdown
from venusocr.utils.fileio import output_name_for
from venusocr.utils.logging import get_logger
from venusocr.utils.profiling import ParseMetrics

logger = get_logger(__name__)


class DocumentResult:
    """Parsing result of a single input document (one or more pages).

    Attributes:
        input_path: Absolute path of the parsed document.
        page_results: Page-level results returned by the pipeline.
    """

    def __init__(
        self,
        input_path: str,
        page_results: List[PageResult],
        metrics: Optional[ParseMetrics] = None,
    ) -> None:
        self.input_path = input_path
        self.page_results = page_results
        self.metrics = metrics or ParseMetrics(
            input_path=input_path,
            page_count=len(page_results),
            layout_regions=sum(
                len(result.get("layout_det_res", {}).get("boxes", []))
                for result in page_results
            ),
            output_blocks=sum(
                len(result.get("parsing_res_list", [])) for result in page_results
            ),
        )

    @property
    def stem(self) -> str:
        """File stem of the input document, used for naming artifacts."""
        return Path(self.input_path).stem

    @property
    def markdown(self) -> str:
        """Concatenated pretty Markdown of all pages.

        This preserves the page-result property's default behavior. Use
        :meth:`to_markdown` when the rendering style must be explicit.
        """
        return self.to_markdown(pretty=True)

    def to_markdown(self, *, pretty: bool = True, show_formula_number: bool = False) -> str:
        """Render all pages with one consistent Markdown policy."""
        parts = []
        for res in self.page_results:
            md = res._to_markdown(
                pretty=pretty, show_formula_number=show_formula_number
            ).get("markdown_texts", "")
            parts.append(md)
        return "\n\n".join(p for p in parts if p)

    def _prepare_document_image_paths(self) -> None:
        """Prefix embedded-image paths with the page index (idempotent).

        Cropped-region image paths are derived from labels and box
        coordinates, which may collide across pages. Rewriting them once per
        document keeps every image reference in the combined markdown unique.
        """
        if getattr(self, "_image_paths_prefixed", False):
            return
        for position, res in enumerate(self.page_results):
            page_index = res.get("page_index")
            if isinstance(page_index, list):
                page_index = page_index[0] if page_index else position
            if page_index is None:
                page_index = position
            for block in res["parsing_res_list"]:
                if block.image is not None:
                    block.image["path"] = (
                        f"imgs/page{page_index}_{Path(block.image['path']).name}"
                    )
            for img in res.get("imgs_in_doc") or []:
                img["path"] = f"imgs/page{page_index}_{Path(img['path']).name}"
        self._image_paths_prefixed = True

    def save(
        self,
        output_dir: str,
        output_config: Optional[OutputConfig] = None,
        *,
        log_summary: bool = True,
    ) -> None:
        """Persist the combined markdown and layout-visualization PDF.

        All artifacts of this document are written into a single
        sub-directory named after the input file::

            <output_dir>/<stem>/<stem>.md           # whole-document markdown
            <output_dir>/<stem>/imgs/...            # cropped region images
            <output_dir>/<stem>/<stem>_layout.pdf   # layout boxes on original pages

        Args:
            output_dir: Root output directory; a ``<stem>`` sub-directory is
                created inside it for this document. For colliding sibling
                names, the input suffix is retained in the directory and artifacts.
            output_config: Controls which artifacts are written. When
                ``None`` everything (markdown + layout PDF) is saved.
            log_summary: Whether to log the document timing summary.
        """
        cfg = output_config or OutputConfig()
        save_started = time.perf_counter()
        document_name = output_name_for(self.input_path)
        doc_dir = Path(output_dir) / document_name
        doc_dir.mkdir(parents=True, exist_ok=True)

        if cfg.save_markdown:
            self._prepare_document_image_paths()
            texts = []
            for res in self.page_results:
                data = res._to_markdown(pretty=cfg.pretty_markdown)
                md = data.get("markdown_texts", "")
                if md:
                    texts.append(md)
                for img_path, img_data in data.get("markdown_images", {}).items():
                    if img_data:
                        write_image(doc_dir / img_path, img_data)
            write_markdown(doc_dir / f"{document_name}.md", "\n\n".join(texts))

        if cfg.save_layout_vis:
            try:
                save_layout_pdf(
                    self.input_path,
                    self.page_results,
                    doc_dir / f"{document_name}_layout.pdf",
                )
            except Exception:  # noqa: BLE001 - visualization must not fail the run
                logger.exception("Failed to save layout PDF for: %s", self.input_path)
        self.metrics.add_stage("result_saving", time.perf_counter() - save_started)
        if log_summary:
            logger.info("%s", self.metrics.format_summary(include_save=True))
