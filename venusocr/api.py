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

"""High-level Python API for VenusOCR."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional, Union

from venusocr.config import VenusOCRConfig, load_config
from venusocr.pipeline import DocumentParsingPipeline
from venusocr.results.document import DocumentResult
from venusocr.utils.fileio import collect_input_files, output_path_for
from venusocr.utils.logging import get_logger

logger = get_logger(__name__)


class VenusOCR:
    """Document parser powered by the VenusOCR model.

    Example:
        >>> from venusocr import VenusOCR
        >>> with VenusOCR(overrides={"engine.vl_rec_server_url": "http://127.0.0.1:8000/v1"}) as ocr:
        ...     result = ocr.parse("paper.pdf")
        ...     result.save("./output")
        ...     print(result.markdown)

    Args:
        config_path: Optional path to a YAML config file.
        config: Optional pre-built :class:`VenusOCRConfig` (takes precedence
            over ``config_path``).
        overrides: Optional dotted-path config overrides, e.g.
            ``{"engine.device": "cpu"}``.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[VenusOCRConfig] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = config.apply_overrides(overrides) if overrides else config
        else:
            self.config = load_config(config_path, overrides)
        self._pipeline: Optional[DocumentParsingPipeline] = None

    def __enter__(self) -> "VenusOCR":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def pipeline(self) -> DocumentParsingPipeline:
        """Lazily constructed parsing pipeline (loads local layout model)."""
        if self._pipeline is None:
            self._pipeline = DocumentParsingPipeline(self.config)
        return self._pipeline

    def parse(self, input_path: str) -> DocumentResult:
        """Parse a single image or PDF document.

        Args:
            input_path: Path to the document.

        Returns:
            A :class:`DocumentResult` holding per-page results.
        """
        logger.info("╭─ VenusOCR · 开始解析")
        logger.info("│ 输入       %s", input_path)
        pipeline_was_uninitialized = self._pipeline is None
        initialization_started = time.perf_counter()
        pipeline = self.pipeline
        initialization_seconds = time.perf_counter() - initialization_started
        page_results = pipeline.predict(input_path)
        metrics = pipeline.last_metrics
        if metrics is not None and pipeline_was_uninitialized:
            metrics.model_initialization_seconds = initialization_seconds
        result = DocumentResult(
            input_path=input_path,
            page_results=page_results,
            metrics=metrics,
        )
        logger.info("╰─ 解析阶段完成 · %d 页", len(page_results))
        return result

    def parse_files(
        self,
        paths: List[str],
        output_dir: Optional[str] = None,
        input_root: Optional[str] = None,
    ) -> Iterator[DocumentResult]:
        """Parse multiple documents sequentially, yielding results as they finish.

        Args:
            paths: Document paths to parse.
            output_dir: If given, each result is saved automatically
                (markdown / JSON / layout visualization per config).
            input_root: Batch input root used to mirror the directory
                structure under ``output_dir``.

        Yields:
            :class:`DocumentResult` for each successfully parsed document.
        """
        for path in paths:
            try:
                result = self.parse(path)
            except Exception:
                logger.exception("Failed to parse: %s", path)
                continue
            if output_dir is not None:
                result.save(
                    output_path_for(input_root, path, output_dir),
                    self.config.output,
                    log_summary=False,
                )
            yield result

    def parse_dir(
        self,
        input_dir: str,
        output_dir: str,
        num_workers: Optional[int] = None,
    ):
        """Parse all documents under a directory on a shared-model pipeline.

        Documents are flattened into pages and streamed through a single
        parsing pipeline (one layout model); pages are batched across
        document boundaries and regrouped per document for saving.

        Args:
            input_dir: Directory containing documents (searched recursively).
            output_dir: Base output directory.
            num_workers: Concurrent page-batch workers; defaults to
                ``config.batch.num_workers``. ``1`` means sequential
                per-document parsing.

        Returns:
            A :class:`~venusocr.batch.BatchSummary` with statistics.
        """
        from venusocr.batch import run_batch

        paths, input_root = collect_input_files(input_dir)
        workers = num_workers if num_workers is not None else self.config.batch.num_workers
        if workers <= 1:
            summary_ok, summary_failed = 0, 0
            total_pages = 0
            total_regions = 0
            total_requests = 0
            total_blocks = 0
            stage_seconds: Dict[str, float] = {}
            started = time.perf_counter()
            for result in self.parse_files(paths, output_dir, input_root):
                summary_ok += 1
                total_pages += result.metrics.page_count
                total_regions += result.metrics.layout_regions
                total_requests += result.metrics.recognition_requests
                total_blocks += result.metrics.output_blocks
                for stage, seconds in result.metrics.stage_seconds.items():
                    stage_seconds[stage] = stage_seconds.get(stage, 0.0) + seconds
                if result.metrics.model_initialization_seconds:
                    stage_seconds["model_initialization"] = (
                        stage_seconds.get("model_initialization", 0.0)
                        + result.metrics.model_initialization_seconds
                    )
            summary_failed = len(paths) - summary_ok
            from venusocr.batch import BatchSummary

            summary = BatchSummary(
                total=len(paths),
                succeeded=summary_ok,
                failed=summary_failed,
                elapsed_seconds=time.perf_counter() - started,
                page_count=total_pages,
                layout_regions=total_regions,
                recognition_requests=total_requests,
                output_blocks=total_blocks,
                stage_seconds=stage_seconds,
            )
            logger.info("%s", summary.format_summary())
            return summary
        return run_batch(
            paths=paths,
            config=self.config,
            output_dir=output_dir,
            num_workers=workers,
            input_root=input_root,
        )

    def close(self) -> None:
        """Release pipeline resources."""
        if self._pipeline is not None:
            self._pipeline.close()
        self._pipeline = None


def parse(
    input_path: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Union[DocumentResult, List[DocumentResult]]:
    """One-shot convenience function to parse a file or directory.

    Args:
        input_path: A document file or a directory of documents.
        output_dir: If given, results are saved to this directory.
        config_path: Optional YAML config path.
        overrides: Optional dotted-path config overrides.

    Returns:
        A single :class:`DocumentResult` for file input, or a list of
        results for directory input.
    """
    paths, input_root = collect_input_files(input_path)
    with VenusOCR(config_path=config_path, overrides=overrides) as ocr:
        if input_root is None:
            result = ocr.parse(paths[0])
            if output_dir is not None:
                result.save(output_dir, ocr.config.output)
            return result
        results = list(ocr.parse_files(paths, output_dir, input_root))
    return results
