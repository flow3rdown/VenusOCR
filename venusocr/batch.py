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

"""Shared-model, page-flattened concurrent batch parsing for VenusOCR.

Instead of spawning one process per worker (each holding its own layout
model), the batch runner builds a single :class:`DocumentParsingPipeline`
and drives it with a four-stage thread pipeline:

1. **Loader threads** flatten every input document into pages (PDFs are
   rendered page by page) and push them onto a bounded page queue.
2. **The layout thread** packs pages — across document boundaries — into
   batches of ``batch.page_batch_size`` and runs layout detection on the
   single shared model.
3. **Batch workers** (``num_workers`` threads) crop regions, issue VLM
   recognition requests (bounded by the client's concurrency semaphore)
   and assemble per-page results.
4. **Saver threads** group finished pages back into documents and persist
   markdown / layout visualizations as soon as a document completes.

Back-pressure flows backwards through the bounded page queue and a
semaphore that limits in-flight page batches, so memory stays bounded
regardless of how many documents are queued.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from tqdm import tqdm

from venusocr.config import VenusOCRConfig
from venusocr.data.readers import ReadImage
from venusocr.data.sampler import ImageBatchSampler, ImgBatch
from venusocr.utils.fileio import output_path_for
from venusocr.utils.logging import get_logger
from venusocr.utils.profiling import STAGE_LABELS, ParseMetrics, format_duration

logger = get_logger(__name__)


def _normalize_stage_seconds(
    stage_seconds: Dict[str, float], elapsed_seconds: float
) -> Dict[str, float]:
    """Allocate batch wall time across stages using accumulated worker ratios.

    Worker stage timers are useful for estimating relative cost, but summing
    them directly over parallel workers can greatly exceed wall time. This
    conversion preserves the measured stage proportions while making the
    displayed stages add up to the actual end-to-end batch duration.
    """
    positive = {
        stage: max(0.0, float(seconds))
        for stage, seconds in stage_seconds.items()
        if seconds > 0
    }
    accumulated_total = sum(positive.values())
    if accumulated_total <= 0 or elapsed_seconds <= 0:
        return {}
    scale = elapsed_seconds / accumulated_total
    normalized = {stage: seconds * scale for stage, seconds in positive.items()}
    # Avoid a visible rounding/drift mismatch when consumers sum the values.
    last_stage = next(reversed(normalized))
    normalized[last_stage] += elapsed_seconds - sum(normalized.values())
    return normalized


@dataclass
class BatchSummary:
    """Statistics of a finished (or interrupted) batch run."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    crashed: bool = False
    elapsed_seconds: float = 0.0
    page_count: int = 0
    layout_regions: int = 0
    recognition_requests: int = 0
    output_blocks: int = 0
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    failed_files: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage_seconds = _normalize_stage_seconds(
            self.stage_seconds, self.elapsed_seconds
        )

    @property
    def processed(self) -> int:
        """Number of files attempted (succeeded + failed)."""
        return self.succeeded + self.failed

    def format_summary(self) -> str:
        """Build a compact summary for a completed directory job."""
        average_page_seconds = (
            self.elapsed_seconds / self.page_count if self.page_count else 0.0
        )
        throughput = self.page_count / self.elapsed_seconds if self.elapsed_seconds else 0.0
        lines = [
            "╭─ VenusOCR · 批量任务完成",
            f"│ 文档       {self.succeeded} 成功 · {self.failed} 失败 · 共 {self.total} 个",
            (
                f"│ 页面       {self.page_count} 页 · {self.layout_regions} 个版面区域 · "
                f"{self.recognition_requests} 次识别请求 · {self.output_blocks} 个输出块"
            ),
        ]
        if self.stage_seconds:
            lines.append("├─ 阶段耗时（墙钟分摊）")
            for stage in (
                "model_initialization",
                "input_loading",
                "layout_detection",
                "region_preparation",
                "vlm_recognition",
                "result_assembly",
                "result_saving",
            ):
                if stage in self.stage_seconds:
                    lines.append(
                        f"│  {STAGE_LABELS[stage]:<20} "
                        f"{format_duration(self.stage_seconds[stage])}"
                    )
        lines.extend(
            [
                "├─ 墙钟统计",
                f"│  总耗时                 {format_duration(self.elapsed_seconds)}",
                f"│  平均每页               {format_duration(average_page_seconds)}",
                f"│  处理吞吐               {throughput:.2f} 页/s",
                "╰────────────────────────────────────────",
            ]
        )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Shared run state
# ----------------------------------------------------------------------


@dataclass
class _PageTask:
    """One flattened page flowing through the pipeline."""

    doc_idx: int
    seq: int  # per-document page sequence (ordering key)
    input_path: str
    page_index: Optional[int]
    page_count: Optional[int]
    image: Any  # decoded page image (numpy array)
    load_seconds: float


@dataclass
class _DocState:
    """Book-keeping for one input document."""

    path: str
    registered: int = 0
    loaded: bool = False
    status: str = "running"  # running / emitted / failed
    results: Dict[int, Any] = field(default_factory=dict)
    metrics: Optional[ParseMetrics] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.metrics = ParseMetrics(input_path=self.path)


class _BatchMetricsShim:
    """Per-page-batch metrics sink recorded by one batch worker thread.

    Mirrors the small subset of the :class:`ParseMetrics` interface that
    ``get_layout_parsing_results`` touches. Each worker owns one shim per
    page batch, so concurrent batches never mix timings.
    """

    def __init__(self) -> None:
        self.stage_seconds: Dict[str, float] = {}
        self.recognition_requests = 0

    def add_stage(self, stage: str, seconds: float) -> None:
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + max(
            0.0, float(seconds)
        )


class _DocumentTracker:
    """Thread-safe per-document bookkeeping for the flattened page stream."""

    def __init__(self, paths: List[str]) -> None:
        self._lock = threading.Lock()
        self._docs = [_DocState(path=path) for path in paths]
        self.succeeded = 0
        self.failed = 0
        self.save_queue: "queue.Queue" = queue.Queue()
        self.errors: List[str] = []

    @property
    def total(self) -> int:
        return len(self._docs)

    @property
    def processed(self) -> int:
        return self.succeeded + self.failed

    def register_page(self, doc_idx: int) -> int:
        """Register one loaded page; returns its per-document sequence."""
        with self._lock:
            doc = self._docs[doc_idx]
            seq = doc.registered
            doc.registered += 1
            return seq

    def finish_loading(self, doc_idx: int) -> None:
        """Mark a document fully loaded; emits zero-page documents at once."""
        with self._lock:
            doc = self._docs[doc_idx]
            doc.loaded = True
            self._maybe_emit_locked(doc_idx, doc)

    def fail_doc(self, doc_idx: int, error: str) -> None:
        """Mark a document failed (first failure wins, idempotent)."""
        with self._lock:
            doc = self._docs[doc_idx]
            if doc.status != "running":
                return
            doc.status = "failed"
            doc.error = error
            doc.results.clear()
            self.failed += 1
            self.errors.append(f"{doc.path}: {error}")

    def is_failed(self, doc_idx: int) -> bool:
        with self._lock:
            return self._docs[doc_idx].status == "failed"

    def add_result(
        self,
        doc_idx: int,
        seq: int,
        page_result: Any,
        *,
        layout_regions: int,
        recognition_requests: int,
        output_blocks: int,
        stage_seconds: Dict[str, float],
    ) -> None:
        """Store one finished page and emit the document when complete."""
        with self._lock:
            doc = self._docs[doc_idx]
            if doc.status != "running":
                return
            doc.results[seq] = page_result
            metrics = doc.metrics
            metrics.page_count += 1
            metrics.layout_regions += layout_regions
            metrics.recognition_requests += recognition_requests
            metrics.output_blocks += output_blocks
            for stage, seconds in stage_seconds.items():
                metrics.add_stage(stage, seconds)
            self._maybe_emit_locked(doc_idx, doc)

    def mark_saved(self, doc_idx: int, ok: bool, error: Optional[str] = None) -> None:
        """Record the outcome of persisting an emitted document."""
        with self._lock:
            doc = self._docs[doc_idx]
            if ok:
                self.succeeded += 1
            elif doc.status == "emitted":
                doc.status = "failed"
                doc.error = error
                self.failed += 1
                self.errors.append(f"{doc.path}: {error}")

    def _maybe_emit_locked(self, doc_idx: int, doc: _DocState) -> None:
        if (
            doc.status == "running"
            and doc.loaded
            and len(doc.results) == doc.registered
        ):
            doc.status = "emitted"
            ordered = [doc.results[seq] for seq in sorted(doc.results)]
            doc.results = {}
            self.save_queue.put((doc_idx, doc.path, ordered, doc.metrics))


class _LoaderPool:
    """Hands out document indices to loader threads and tracks liveness."""

    def __init__(self, total: int) -> None:
        self._lock = threading.Lock()
        self._next = 0
        self._total = total
        self.active = 0

    def start_worker(self) -> None:
        with self._lock:
            self.active += 1

    def next_doc(self) -> Optional[int]:
        with self._lock:
            if self._next >= self._total:
                return None
            doc_idx = self._next
            self._next += 1
            return doc_idx

    def finish_worker(self) -> None:
        with self._lock:
            self.active -= 1

    @property
    def all_done(self) -> bool:
        with self._lock:
            return self.active == 0 and self._next >= self._total


# ----------------------------------------------------------------------
# Stage 1: document loading (flatten documents into pages)
# ----------------------------------------------------------------------


def _iter_document_pages(path: str, sampler: ImageBatchSampler, img_reader: ReadImage):
    """Yield ``(image, page_index, page_count, load_seconds)`` for one document.

    PDFs are rendered page by page; image files yield a single decoded page.
    """
    is_url = path.startswith(("http://", "https://"))
    suffix_source = urlparse(path).path if is_url else path
    suffix = Path(suffix_source).suffix.lower().lstrip(".")

    if suffix in ImageBatchSampler.PDF_SUFFIX:
        file_path = sampler._download_from_url(path) if is_url else path
        pdf_reader = sampler.pdf_reader
        doc = pdf_reader.load(file_path)
        page_count = len(doc)
        page_iter = pdf_reader.read(doc)
        page_index = 0
        while True:
            started = time.perf_counter()
            try:
                image = next(page_iter)
            except StopIteration:
                break
            load_seconds = time.perf_counter() - started
            yield image, page_index, page_count, load_seconds
            page_index += 1
    elif suffix in ImageBatchSampler.IMG_SUFFIX:
        file_path = sampler._download_from_url(path) if is_url else path
        started = time.perf_counter()
        image = img_reader.read(file_path)
        yield image, None, None, time.perf_counter() - started
    else:
        raise ValueError(f"Unsupported input file type: {path}")


def _loader_worker(
    pool: _LoaderPool,
    paths: List[str],
    page_queue: "queue.Queue",
    tracker: _DocumentTracker,
    shutdown: threading.Event,
) -> None:
    """Flatten documents into pages and push them onto the page queue."""
    pool.start_worker()
    try:
        sampler = ImageBatchSampler(batch_size=1)
        img_reader = ReadImage(format="BGR")
        while not shutdown.is_set():
            doc_idx = pool.next_doc()
            if doc_idx is None:
                return
            path = paths[doc_idx]
            try:
                for image, page_index, page_count, load_seconds in _iter_document_pages(
                    path, sampler, img_reader
                ):
                    if shutdown.is_set():
                        return
                    if tracker.is_failed(doc_idx):
                        # Stop rendering the rest of a document that has
                        # already failed downstream (e.g. in a shared batch).
                        break
                    seq = tracker.register_page(doc_idx)
                    task = _PageTask(
                        doc_idx=doc_idx,
                        seq=seq,
                        input_path=path,
                        page_index=page_index,
                        page_count=page_count,
                        image=image,
                        load_seconds=load_seconds,
                    )
                    while not shutdown.is_set():
                        try:
                            page_queue.put(task, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                    else:
                        return
                tracker.finish_loading(doc_idx)
            except Exception as e:  # noqa: BLE001 - isolate per-document failures
                logger.warning("Failed to load %s: %r", path, e)
                tracker.fail_doc(doc_idx, repr(e))
    finally:
        pool.finish_worker()


# ----------------------------------------------------------------------
# Stage 2 + 3: layout detection and recognition workers
# ----------------------------------------------------------------------


def _build_pseudo_layout_results(pages: List[_PageTask], prompt_label: str):
    """Single full-page box per page when layout detection is disabled."""
    results = []
    for page in pages:
        image = page.image
        results.append(
            {
                "input_path": None,
                "page_index": None,
                "boxes": [
                    {
                        "cls_id": 0,
                        "label": prompt_label.lower(),
                        "score": 1,
                        "coordinate": [0, 0, image.shape[1], image.shape[0]],
                    }
                ],
            }
        )
    return results


def _process_page_batch(
    pipeline,
    pages: List[_PageTask],
    layout_det_results,
    layout_seconds: float,
    model_settings,
    vlm_kwargs,
    cfg,
    tracker: _DocumentTracker,
) -> None:
    """Crop regions, recognize, and assemble one flattened page batch."""
    from venusocr.pipeline.crop import gather_imgs

    use_layout = model_settings["use_layout_detection"]
    image_arrays = [page.image for page in pages]

    prep_started = time.perf_counter()
    if use_layout:
        imgs_in_doc = [
            gather_imgs(image, layout_det_res["boxes"])
            for image, layout_det_res in zip(image_arrays, layout_det_results)
        ]
    else:
        imgs_in_doc = [[] for _ in layout_det_results]
    gather_seconds = time.perf_counter() - prep_started

    batch_data = ImgBatch()
    for page in pages:
        batch_data.append(
            page.input_path, page.input_path, page.page_index, page.page_count
        )

    shim = _BatchMetricsShim()
    pipeline._active_metrics = shim
    try:
        page_results, request_counts = pipeline._process_vlm(
            batch_data,
            image_arrays,
            layout_det_results,
            imgs_in_doc,
            model_settings,
            vlm_kwargs,
            cfg,
        )
    finally:
        pipeline._active_metrics = None

    batch_stage_seconds = dict(shim.stage_seconds)
    batch_stage_seconds["region_preparation"] = (
        batch_stage_seconds.get("region_preparation", 0.0) + gather_seconds
    )
    if use_layout:
        batch_stage_seconds["layout_detection"] = layout_seconds

    num_pages = len(pages)
    per_page_stage_seconds = {
        stage: seconds / num_pages for stage, seconds in batch_stage_seconds.items()
    }
    for page, layout_det_res, page_result, request_count in zip(
        pages, layout_det_results, page_results, request_counts
    ):
        stage_seconds = dict(per_page_stage_seconds)
        if page.load_seconds:
            stage_seconds["input_loading"] = page.load_seconds
        tracker.add_result(
            page.doc_idx,
            page.seq,
            page_result,
            layout_regions=len(layout_det_res["boxes"]),
            recognition_requests=request_count,
            output_blocks=len(page_result.get("parsing_res_list", [])),
            stage_seconds=stage_seconds,
        )


def _run_page_batch(
    in_flight: threading.Semaphore,
    *args,
) -> None:
    """Executor wrapper: guarantee the in-flight slot is always released."""
    tracker = args[-1]
    pages = args[1]
    try:
        _process_page_batch(*args)
    except Exception as e:  # noqa: BLE001 - fail affected docs, keep going
        logger.exception("Recognition failed for a page batch")
        for doc_idx in {page.doc_idx for page in pages}:
            tracker.fail_doc(doc_idx, f"recognition failed: {e!r}")
    finally:
        in_flight.release()


def _layout_worker(
    pipeline,
    loader_pool: _LoaderPool,
    page_queue: "queue.Queue",
    page_batch_size: int,
    model_settings,
    vlm_kwargs,
    cfg,
    prompt_label: str,
    tracker: _DocumentTracker,
    executor: ThreadPoolExecutor,
    in_flight: threading.Semaphore,
    shutdown: threading.Event,
) -> None:
    """Pack flattened pages into batches and run shared layout detection.

    The thread only returns after every submitted page batch has finished,
    so "all stage threads dead" in the main loop is a reliable completion /
    crash signal.
    """
    use_layout = model_settings["use_layout_detection"]
    pages: List[_PageTask] = []
    futures: List[Any] = []

    def flush() -> None:
        if not pages:
            return
        batch = pages[:]
        pages.clear()
        try:
            if use_layout:
                started = time.perf_counter()
                layout_det_results = pipeline.layout_detector.predict(
                    [page.image for page in batch],
                    layout_shape_mode=cfg.layout_shape_mode,
                )
                layout_seconds = time.perf_counter() - started
            else:
                layout_det_results = _build_pseudo_layout_results(batch, prompt_label)
                layout_seconds = 0.0
        except Exception as e:  # noqa: BLE001 - fail affected docs, keep going
            logger.exception("Layout detection failed for a page batch")
            for doc_idx in {page.doc_idx for page in batch}:
                tracker.fail_doc(doc_idx, f"layout detection failed: {e!r}")
            return
        while not shutdown.is_set():
            if in_flight.acquire(timeout=0.5):
                break
        else:
            for page in batch:
                tracker.fail_doc(page.doc_idx, "pipeline shut down before recognition")
            return
        futures.append(
            executor.submit(
                _run_page_batch,
                in_flight,
                pipeline,
                batch,
                layout_det_results,
                layout_seconds,
                model_settings,
                vlm_kwargs,
                cfg,
                tracker,
            )
        )

    try:
        while not shutdown.is_set():
            try:
                item = page_queue.get(timeout=0.1)
            except queue.Empty:
                if loader_pool.all_done:
                    flush()
                    return
                continue
            if tracker.is_failed(item.doc_idx):
                continue
            pages.append(item)
            if len(pages) >= page_batch_size:
                flush()
    finally:
        # Pages never submitted cannot complete; fail their documents.
        for page in pages:
            tracker.fail_doc(page.doc_idx, "pipeline shut down before layout detection")
        pages.clear()
        # Wait for all in-flight batches so that every document is accounted
        # for (failed or emitted) by the time this thread exits.
        for future in futures:
            try:
                future.result()
            except Exception:  # noqa: BLE001 - _run_page_batch never raises
                pass


# ----------------------------------------------------------------------
# Stage 4: result saving
# ----------------------------------------------------------------------


def _saver_worker(
    tracker: _DocumentTracker,
    output_dir: str,
    input_root: Optional[str],
    output_config,
) -> None:
    """Persist emitted documents as they complete."""
    from venusocr.results.document import DocumentResult

    while True:
        item = tracker.save_queue.get()
        if item is None:
            return
        doc_idx, path, page_results, metrics = item
        try:
            result = DocumentResult(
                input_path=path,
                page_results=page_results,
                metrics=metrics,
            )
            result.save(
                output_path_for(input_root, path, output_dir),
                output_config,
                log_summary=False,
            )
        except Exception as e:  # noqa: BLE001 - record and continue
            logger.exception("Failed to save results for %s", path)
            tracker.mark_saved(doc_idx, False, f"save failed: {e!r}")
        else:
            tracker.mark_saved(doc_idx, True)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def run_batch(
    paths: List[str],
    config: VenusOCRConfig,
    output_dir: str,
    num_workers: int = 8,
    input_root: Optional[str] = None,
    show_progress: bool = True,
) -> BatchSummary:
    """Parse many documents concurrently on a single shared pipeline.

    All documents are flattened into pages and streamed through one
    :class:`DocumentParsingPipeline` (one layout model instance). Pages are
    packed into batches of ``config.batch.page_batch_size`` — across document
    boundaries — and ``num_workers`` batch-worker threads run region
    preparation, VLM recognition, and assembly concurrently. Finished pages
    are regrouped by document and saved as soon as a document completes.

    A ``failed_files.txt`` and a ``_DONE`` marker file are written to
    ``output_dir`` at the end for auditing and resumability checks.

    Args:
        paths: Absolute document paths to parse.
        config: Shared configuration.
        output_dir: Base output directory.
        num_workers: Number of concurrent page-batch worker threads.
        input_root: Optional batch input root to mirror the directory tree.
        show_progress: Whether to render a tqdm progress bar.

    Returns:
        A :class:`BatchSummary` with run statistics.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be greater than zero")
    os.makedirs(output_dir, exist_ok=True)
    batch_cfg = config.batch
    page_batch_size = max(1, batch_cfg.page_batch_size)
    num_loaders = max(1, batch_cfg.loader_workers)
    num_savers = max(1, batch_cfg.saver_workers)
    logger.info(
        "╭─ VenusOCR · 批量解析\n"
        "│ 文档       %d 个\n"
        "│ 并发       %d 个批处理 Worker · %d 个加载线程 · 共享 1 个版面模型\n"
        "│ 页面批大小 %d 页\n"
        "│ 输出       %s\n"
        "╰────────────────────────────────────────",
        len(paths),
        num_workers,
        num_loaders,
        page_batch_size,
        output_dir,
    )

    t0 = time.perf_counter()
    tracker = _DocumentTracker(paths)
    crashed = False
    pipeline = None
    shutdown = threading.Event()
    stage_threads: List[threading.Thread] = []
    saver_threads: List[threading.Thread] = []
    executor: Optional[ThreadPoolExecutor] = None
    initialization_seconds = 0.0

    pbar = tqdm(total=len(paths), desc="Parsing", ncols=100) if show_progress else None
    try:
        if paths:
            # One shared pipeline (single layout model) for all documents.
            from venusocr.pipeline import DocumentParsingPipeline

            initialization_started = time.perf_counter()
            try:
                pipeline = DocumentParsingPipeline(config)
            except Exception as e:  # noqa: BLE001 - report as a crashed run
                logger.exception("Failed to initialize the parsing pipeline")
                tracker.errors.append(f"<pipeline-init>: {e!r}")
                for doc_idx in range(len(paths)):
                    tracker.fail_doc(doc_idx, "pipeline initialization failed")
                crashed = True
            initialization_seconds = time.perf_counter() - initialization_started

        if paths and not crashed:
            model_settings, vlm_kwargs, prompt_label = pipeline._prepare_settings()
            predict_cfg = config.predict

            # Bounded page queue: back-pressure towards the loaders.
            page_queue: "queue.Queue" = queue.Queue(maxsize=2 * page_batch_size)
            # At most num_workers page batches are in flight at once.
            in_flight = threading.Semaphore(num_workers)
            executor = ThreadPoolExecutor(
                max_workers=num_workers, thread_name_prefix="venusocr-batch"
            )

            loader_pool = _LoaderPool(len(paths))
            for _ in range(min(num_loaders, len(paths))):
                thread = threading.Thread(
                    target=_loader_worker,
                    args=(loader_pool, paths, page_queue, tracker, shutdown),
                    name="venusocr-loader",
                    daemon=True,
                )
                thread.start()
                stage_threads.append(thread)

            layout_thread = threading.Thread(
                target=_layout_worker,
                args=(
                    pipeline,
                    loader_pool,
                    page_queue,
                    page_batch_size,
                    model_settings,
                    vlm_kwargs,
                    predict_cfg,
                    prompt_label,
                    tracker,
                    executor,
                    in_flight,
                    shutdown,
                ),
                name="venusocr-layout",
                daemon=True,
            )
            layout_thread.start()
            stage_threads.append(layout_thread)

            for _ in range(num_savers):
                thread = threading.Thread(
                    target=_saver_worker,
                    args=(tracker, output_dir, input_root, config.output),
                    name="venusocr-saver",
                    daemon=True,
                )
                thread.start()
                saver_threads.append(thread)

            try:
                while tracker.processed < len(paths):
                    if pbar is not None:
                        pbar.update(tracker.processed - pbar.n)
                    if all(not t.is_alive() for t in stage_threads):
                        # Loader + layout threads only exit after every page
                        # has been accounted for; remaining documents are
                        # either already failed or queued for saving.
                        break
                    time.sleep(0.3)
            finally:
                if pbar is not None:
                    pbar.update(tracker.processed - pbar.n)
                    pbar.close()

            # Drain: stage threads first (layout waits for in-flight work),
            # then signal the savers after the last emission.
            for thread in stage_threads:
                thread.join(timeout=60)
            for _ in saver_threads:
                tracker.save_queue.put(None)
            for thread in saver_threads:
                thread.join(timeout=120)

        succeeded = tracker.succeeded
        failed = tracker.failed
        unprocessed = max(0, len(paths) - succeeded - failed)
        if unprocessed:
            crashed = True
            tracker.errors.append(
                f"<unprocessed>: {unprocessed} file(s) were not attempted or "
                "did not complete"
            )
            failed += unprocessed

        page_count = 0
        layout_regions = 0
        recognition_requests = 0
        output_blocks = 0
        stage_seconds: Dict[str, float] = {}
        if initialization_seconds:
            stage_seconds["model_initialization"] = initialization_seconds
        for doc in tracker._docs:
            if doc.metrics is None:
                continue
            page_count += doc.metrics.page_count
            layout_regions += doc.metrics.layout_regions
            recognition_requests += doc.metrics.recognition_requests
            output_blocks += doc.metrics.output_blocks
            for stage, seconds in doc.metrics.stage_seconds.items():
                stage_seconds[stage] = stage_seconds.get(stage, 0.0) + seconds

        summary = BatchSummary(
            total=len(paths),
            succeeded=succeeded,
            failed=failed,
            crashed=crashed,
            elapsed_seconds=time.perf_counter() - t0,
            page_count=page_count,
            layout_regions=layout_regions,
            recognition_requests=recognition_requests,
            output_blocks=output_blocks,
            stage_seconds=stage_seconds,
            failed_files=list(tracker.errors),
        )
    finally:
        shutdown.set()
        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=False)
        for thread in stage_threads:
            thread.join(timeout=5)
        if pipeline is not None:
            pipeline.close()

    failed_files = summary.failed_files
    if failed_files:
        logger.warning("%d file(s) failed:", len(failed_files))
        for item in failed_files[:20]:
            logger.warning("  - %s", item)
        with open(os.path.join(output_dir, "failed_files.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(failed_files) + "\n")

    with open(os.path.join(output_dir, "_DONE"), "w", encoding="utf-8") as f:
        f.write(f"total={summary.total}\n")
        f.write(f"succeeded={summary.succeeded}\n")
        f.write(f"failed={summary.failed}\n")
        f.write(f"crashed={summary.crashed}\n")
        f.write(f"elapsed_s={summary.elapsed_seconds:.0f}\n")
        f.write(f"pages={summary.page_count}\n")
        f.write(f"layout_regions={summary.layout_regions}\n")
        f.write(f"recognition_requests={summary.recognition_requests}\n")
        f.write(f"output_blocks={summary.output_blocks}\n")
        f.write(
            "avg_page_s="
            f"{summary.elapsed_seconds / summary.page_count if summary.page_count else 0:.3f}\n"
        )
        for stage, seconds in sorted(summary.stage_seconds.items()):
            f.write(f"stage_{stage}_s={seconds:.3f}\n")
        f.write(f"finished_at={time.strftime('%F %T')}\n")

    logger.info("%s", summary.format_summary())
    return summary
