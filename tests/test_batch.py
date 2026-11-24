# Copyright (c) 2026 VenusOCR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for batch-runner invariants that do not require model inference."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from venusocr.batch import run_batch
from venusocr.config import load_config


def test_batch_rejects_non_positive_workers(tmp_path):
    with pytest.raises(ValueError):
        run_batch([], load_config(), str(tmp_path), num_workers=0, show_progress=False)


def _write_png(path, shape=(32, 24, 3)):
    import cv2

    cv2.imwrite(str(path), np.zeros(shape, dtype=np.uint8))


def _write_pdf(path, num_pages):
    import fitz

    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page(width=200, height=100)
    doc.save(str(path))
    doc.close()


class _FakePipeline:
    """Model-free stand-in verifying the shared-pipeline batch contract."""

    instances = 0
    batch_doc_sets = []
    _lock = threading.Lock()

    def __init__(self, config):
        with self._lock:
            type(self).instances += 1
        self.layout_detector = SimpleNamespace(
            predict=lambda images, **kwargs: [{"boxes": []} for _ in images]
        )

    def _prepare_settings(self):
        return {"use_layout_detection": True}, {}, "text"

    def _process_vlm(
        self,
        batch_data,
        image_arrays,
        layout_det_results,
        imgs_in_doc,
        model_settings,
        vlm_kwargs,
        cfg,
    ):
        from venusocr.results.page import PageResult

        with self._lock:
            type(self).batch_doc_sets.append(set(batch_data.input_paths))
        page_results = []
        for input_path, page_index, page_count, image, layout_res in zip(
            batch_data.input_paths,
            batch_data.page_indexes,
            batch_data.page_counts,
            image_arrays,
            layout_det_results,
        ):
            page_results.append(
                PageResult(
                    {
                        "input_path": input_path,
                        "page_index": page_index,
                        "page_count": page_count,
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "doc_preprocessor_res": {"output_img": image},
                        "layout_det_res": layout_res,
                        "table_res_list": [],
                        "parsing_res_list": [],
                        "spotting_res": {},
                        "imgs_in_doc": [],
                        "model_settings": {"markdown_ignore_labels": []},
                    }
                )
            )
        return page_results, [0] * len(image_arrays)

    def close(self):
        pass


def test_run_batch_shared_pipeline_flattens_and_regroups(tmp_path, monkeypatch):
    """Pages are flattened across documents, batched together on one shared
    pipeline instance, and results are saved per document."""
    import venusocr.pipeline

    _FakePipeline.instances = 0
    _FakePipeline.batch_doc_sets = []
    monkeypatch.setattr(venusocr.pipeline, "DocumentParsingPipeline", _FakePipeline)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for i in range(3):
        _write_png(input_dir / f"img{i}.png")
    _write_pdf(input_dir / "doc.pdf", num_pages=3)
    (input_dir / "broken.pdf").write_bytes(b"this is not a pdf")

    output_dir = tmp_path / "outputs"
    config = load_config(
        overrides={
            "batch.page_batch_size": 4,
            "batch.loader_workers": 2,
            "batch.saver_workers": 2,
            "output.save_layout_vis": False,
        }
    )
    paths = sorted(str(p) for p in input_dir.iterdir())
    summary = run_batch(
        paths,
        config,
        str(output_dir),
        num_workers=2,
        input_root=str(input_dir),
        show_progress=False,
    )

    assert not summary.crashed
    assert summary.total == 5
    assert summary.succeeded == 4
    assert summary.failed == 1
    assert summary.page_count == 6
    # One shared pipeline instance for the whole run.
    assert _FakePipeline.instances == 1
    # Pages from different documents were packed into the same batch.
    assert any(len(paths) > 1 for paths in _FakePipeline.batch_doc_sets)
    # Per-document outputs were written.
    for stem in ("img0", "img1", "img2", "doc"):
        assert (output_dir / stem / f"{stem}.md").is_file()
    assert not (output_dir / "broken").exists()
    assert (output_dir / "_DONE").is_file()
    assert (output_dir / "failed_files.txt").is_file()
