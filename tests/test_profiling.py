# Copyright (c) 2026 VenusOCR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for parse timing and human-readable summaries."""

import pytest

from venusocr.batch import BatchSummary
from venusocr.utils.profiling import ParseMetrics, format_duration


def test_format_duration_uses_readable_units():
    assert format_duration(0.0005) == "500 μs"
    assert format_duration(0.25) == "250.0 ms"
    assert format_duration(2.5) == "2.50 s"
    assert format_duration(65) == "1m 05.0s"


def test_parse_metrics_accumulates_stages_and_rates():
    metrics = ParseMetrics(
        input_path="/tmp/paper.pdf",
        page_count=4,
        layout_regions=20,
        recognition_requests=16,
        model_initialization_seconds=1.0,
        pipeline_seconds=7.0,
    )
    metrics.add_stage("layout_detection", 1.5)
    metrics.add_stage("layout_detection", 0.5)
    metrics.add_stage("result_saving", 2.0)

    assert metrics.stage_seconds["layout_detection"] == 2.0
    assert metrics.total_seconds == 9.0
    assert metrics.average_page_seconds == 2.25
    assert metrics.pages_per_second == 4 / 9
    summary = metrics.format_summary()
    assert "paper.pdf" in summary
    assert "4 页" in summary
    assert "平均每页" in summary
    assert "2.25 s" in summary
    assert metrics.to_dict()["recognition_requests"] == 16


def test_batch_summary_normalizes_worker_stages_to_wall_time():
    summary = BatchSummary(
        total=3,
        succeeded=3,
        page_count=12,
        layout_regions=60,
        recognition_requests=48,
        output_blocks=51,
        elapsed_seconds=6.0,
        stage_seconds={"layout_detection": 8.0, "vlm_recognition": 10.0},
    )

    text = summary.format_summary()
    assert "阶段耗时（墙钟分摊）" in text
    assert "墙钟统计" in text
    assert "2.00 页/s" in text
    assert "51 个输出块" in text
    assert sum(summary.stage_seconds.values()) == summary.elapsed_seconds
    assert summary.stage_seconds["vlm_recognition"] == pytest.approx(10 / 18 * 6)


def test_parse_metrics_is_available_from_public_api():
    from venusocr import ParseMetrics as PublicParseMetrics

    assert PublicParseMetrics is ParseMetrics
