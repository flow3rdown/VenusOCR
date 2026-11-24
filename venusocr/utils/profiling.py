# Copyright (c) 2026 VenusOCR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Lightweight timing and throughput statistics for document parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


STAGE_LABELS = {
    "model_initialization": "模型初始化",
    "input_loading": "输入加载 / PDF 渲染",
    "layout_detection": "版面检测",
    "region_preparation": "区域裁剪与预处理",
    "vlm_recognition": "VenusOCR 识别",
    "result_assembly": "结果后处理与组装",
    "result_saving": "结果保存",
}


def format_duration(seconds: float) -> str:
    """Format a duration for concise human-readable logs."""
    seconds = max(0.0, float(seconds))
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} μs"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {remainder:04.1f}s"


@dataclass
class ParseMetrics:
    """Accumulated timings and counters for one parsed document."""

    input_path: str = ""
    page_count: int = 0
    page_batches: int = 0
    layout_regions: int = 0
    recognition_requests: int = 0
    output_blocks: int = 0
    model_initialization_seconds: float = 0.0
    pipeline_seconds: float = 0.0
    stage_seconds: Dict[str, float] = field(default_factory=dict)

    def add_stage(self, stage: str, seconds: float) -> None:
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + max(
            0.0, float(seconds)
        )

    @property
    def save_seconds(self) -> float:
        return self.stage_seconds.get("result_saving", 0.0)

    @property
    def total_seconds(self) -> float:
        """Parsing time only; model initialization is excluded on purpose."""
        return self.pipeline_seconds + self.save_seconds

    @property
    def average_page_seconds(self) -> float:
        return self.total_seconds / self.page_count if self.page_count else 0.0

    @property
    def pages_per_second(self) -> float:
        return self.page_count / self.total_seconds if self.total_seconds else 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serializable metrics snapshot."""
        return {
            "input_path": self.input_path,
            "page_count": self.page_count,
            "page_batches": self.page_batches,
            "layout_regions": self.layout_regions,
            "recognition_requests": self.recognition_requests,
            "output_blocks": self.output_blocks,
            "model_initialization_seconds": self.model_initialization_seconds,
            "pipeline_seconds": self.pipeline_seconds,
            "save_seconds": self.save_seconds,
            "total_seconds": self.total_seconds,
            "average_page_seconds": self.average_page_seconds,
            "pages_per_second": self.pages_per_second,
            "stage_seconds": dict(self.stage_seconds),
        }

    def format_summary(self, *, include_save: bool = True) -> str:
        """Build a compact multi-line document summary for terminal logs."""
        stages = [
            "input_loading",
            "layout_detection",
            "region_preparation",
            "vlm_recognition",
            "result_assembly",
        ]
        if include_save:
            stages.append("result_saving")
        path = Path(self.input_path).name if self.input_path else "<memory>"
        lines = [
            "╭─ VenusOCR · 解析完成",
            f"│ 文档       {path}",
            (
                f"│ 规模       {self.page_count} 页 · "
                f"{self.layout_regions} 个版面区域 · "
                f"{self.recognition_requests} 次识别请求 · "
                f"{self.output_blocks} 个输出块"
            ),
            "├─ 阶段耗时",
        ]
        if self.model_initialization_seconds > 0:
            lines.append(
                f"│  {STAGE_LABELS['model_initialization']:<20} "
                f"{format_duration(self.model_initialization_seconds)}"
            )
        for stage in stages:
            if stage in self.stage_seconds:
                lines.append(
                    f"│  {STAGE_LABELS[stage]:<20} "
                    f"{format_duration(self.stage_seconds[stage])}"
                )
        total = self.total_seconds if include_save else self.pipeline_seconds
        average = total / self.page_count if self.page_count else 0.0
        throughput = self.page_count / total if total else 0.0
        lines.extend(
            [
                "├─ 汇总",
                f"│  总耗时                 {format_duration(total)}",
                f"│  平均每页               {format_duration(average)}",
                f"│  处理吞吐               {throughput:.2f} 页/s",
                "╰────────────────────────────────────────",
            ]
        )
        return "\n".join(lines)
