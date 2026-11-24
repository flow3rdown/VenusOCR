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

"""Configuration management for VenusOCR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class _StrictConfigModel(BaseModel):
    """Base for user-facing configuration sections."""

    model_config = ConfigDict(extra="forbid")


class EngineConfig(_StrictConfigModel):
    """Inference engine settings (layout detection + VL recognition backend)."""

    layout_backend: Literal["paddlex", "transformers"] = "transformers"
    layout_model_name: str = "PP-DocLayoutV3"
    layout_model_dir: Optional[str] = None
    device: str = "gpu:0"
    layout_batch_size: int = Field(default=8, ge=1)
    # None selects the pinned PaddleOCR/PaddleX 3.6.0 compatibility defaults.
    layout_threshold: Optional[Union[float, Dict[int, float]]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[Union[float, List[float]]] = None
    layout_merge_bboxes_mode: Optional[Union[str, Dict[int, str]]] = None
    # VL recognition server (OpenAI-compatible, e.g. vLLM).
    vl_rec_backend: Literal["vllm-server"] = "vllm-server"
    vl_rec_server_url: str = "http://127.0.0.1:8000/v1"
    vl_rec_api_model_name: Optional[str] = None
    vl_rec_api_key: Optional[str] = None
    vl_rec_max_concurrency: int = Field(default=200, ge=1)


class PredictConfig(_StrictConfigModel):
    """Per-document prediction switches and sampling parameters."""

    use_layout_detection: bool = True
    use_chart_recognition: bool = False
    use_seal_recognition: bool = False
    use_ocr_for_image_block: bool = False
    # ``ocr`` is retained as a backward-compatible alias of ``text``.
    prompt_label: Optional[Literal["text", "ocr", "formula", "table"]] = None
    layout_shape_mode: Literal["auto", "rect", "quad", "poly"] = "auto"
    merge_layout_blocks: Optional[bool] = None
    format_block_content: Optional[bool] = None
    markdown_ignore_labels: List[str] = Field(default_factory=list)
    # Sampling related (None means the pipeline/server default).
    repetition_penalty: Optional[float] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    min_pixels: Optional[int] = Field(default=None, ge=1)
    max_pixels: Optional[int] = Field(default=None, ge=1)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_pixel_range(self):
        if (
            self.use_chart_recognition
            or self.use_seal_recognition
            or self.use_ocr_for_image_block
        ):
            raise ValueError(
                "VenusOCR only supports text/formula/table recognition; "
                "image, chart, seal, and spotting regions are exported as images"
            )
        if (
            self.min_pixels is not None
            and self.max_pixels is not None
            and self.min_pixels > self.max_pixels
        ):
            raise ValueError("predict.min_pixels cannot exceed predict.max_pixels")
        return self


class OutputConfig(_StrictConfigModel):
    """Which artifacts to persist for each parsed document.

    Every document gets its own ``<output_dir>/<stem>/`` sub-directory
    holding one combined markdown file and one layout-visualization PDF.
    """

    save_markdown: bool = True
    save_layout_vis: bool = True
    pretty_markdown: bool = False


class BatchConfig(_StrictConfigModel):
    """Settings for concurrent directory parsing.

    Batch parsing uses a single shared pipeline (one layout model) with a
    page-flattened thread pipeline: documents are rendered page by page,
    packed into batches of ``page_batch_size`` pages across document
    boundaries, and processed by ``num_workers`` concurrent batch workers.
    """

    num_workers: int = Field(default=8, ge=1)
    page_batch_size: int = Field(default=64, ge=1)
    loader_workers: int = Field(default=2, ge=1)
    saver_workers: int = Field(default=2, ge=1)


class VenusOCRConfig(_StrictConfigModel):
    """Top-level VenusOCR configuration."""

    engine: EngineConfig = Field(default_factory=EngineConfig)
    predict: PredictConfig = Field(default_factory=PredictConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)

    def apply_overrides(self, overrides: Dict[str, Any]) -> "VenusOCRConfig":
        """Return a new config with dotted-path overrides applied.

        Args:
            overrides: Mapping such as ``{"engine.vl_rec_server_url": "http://..."}``.

        Returns:
            A validated copy of the config with overrides applied.
        """
        data = self.model_dump()
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            node = data
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise KeyError(f"Unknown config section in key: {dotted_key!r}")
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f"Unknown config key: {dotted_key!r}")
            node[parts[-1]] = value
        return VenusOCRConfig.model_validate(data)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> VenusOCRConfig:
    """Load configuration from a YAML file, falling back to package defaults.

    Args:
        config_path: Optional path to a user YAML config. Keys missing from the
            user file inherit the package defaults (deep-merged).
        overrides: Optional dotted-path overrides applied last.

    Returns:
        The merged and validated configuration.
    """
    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f) or {}

    merged = defaults
    if config_path:
        user_path = Path(config_path)
        if not user_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(user_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError("The root of a VenusOCR config file must be a mapping")
        merged = _deep_merge(defaults, user_cfg)

    config = VenusOCRConfig.model_validate(merged)
    if overrides:
        config = config.apply_overrides(overrides)
    return config


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``extra`` into a copy of ``base``."""
    result = dict(base)
    for key, value in extra.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
