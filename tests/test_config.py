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

"""Tests for venusocr.config."""

import pytest

from venusocr.config import VenusOCRConfig, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.engine.layout_backend == "transformers"
    assert cfg.engine.device == "gpu:0"
    assert cfg.engine.layout_model_name == "PP-DocLayoutV3"
    assert cfg.engine.vl_rec_backend == "vllm-server"
    assert cfg.predict.use_layout_detection is True
    assert cfg.output.save_markdown is True
    assert cfg.batch.num_workers > 0


def test_user_config_deep_merge(tmp_path):
    user_yaml = tmp_path / "my.yaml"
    user_yaml.write_text(
        "engine:\n  device: cpu\noutput:\n  save_layout_vis: false\n", encoding="utf-8"
    )
    cfg = load_config(str(user_yaml))
    assert cfg.engine.device == "cpu"
    # untouched sections inherit defaults
    assert cfg.engine.vl_rec_server_url == "http://127.0.0.1:8000/v1"
    assert cfg.output.save_layout_vis is False
    assert cfg.output.save_markdown is True


def test_dotted_overrides():
    cfg = load_config(
        overrides={
            "engine.vl_rec_server_url": "http://x:9000/v1",
            "engine.layout_backend": "transformers",
        }
    )
    assert cfg.engine.vl_rec_server_url == "http://x:9000/v1"
    assert cfg.engine.layout_backend == "transformers"


def test_invalid_layout_backend_rejected():
    with pytest.raises(ValueError):
        VenusOCRConfig.model_validate({"engine": {"layout_backend": "unknown"}})


def test_unknown_yaml_key_is_rejected(tmp_path):
    user_yaml = tmp_path / "invalid.yaml"
    user_yaml.write_text("engine:\n  device: cpu\n  typo_option: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(user_yaml))


def test_sampling_limits_are_validated():
    with pytest.raises(ValueError):
        VenusOCRConfig.model_validate(
            {"predict": {"min_pixels": 200, "max_pixels": 100}}
        )
    with pytest.raises(ValueError):
        VenusOCRConfig.model_validate({"predict": {"top_p": 1.5}})


def test_unknown_override_key_raises():
    cfg = VenusOCRConfig()
    with pytest.raises(KeyError):
        cfg.apply_overrides({"engine.no_such_key": 1})
    with pytest.raises(KeyError):
        cfg.apply_overrides({"no_such_section.x": 1})


def test_config_pickleable_roundtrip():
    cfg = load_config()
    data = cfg.model_dump()
    cfg2 = VenusOCRConfig.model_validate(data)
    assert cfg2 == cfg
