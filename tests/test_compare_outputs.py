# Copyright (c) 2026 VenusOCR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the artifact comparison helper."""

import importlib.util
from pathlib import Path


def _load_compare_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "compare_outputs.py"
    spec = importlib.util.spec_from_file_location("compare_outputs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_outputs_supports_artifact_subdirectories(tmp_path):
    module = _load_compare_module()
    reference = tmp_path / "reference" / "markdown"
    candidate = tmp_path / "candidate" / "markdown"
    reference.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (reference / "page.md").write_text("same", encoding="utf-8")
    (candidate / "page.md").write_text("same", encoding="utf-8")

    report = module._compare_kind(
        tmp_path / "reference", tmp_path / "candidate", ".md", set()
    )

    assert report["identical"] == 1
    assert report["different"] == []


def test_compare_outputs_can_ignore_metadata_keys(tmp_path):
    module = _load_compare_module()
    reference = tmp_path / "reference" / "json"
    candidate = tmp_path / "candidate" / "json"
    reference.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (reference / "page.json").write_text(
        '{"content": "same", "model_settings": {"a": 1}}', encoding="utf-8"
    )
    (candidate / "page.json").write_text(
        '{"content": "same", "model_settings": {"b": 2}}', encoding="utf-8"
    )

    report = module._compare_kind(
        tmp_path / "reference",
        tmp_path / "candidate",
        ".json",
        {"model_settings"},
    )

    assert report["identical"] == 1
