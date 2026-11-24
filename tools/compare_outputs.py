#!/usr/bin/env python3
# Copyright (c) 2026 VenusOCR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compare PaddleOCR reference artifacts with VenusOCR artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _files(root: Path, suffix: str) -> dict[str, Path]:
    artifact_dir = root / ("markdown" if suffix == ".md" else "json")
    if artifact_dir.is_dir():
        root = artifact_dir
    return {
        str(path.relative_to(root)): path
        for path in root.rglob(f"*{suffix}")
        if path.is_file()
    }


def _remove_keys(value, ignored_keys: set[str]):
    if isinstance(value, dict):
        return {
            key: _remove_keys(item, ignored_keys)
            for key, item in value.items()
            if key not in ignored_keys
        }
    if isinstance(value, list):
        return [_remove_keys(item, ignored_keys) for item in value]
    return value


def _compare_kind(
    reference_root: Path,
    candidate_root: Path,
    suffix: str,
    ignored_json_keys: set[str],
) -> dict:
    reference = _files(reference_root, suffix)
    candidate = _files(candidate_root, suffix)
    common = sorted(reference.keys() & candidate.keys())
    different = []
    for name in common:
        if suffix == ".json":
            left = _remove_keys(
                json.loads(reference[name].read_text(encoding="utf-8")),
                ignored_json_keys,
            )
            right = _remove_keys(
                json.loads(candidate[name].read_text(encoding="utf-8")),
                ignored_json_keys,
            )
        else:
            left = reference[name].read_bytes()
            right = candidate[name].read_bytes()
        if left != right:
            different.append(name)
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "missing": sorted(reference.keys() - candidate.keys()),
        "unexpected": sorted(candidate.keys() - reference.keys()),
        "different": different,
        "identical": len(common) - len(different),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="PaddleOCR output directory")
    parser.add_argument("candidate", help="VenusOCR output directory")
    parser.add_argument("--output", default="output_diff.json")
    parser.add_argument(
        "--kind",
        choices=["all", "markdown", "json"],
        default="all",
        help="Artifact type to compare",
    )
    parser.add_argument(
        "--ignore-json-key",
        action="append",
        default=[],
        help="Ignore a JSON key at any nesting level; may be supplied repeatedly",
    )
    args = parser.parse_args()

    reference = Path(args.reference)
    candidate = Path(args.candidate)
    if not reference.is_dir() or not candidate.is_dir():
        raise ValueError("reference and candidate must both be directories")

    ignored_keys = set(args.ignore_json_key)
    report = {}
    if args.kind in {"all", "markdown"}:
        report["markdown"] = _compare_kind(
            reference, candidate, ".md", ignored_keys
        )
    if args.kind in {"all", "json"}:
        report["json"] = _compare_kind(reference, candidate, ".json", ignored_keys)
    report["passed"] = all(
        not section["missing"]
        and not section["unexpected"]
        and not section["different"]
        for section in report.values()
    )
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
