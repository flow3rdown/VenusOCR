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

"""Model layer: local layout detection and remote VL recognition."""

from __future__ import annotations

import importlib

__all__ = [
    "LayoutDetector",
    "PaddleXLayoutDetector",
    "TransformersLayoutDetector",
    "create_layout_detector",
    "VLRecognizer",
    "GenAIClient",
]

_LAZY_ATTRS = {
    "LayoutDetector": ("layout", "LayoutDetector"),
    "PaddleXLayoutDetector": ("layout", "PaddleXLayoutDetector"),
    "TransformersLayoutDetector": ("layout", "TransformersLayoutDetector"),
    "create_layout_detector": ("layout", "create_layout_detector"),
    "VLRecognizer": ("vlm", "VLRecognizer"),
    "GenAIClient": ("vlm", "GenAIClient"),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, attr_name)


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
