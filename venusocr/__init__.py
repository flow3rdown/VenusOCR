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

"""VenusOCR - Document parsing powered by the VenusOCR vision-language model."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from venusocr.version import __version__

__author__ = "VenusOCR"

__all__ = [
    "__version__",
    "VenusOCR",
    "parse",
    "DocumentResult",
    "VenusOCRConfig",
    "load_config",
    "run_batch",
    "BatchSummary",
    "ParseMetrics",
]


_LAZY_ATTRS = {
    "VenusOCR": ("api", "VenusOCR"),
    "parse": ("api", "parse"),
    "DocumentResult": ("results.document", "DocumentResult"),
    "VenusOCRConfig": ("config", "VenusOCRConfig"),
    "load_config": ("config", "load_config"),
    "run_batch": ("batch", "run_batch"),
    "BatchSummary": ("batch", "BatchSummary"),
    "ParseMetrics": ("utils.profiling", "ParseMetrics"),
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


if TYPE_CHECKING:  # pragma: no cover
    from venusocr.api import VenusOCR, parse
    from venusocr.batch import BatchSummary, run_batch
    from venusocr.config import VenusOCRConfig, load_config
    from venusocr.results.document import DocumentResult
    from venusocr.utils.profiling import ParseMetrics
