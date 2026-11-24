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

"""vLLM out-of-tree model plugin for VenusOCR.

This module is registered under the ``vllm.general_plugins`` entry point
(see ``pyproject.toml``). vLLM loads it automatically at import time, so
installing ``venusocr[serve]`` is enough for ``vllm serve`` / the offline
``LLM`` API to resolve the ``VenusOCRForConditionalGeneration`` architecture
without patching the vLLM source tree.
"""

MODEL_ARCH = "VenusOCRForConditionalGeneration"
MODEL_IMPL = "venusocr.vllm_plugin.venusocr:VenusOCRForConditionalGeneration"


def register() -> None:
    """Register VenusOCR with vLLM's model registry (idempotent)."""
    from vllm import ModelRegistry

    if MODEL_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(MODEL_ARCH, MODEL_IMPL)
