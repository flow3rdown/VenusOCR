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

"""Helpers to launch a vLLM OpenAI-compatible server for the VenusOCR model."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Optional

from venusocr.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SERVED_MODEL_NAME = "VenusOCR"


def build_vllm_command(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
    max_num_batched_tokens: int = 16384,
    enable_prefix_caching: bool = False,
    mm_processor_cache_gb: float = 0,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build the ``vllm serve`` command line with recommended OCR defaults.

    Args:
        model_path: Local path (or HF id) of the VenusOCR checkpoint.
        host: Bind host.
        port: Bind port.
        served_model_name: Name exposed through the OpenAI API.
        max_num_batched_tokens: vLLM batching budget.
        enable_prefix_caching: Whether to enable prefix caching
            (disabled by default; OCR workloads rarely benefit).
        mm_processor_cache_gb: Multi-modal processor cache size in GB.
        extra_args: Additional raw arguments appended verbatim.

    Returns:
        The command as a list of strings.
    """
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--trust-remote-code",
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--mm-processor-cache-gb",
        str(mm_processor_cache_gb),
    ]
    if not enable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def launch_server(cmd: List[str]) -> int:
    """Run a vLLM server command in the foreground.

    Args:
        cmd: Command produced by :func:`build_vllm_command`.

    Returns:
        The server process exit code.
    """
    if shutil.which("vllm") is None:
        raise RuntimeError(
            "vllm executable not found. Install it first, e.g. "
            "`pip install vllm` or `pip install venusocr[serve]`."
        )
    logger.info("Launching server: %s", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("VLLM_PLUGINS", "venusocr")
    return subprocess.call(cmd, env=env)
