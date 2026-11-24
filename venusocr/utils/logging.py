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

"""Logging helpers for VenusOCR."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
_DEBUG_LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)s:%(lineno)d │ %(message)s"
_configured = False


# Third-party loggers whose INFO chatter (e.g. "HTTP Request: ...") is
# noise for end users. They stay visible at DEBUG level.
_QUIET_LOGGERS = ("httpx", "httpcore", "openai", "urllib3")


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for CLI usage.

    Args:
        level: Log level name, e.g. ``"DEBUG"``, ``"INFO"``.
    """
    global _configured
    handler = logging.StreamHandler(sys.stderr)
    log_format = _DEBUG_LOG_FORMAT if level.upper() == "DEBUG" else _LOG_FORMAT
    handler.setFormatter(logging.Formatter(log_format, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    if level.upper() != "DEBUG":
        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring defaults if needed."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
