# Copyright (c) 2026 VenusOCR. All Rights Reserved.
# Portions Copyright (c) PaddlePaddle Authors.
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

"""Vision-language recognition through an OpenAI-compatible vLLM server."""

from __future__ import annotations

import asyncio
import atexit
import base64
import concurrent.futures
import io
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from venusocr.utils.logging import get_logger

logger = get_logger(__name__)

# Same as upstream ``PADDLEOCR_VL_GENAI_CLIENT_BATCH_SIZE``.
DEFAULT_RECOGNITION_BATCH_SIZE = 8192
# Same as upstream ``PADDLEOCR_VL_MAX_NEW_TOKENS``; the parsing pipeline
# always passes its own default (4096), this is only a fallback.
DEFAULT_MAX_NEW_TOKENS = 8192
REQUEST_TIMEOUT_SECONDS = 600


class _AsyncThreadManager:
    """Manages an asyncio event loop running in a dedicated background thread.

    Provides a bridge between synchronous code and async operations,
    allowing sync code to submit coroutines to be executed in the
    background event loop. Vendored from paddlex ``genai.py``.
    """

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.stopped = False
        self._event_start = threading.Event()
        # Event to signal that cleanup has completed in the loop thread
        self._event_cleanup_done = threading.Event()
        # Flag to reject new tasks during shutdown
        self._shutting_down = False

    def start(self):
        """Start the background event loop thread (idempotent)."""
        if self.is_running():
            return

        # Reset state for potential restart
        self._shutting_down = False
        self.stopped = False
        self._event_start.clear()
        self._event_cleanup_done.clear()

        def _run_loop():
            # Create a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            # Signal that loop is ready to accept tasks
            self._event_start.set()
            try:
                # Run until stop() is called
                self.loop.run_forever()
            finally:
                # IMPORTANT: Cleanup runs in the loop thread after
                # run_forever() returns. This ensures all asyncio operations
                # are executed in the correct thread context.
                self._cleanup_loop_internal()
                self._event_cleanup_done.set()
                self.stopped = True

        self.thread = threading.Thread(target=_run_loop, daemon=True)
        self.thread.start()
        # Wait for the loop to be fully initialized
        self._event_start.wait()

    def _cleanup_loop_internal(self):
        """Perform cleanup within the event loop thread.

        MUST be called from the event loop thread.
        """
        if self.loop is None:
            return

        try:
            # Get all pending tasks - safe because we're in the loop thread
            pending = asyncio.all_tasks(self.loop)

            if pending:
                logger.debug("Cancelling %d pending tasks during cleanup", len(pending))
                for task in pending:
                    task.cancel()
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            # Shutdown async generators (prevents ResourceWarning)
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())

            # Shutdown the default executor and wait for threads to finish
            if hasattr(self.loop, "shutdown_default_executor"):
                self.loop.run_until_complete(self.loop.shutdown_default_executor())

        except Exception as e:
            logger.warning("Error during event loop cleanup: %s", e)
        finally:
            self.loop.close()
            logger.debug("Event loop closed successfully")

    def stop(self, timeout: float = 5.0):
        """Gracefully stop the event loop.

        Waits for pending tasks to complete (up to ``timeout``), cancels the
        rest, then stops the loop and joins the background thread.
        """
        if not self.is_running():
            return

        # Reject new task submissions
        self._shutting_down = True

        async def _graceful_shutdown():
            current_task = asyncio.current_task()
            pending = [
                t
                for t in asyncio.all_tasks(self.loop)
                if t is not current_task and not t.done()
            ]

            if not pending:
                return

            done, still_pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.ALL_COMPLETED
            )

            if still_pending:
                logger.warning(
                    "Graceful shutdown: cancelling %d tasks that did not "
                    "complete within %ss",
                    len(still_pending),
                    timeout,
                )
                for task in still_pending:
                    task.cancel()
                await asyncio.gather(*still_pending, return_exceptions=True)

        try:
            future = asyncio.run_coroutine_threadsafe(_graceful_shutdown(), self.loop)
            future.result(timeout=timeout + 2.0)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Graceful shutdown timed out after %ss, forcing loop stop",
                timeout + 2.0,
            )
        except Exception as e:
            logger.warning("Error during graceful shutdown: %s", e)

        # Signal the event loop to stop (thread-safe)
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:
            # Loop may already be closed
            pass

        # Wait for cleanup to complete in the loop thread
        cleanup_completed = self._event_cleanup_done.wait(timeout=5.0)
        if not cleanup_completed:
            logger.warning("Event loop cleanup did not complete within 5s")

        # Wait for the background thread to terminate
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(
                    "Background thread did not terminate in time. "
                    "Some resources may not be properly released."
                )

        self.loop = None
        self.thread = None

    def run_async(self, coro):
        """Submit a coroutine to the background event loop.

        Returns:
            concurrent.futures.Future for the coroutine.
        """
        if not self.is_running():
            raise RuntimeError("Event loop is not running")

        if self._shutting_down:
            raise RuntimeError(
                "Event loop is shutting down, cannot accept new tasks. "
                "Please ensure all async operations complete before calling stop()."
            )

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future

    def is_running(self):
        """Check if the event loop is currently running and accepting tasks."""
        return self.loop is not None and not self.loop.is_closed() and not self.stopped

    @property
    def is_shutting_down(self):
        """Check if the event loop is in the process of shutting down."""
        return self._shutting_down


_async_thread_manager = None


def get_async_manager() -> _AsyncThreadManager:
    """Return the global background event-loop manager."""
    global _async_thread_manager
    if _async_thread_manager is None:
        _async_thread_manager = _AsyncThreadManager()
    return _async_thread_manager


def start_aio_loop():
    """Start the global async event loop if not already running."""
    manager = get_async_manager()
    if not manager.is_running():
        manager.start()
        atexit.register(manager.stop)


def close_aio_loop(timeout: float = 5.0):
    """Gracefully close the global async event loop."""
    manager = get_async_manager()
    if manager.is_running():
        manager.stop(timeout=timeout)


def run_async(coro, return_future=False, timeout=None):
    """Execute a coroutine in the background event loop.

    Args:
        coro: The coroutine to execute.
        return_future: If True, return a Future immediately without waiting.
        timeout: Maximum seconds to wait for completion (blocking mode only).
    """
    manager = get_async_manager()

    if not manager.is_running():
        start_aio_loop()

    if not manager.is_running():
        raise RuntimeError("Failed to start event loop")

    if manager.is_shutting_down:
        raise RuntimeError("Event loop is shutting down, cannot accept new tasks")

    future = manager.run_async(coro)

    if return_future:
        return future

    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning("Task timed out after %s seconds", timeout)
        raise
    except Exception as e:
        logger.error("Task failed with error: %s", e)
        raise


class GenAIClient:
    """Minimal client for an OpenAI-compatible (vLLM) chat-completion server.

    When ``model_name`` is ``None`` the first entry of ``GET /v1/models``
    is used, so the server does not need to expose any specific model name.
    """

    def __init__(
        self, base_url: str, max_concurrency: int = 200, model_name: Optional[str] = None,
        api_key: Optional[str] = None, **kwargs,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "The `openai` package is required for VL recognition. "
                "Install it with `pip install openai`."
            ) from e

        self._max_concurrency = max_concurrency
        if api_key is None:
            api_key = "null"

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, **kwargs)

        if model_name is None:
            model_name = run_async(self._get_model_name(), timeout=10)
        self._model_name = model_name

        self._semaphore = asyncio.Semaphore(self._max_concurrency)

    @property
    def openai_client(self):
        """The underlying ``AsyncOpenAI`` client."""
        return self._client

    @property
    def model_name(self) -> str:
        """The served model name used in requests."""
        return self._model_name

    def create_chat_completion(self, messages, *, return_future=False, **kwargs):
        """Create a chat completion (optionally as a background future)."""
        async def _create_chat_completion_with_semaphore(*args, **kwargs):
            async with self._semaphore:
                return await self._client.chat.completions.create(
                    *args,
                    **kwargs,
                )

        return run_async(
            _create_chat_completion_with_semaphore(
                model=self._model_name,
                messages=messages,
                **kwargs,
            ),
            return_future=return_future,
        )

    def close(self):
        """Close the underlying HTTP client."""
        run_async(self._client.close(), timeout=5)

    async def _get_model_name(self):
        try:
            models = await self._client.models.list()
        except Exception as e:
            raise RuntimeError(
                f"Failed to get the model list from the OpenAI-compatible server: {e}"
            ) from e
        return models.data[0].id


def _image_to_data_url(image, image_format: str = "JPEG") -> str:
    """Convert an image (path / URL / BGR numpy array) to a data URL."""
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            return image
        from PIL import Image

        with Image.open(image) as img:
            img = img.convert("RGB")
            with io.BytesIO() as buf:
                img.save(buf, format=image_format)
                return (
                    f"data:image/{image_format.lower()};base64,"
                    + base64.b64encode(buf.getvalue()).decode("ascii")
                )
    elif isinstance(image, np.ndarray):
        import cv2
        from PIL import Image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        with io.BytesIO() as buf:
            img.save(buf, format=image_format)
            return (
                f"data:image/{image_format.lower()};base64,"
                + base64.b64encode(buf.getvalue()).decode("ascii")
            )
    raise TypeError(f"Not supported image type: {type(image)}")


class VLRecognizer:
    """Region-level text / table / formula recognition via a VenusOCR server.

    Sends each block image with its prompt to an OpenAI-compatible
    (vLLM) endpoint and returns the recognized content. Request semantics
    are identical to the upstream ``DocVLMGenAIClientPredictor`` with the
    ``vllm-server`` backend.

    Args:
        server_url: Base URL of the OpenAI-compatible server
            (e.g. ``http://127.0.0.1:8000/v1``).
        model_name: Served model name. ``None`` auto-fetches it from the
            server's ``/v1/models``.
        api_key: API key for the server (``"null"`` by default).
        max_concurrency: Maximum number of concurrent requests.
        batch_size: Number of recognition items submitted per batch
            (8192 by default, same as upstream PaddleOCR-VL).
    """

    def __init__(
        self,
        server_url: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrency: int = 200,
        batch_size: int = DEFAULT_RECOGNITION_BATCH_SIZE,
    ) -> None:
        self.batch_size = batch_size
        self._client = GenAIClient(
            base_url=server_url,
            max_concurrency=max_concurrency,
            model_name=model_name,
            api_key=api_key,
        )

    @property
    def client(self) -> GenAIClient:
        """The underlying :class:`GenAIClient`."""
        return self._client

    def close(self) -> None:
        """Close the underlying client."""
        self._client.close()

    def recognize(
        self,
        items: List[Dict[str, Any]],
        *,
        max_new_tokens: Optional[int] = None,
        skip_special_tokens: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Recognize a list of ``{"image": ..., "query": ...}`` items.

        Returns:
            A list of dicts with the input keys plus ``"result"`` (the
            recognized text), in the same order as ``items``.
        """
        results: List[Dict[str, Any]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            preds = self._process_batch(
                batch,
                max_new_tokens=max_new_tokens,
                skip_special_tokens=skip_special_tokens,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            for item, pred in zip(batch, preds):
                record = dict(item)
                if isinstance(record.get("image"), str):
                    record["input_path"] = record["image"]
                record["result"] = pred
                results.append(record)
        return results

    # alias kept for familiarity with the upstream predictor interface
    predict = recognize

    def _build_request_specs(
        self,
        data: List[Dict[str, Any]],
        max_new_tokens,
        skip_special_tokens,
        repetition_penalty,
        temperature,
        top_p,
        min_pixels,
        max_pixels,
    ):
        specs = []
        for item in data:
            image_url = _image_to_data_url(item["image"], image_format="JPEG")

            request_kwargs: Dict[str, Any] = {
                "temperature": 0 if temperature is None else temperature,
            }
            if top_p is not None:
                request_kwargs["top_p"] = top_p

            if max_new_tokens is not None:
                request_kwargs["max_completion_tokens"] = max_new_tokens
            else:
                request_kwargs["max_completion_tokens"] = DEFAULT_MAX_NEW_TOKENS

            request_kwargs["extra_body"] = {}
            if skip_special_tokens is not None:
                request_kwargs["extra_body"]["skip_special_tokens"] = skip_special_tokens
            if repetition_penalty is not None:
                request_kwargs["extra_body"]["repetition_penalty"] = repetition_penalty
            if min_pixels is not None or max_pixels is not None:
                mm_processor_kwargs = {}
                if min_pixels is not None:
                    mm_processor_kwargs["min_pixels"] = min_pixels
                if max_pixels is not None:
                    mm_processor_kwargs["max_pixels"] = max_pixels
                request_kwargs["extra_body"]["mm_processor_kwargs"] = mm_processor_kwargs

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": item["query"]},
                    ],
                }
            ]
            specs.append((messages, request_kwargs))
        return specs

    def _process_batch(self, data, **sampling_kwargs) -> List[str]:
        specs = self._build_request_specs(data, **sampling_kwargs)
        futures = []
        try:
            for messages, request_kwargs in specs:
                future = self._client.create_chat_completion(
                    messages,
                    return_future=True,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    **request_kwargs,
                )
                futures.append(future)
            results = []
            for future in futures:
                result = future.result()
                results.append(result.choices[0].message.content)
            return results
        except Exception:
            # Cancel all pending futures to avoid wasting resources
            for future in futures:
                if not future.done():
                    future.cancel()
            raise
