# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
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
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import json
import os
import threading
import time
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Type, Union

import httpx
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel, ValidationError

from camel.logger import get_logger
from camel.messages import OpenAIMessage
from camel.models._utils import try_modify_message_with_format
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
)
from camel.utils.replay_capture import OpenAIReplayCapture

logger = get_logger(__name__)

# Per-call text-to-text latency sidecar. CAMEL's trajectory keeps only role/
# content messages, so there is no per-LLM-call slot in it; when OWL_T2T_LOG is
# set we append one JSON line per request (latency + token usage + response id)
# so a run can be joined back to the message trajectory afterwards.
#
# All concurrent GAIA workers append to the SAME file, so each row carries a
# `pid` (the ProcessPoolExecutor worker) and `task_id` (the worker's current
# GAIA task, stamped into OWL_T2T_TASK by the runner) — otherwise interleaved
# rows can't be attributed to a task. The threading.Lock only serializes threads
# within one process; cross-process append integrity relies on POSIX O_APPEND
# atomicity (rows are small, well under PIPE_BUF).
_T2T_LOCK = threading.Lock()


def _record_t2t(model_type: Any, latency_s: float, response: Any) -> None:
    path = os.environ.get("OWL_T2T_LOG")
    if not path:
        return
    usage = getattr(response, "usage", None)  # None for streaming responses
    row = {
        "ts": time.time(),
        "pid": os.getpid(),                          # worker process (concurrency slot)
        "task_id": os.environ.get("OWL_T2T_TASK"),   # this worker's current GAIA task
        "t2t_latency_s": latency_s,
        "model": str(model_type),
        "response_id": getattr(response, "id", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    with _T2T_LOCK:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


class OpenAICompatibleModel(BaseModelBackend):
    r"""Constructor for model backend supporting OpenAI compatibility.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`{}` will be used. (default: :obj:`None`)
        api_key (str): The API key for authenticating with the model service.
        url (str): The url to the model service.
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)
        timeout (Optional[float], optional): The timeout value in seconds for
            API calls. If not provided, will fall back to the MODEL_TIMEOUT
            environment variable or default to 180 seconds.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        timeout: Optional[float] = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_COMPATIBILITY_API_KEY")
        url = url or os.environ.get("OPENAI_COMPATIBILITY_API_BASE_URL")
        timeout = timeout or float(os.environ.get("MODEL_TIMEOUT", 180))
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter, timeout
        )
        replay_capture = OpenAIReplayCapture(self)
        self._client = OpenAI(
            timeout=self._timeout,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
            http_client=httpx.Client(event_hooks=replay_capture.sync_hooks),
        )

        self._async_client = AsyncOpenAI(
            timeout=self._timeout,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
            http_client=httpx.AsyncClient(event_hooks=replay_capture.async_hooks),
        )

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of OpenAI chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the
                response.
            tools (Optional[List[Dict[str, Any]]]): The schema of the tools to
                use for the request.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        response_format = response_format or self.model_config_dict.get(
            "response_format", None
        )
        if response_format:
            return self._request_parse(messages, response_format, tools)
        else:
            return self._request_chat_completion(messages, tools)

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Runs inference of OpenAI chat completion in async mode.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the
                response.
            tools (Optional[List[Dict[str, Any]]]): The schema of the tools to
                use for the request.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `AsyncStream[ChatCompletionChunk]` in the stream mode.
        """
        response_format = response_format or self.model_config_dict.get(
            "response_format", None
        )
        if response_format:
            return await self._arequest_parse(messages, response_format, tools)
        else:
            return await self._arequest_chat_completion(messages, tools)

    def _request_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        request_config = self.model_config_dict.copy()

        if tools:
            request_config["tools"] = tools

        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )
        _record_t2t(self.model_type, time.monotonic() - t0, response)
        return response

    async def _arequest_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        request_config = self.model_config_dict.copy()

        if tools:
            request_config["tools"] = tools

        t0 = time.monotonic()
        response = await self._async_client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )
        _record_t2t(self.model_type, time.monotonic() - t0, response)
        return response

    def _request_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        import copy

        request_config = copy.deepcopy(self.model_config_dict)
        # Remove stream from request_config since OpenAI does not support it
        # when structured response is used
        request_config["response_format"] = response_format
        request_config.pop("stream", None)
        if tools is not None:
            request_config["tools"] = tools

        try:
            return self._client.beta.chat.completions.parse(
                messages=messages,
                model=self.model_type,
                **request_config,
            )
        except (ValidationError, JSONDecodeError) as e:
            logger.warning(
                f"Format validation error: {e}. "
                f"Attempting fallback with JSON format."
            )
            try_modify_message_with_format(messages[-1], response_format)
            request_config["response_format"] = {"type": "json_object"}
            try:
                return self._client.beta.chat.completions.parse(
                    messages=messages,
                    model=self.model_type,
                    **request_config,
                )
            except Exception as e:
                logger.error(f"Fallback attempt also failed: {e}")
                raise

    async def _arequest_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        import copy

        request_config = copy.deepcopy(self.model_config_dict)
        # Remove stream from request_config since OpenAI does not support it
        # when structured response is used
        request_config["response_format"] = response_format
        request_config.pop("stream", None)
        if tools is not None:
            request_config["tools"] = tools

        try:
            return await self._async_client.beta.chat.completions.parse(
                messages=messages,
                model=self.model_type,
                **request_config,
            )
        except (ValidationError, JSONDecodeError) as e:
            logger.warning(
                f"Format validation error: {e}. "
                f"Attempting fallback with JSON format."
            )
            try_modify_message_with_format(messages[-1], response_format)
            request_config["response_format"] = {"type": "json_object"}
            try:
                return await self._async_client.beta.chat.completions.parse(
                    messages=messages,
                    model=self.model_type,
                    **request_config,
                )
            except Exception as e:
                logger.error(f"Fallback attempt also failed: {e}")
                raise

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            OpenAITokenCounter: The token counter following the model's
                tokenization style.
        """

        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)

    def check_model_config(self):
        pass
