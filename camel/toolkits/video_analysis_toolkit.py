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

import tempfile
from pathlib import Path
from typing import List, Optional

import ffmpeg
from PIL import Image
from scenedetect import (  # type: ignore[import-untyped]
    SceneManager,
    VideoManager,
)
from scenedetect.detectors import (  # type: ignore[import-untyped]
    ContentDetector,
)

from camel.agents import ChatAgent
from camel.configs import QwenConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory, OpenAIAudioModels
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.types import ModelPlatformType, ModelType
from camel.utils import dependencies_required
from loguru import logger

from .video_download_toolkit import (
    VideoDownloaderToolkit,
    _capture_screenshot,
)

import os


class VideoAnalysisToolkit(BaseToolkit):


    def __init__(
        self,
        download_directory: Optional[str] = None,
        vllm_url: Optional[str] = None,
        vllm_api_key: str = "EMPTY",
        vllm_model: Optional[str] = None,
    ):
        """
        Args:
            download_directory: Where to cache downloaded videos.
            vllm_url: OpenAI-compatible vLLM endpoint serving a video-capable
                VL model (e.g. Qwen2.5-VL, Qwen3-VL). When set, video questions
                go to this endpoint instead of Gemini.
            vllm_api_key: API key for the vLLM endpoint (typically "EMPTY").
            vllm_model: Model name on the vLLM server. If None, auto-discovered
                via /v1/models on first use.
        """
        self.video_downloader_toolkit = VideoDownloaderToolkit(
            download_directory=download_directory
        )
        self.vllm_url = vllm_url
        self.vllm_api_key = vllm_api_key
        self._vllm_model = vllm_model
        self._vllm_client = None  # lazy

    def _get_vllm_client_and_model(self):
        from openai import OpenAI
        if self._vllm_client is None:
            self._vllm_client = OpenAI(api_key=self.vllm_api_key, base_url=self.vllm_url)
        if self._vllm_model is None:
            self._vllm_model = self._vllm_client.models.list().data[0].id
        return self._vllm_client, self._vllm_model

    @staticmethod
    def _to_video_url(video_path: str) -> str:
        """Normalize a path/URL into something vLLM's video_url field accepts."""
        if video_path.startswith(("http://", "https://", "file://", "data:")):
            return video_path
        return "file://" + os.path.abspath(video_path)

    def _ask_via_vllm(self, video_path: str, question: str) -> str:
        client, model = self._get_vllm_client_and_model()
        video_url = self._to_video_url(video_path)
        logger.debug(f"Video Q via vLLM ({model}): {video_url}")
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }],
        )
        return response.choices[0].message.content

    def _ask_via_gemini(self, video_path: str, question: str) -> str:
        os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
        from google import genai
        from google.genai import types
        client = genai.Client()
        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=types.Content(parts=[
                types.Part(text=question),
                types.Part(file_data=types.FileData(file_uri=video_path)),
            ]),
        )
        logger.debug(f"Video analysis response from gemini: {response.text}")
        return response.text

    def ask_question_about_video(self, video_path: str, question: str) -> str:
        r"""Ask a question about the video.

        Args:
            video_path (str): The path or URL of the video file.
            question (str): The question to ask about the video.

        Returns:
            str: The answer to the question.
        """
        if self.vllm_url:
            return self._ask_via_vllm(video_path, question)
        return self._ask_via_gemini(video_path, question)
    
        
    def get_tools(self) -> List[FunctionTool]:
        """
        Get the tools in the toolkit.
        """
        return [FunctionTool(self.ask_question_about_video)]
    

