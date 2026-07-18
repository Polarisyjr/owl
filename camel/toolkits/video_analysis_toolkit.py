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
from urllib.parse import urlparse

import ffmpeg
import requests
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
from camel.models import BaseModelBackend, ModelFactory, OpenAIAudioModels
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
        model: Optional[BaseModelBackend] = None,
    ):
        self.video_downloader_toolkit = VideoDownloaderToolkit(download_directory=download_directory)
        self.model = model


    def ask_question_about_video(
        self, video_path: str, question: str, sys_prompt: Optional[str] = None
    ) -> str:
        r"""Answers video questions with optional custom instructions.

        Args:
            video_path (str): Local path or URL to a video file.
            question (str): Query about the video content.
            sys_prompt (Optional[str]): Custom system prompt for the analysis.
                (default: :obj:`None`)

        Returns:
            str: Detailed answer based on video understanding.
        """
        logger.info(
            f"Calling video analysis toolkit with question: {question} "
            f"and video path: {video_path}"
        )
        if self.model is None:
            return self._ask_via_gemini(video_path, question)

        default_content = """Answer questions about videos by:
            1. Examining the sampled frames carefully
            2. Reasoning across frames for temporal context
            3. Transcribing on-screen text where relevant
            4. Logical deduction from visual evidence"""

        system_msg = BaseMessage.make_assistant_message(
            role_name="Video QA Specialist",
            content=sys_prompt if sys_prompt else default_content,
        )

        return self._analyze_video(
            video_path=video_path,
            prompt=question,
            system_message=system_msg,
        )

    def _load_video_bytes(self, video_path: str) -> bytes:
        r"""Loads a video from either local path or URL.

        Args:
            video_path (str): Local path or URL to video.

        Returns:
            bytes: Raw video file contents.

        Raises:
            ValueError: For invalid paths or unreadable files.
            requests.exceptions.RequestException: For URL fetch failures.
        """
        parsed = urlparse(video_path)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        }

        if parsed.scheme in ("http", "https"):
            logger.debug(f"Fetching video from URL: {video_path}")
            try:
                response = requests.get(video_path, timeout=30, headers=headers)
                response.raise_for_status()
                return response.content
            except requests.exceptions.RequestException as e:
                logger.error(f"URL fetch failed: {e}")
                raise
        else:
            logger.debug(f"Loading local video: {video_path}")
            try:
                with open(video_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Video loading failed: {e}")
                raise ValueError(f"Invalid video file: {e}")

    def _analyze_video(
        self,
        video_path: str,
        prompt: str,
        system_message: BaseMessage,
    ) -> str:
        r"""Core analysis method handling video loading and processing.

        Args:
            video_path (str): Video location.
            prompt (str): Analysis query/instructions.
            system_message (BaseMessage): Custom system prompt for the
                analysis.

        Returns:
            str: Analysis result or error message.
        """
        try:
            video_bytes = self._load_video_bytes(video_path)
            logger.info(f"Analyzing video: {video_path}")

            agent = ChatAgent(
                system_message=system_message,
                model=self.model,
            )

            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                video_bytes=video_bytes,
            )

            response = agent.step(user_msg)
            agent.reset()
            return response.msgs[0].content

        except (ValueError, requests.exceptions.RequestException) as e:
            logger.error(f"Video handling error: {e}")
            return f"Video error: {e!s}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Analysis failed: {e!s}"

    def _ask_via_gemini(self, video_path: str, question: str) -> str:
        r"""Fallback path: Google Gemini video understanding (paid)."""
        from camel.utils.replay_capture import record_unreplayable_model_call

        record_unreplayable_model_call(
            name="video_understanding",
            provider="google-genai",
            details={"model": "models/gemini-2.0-flash"},
        )
        os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

        import pathlib
        from google import genai
        from google.genai import types

        client = genai.Client()

        model = 'models/gemini-2.0-flash'

        response = client.models.generate_content(
            model=model,
            contents=types.Content(
                parts=[
                    types.Part(text=question),
                    types.Part(file_data=types.FileData(file_uri=video_path))
                ]
            )
        )

        logger.debug(f"Video analysis response from gemini: {response.text}")
        return response.text
    
        
    def get_tools(self) -> List[FunctionTool]:
        """
        Get the tools in the toolkit.
        """
        return [FunctionTool(self.ask_question_about_video)]
    
