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
from camel.utils.constants import Constants
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

    def resolve_video_path(self, video_path: str) -> str:
        r"""Resolve a local video path, downloading remote pages with yt-dlp.

        Args:
            video_path (str): Local path or URL to video.

        Returns:
            str: A local path containing the video payload.

        Raises:
            ValueError: If a local path does not exist.
        """
        parsed = urlparse(video_path)
        if parsed.scheme in ("http", "https"):
            return self.video_downloader_toolkit.download_video(video_path)
        path = Path(video_path).resolve()
        if not path.is_file():
            raise ValueError(f"Invalid video file: {video_path}")
        return str(path)

    def extract_video_frames(
        self,
        video_path: str,
        *,
        frame_interval: int = Constants.VIDEO_IMAGE_EXTRACTION_INTERVAL,
        image_size: int = Constants.VIDEO_DEFAULT_IMAGE_SIZE,
    ) -> List[Image.Image]:
        r"""Extract the same sampled, resized frames used by BaseMessage."""

        import imageio.v3 as iio

        if frame_interval <= 0:
            raise ValueError("frame_interval must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        frames: List[Image.Image] = []
        for frame_count, frame in enumerate(
            iio.imiter(video_path, plugin=Constants.VIDEO_DEFAULT_PLUG_PYAV),
            start=1,
        ):
            if frame_count % frame_interval != 0:
                continue
            frame_image = Image.fromarray(frame)
            width, height = frame_image.size
            if height <= 0:
                raise ValueError("video frame has invalid height")
            new_height = int(image_size / (width / height))
            resized_frame = frame_image.resize((image_size, new_height))
            # BaseMessage's image-list encoder needs an explicit format in
            # order to serialize PIL images into OpenAI image_url parts.
            resized_frame.format = "JPEG"
            frames.append(resized_frame)
        if not frames:
            raise ValueError("No frames were extracted from the video")
        return frames

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
            from camel.utils.replay_capture import run_tool_primitive

            local_video_path = run_tool_primitive(
                name="video_download",
                arguments={"video_path": video_path},
                function=lambda: self.resolve_video_path(video_path),
                toolkit="VideoAnalysisPrimitive",
            )
            frames = run_tool_primitive(
                name="video_extract_frames",
                arguments={
                    "frame_interval": Constants.VIDEO_IMAGE_EXTRACTION_INTERVAL,
                    "image_size": Constants.VIDEO_DEFAULT_IMAGE_SIZE,
                },
                function=lambda: self.extract_video_frames(local_video_path),
                toolkit="VideoAnalysisPrimitive",
                record_result=lambda sampled_frames: {
                    "frame_count": len(sampled_frames),
                    "frames": [
                        {
                            "width": frame.width,
                            "height": frame.height,
                            "mode": frame.mode,
                        }
                        for frame in sampled_frames
                    ],
                },
            )
            logger.info(f"Analyzing video: {video_path}")

            agent = ChatAgent(
                system_message=system_message,
                model=self.model,
            )

            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=frames,
                image_detail="low",
            )
            # Qwen VL models can over-weight the final frame when a long image
            # sequence follows the question. Keep normal image-message behavior
            # unchanged, but place the video question after its sampled frames.
            user_msg.media_before_text = True

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
    
