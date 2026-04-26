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

import base64
import os
from typing import List, Optional
from urllib.parse import urlparse


import openai
import requests
from pydub.utils import mediainfo

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.agents import ChatAgent
from camel.models import BaseModelBackend

import logging
logger = logging.getLogger(__name__)


class AudioAnalysisToolkit(BaseToolkit):
    r"""A class representing a toolkit for audio operations.

    This class provides methods for processing and understanding audio data.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        audio_reasoning_model: Optional[BaseModelBackend] = None,
        whisper_model_size: str = "large-v3",
        whisper_device_index: Optional[int] = None,
    ):
        self.cache_dir = cache_dir or "tmp/"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.audio_reasoning_model = audio_reasoning_model
        self._whisper_model_size = whisper_model_size
        self._whisper_device_index = whisper_device_index  # None → faster-whisper default (cuda:0)
        self._whisper_model = None  # lazy: load only when transcribing
        self._openai_client = None  # lazy: only used by the gpt-4o-audio fallback branch

    @property
    def client(self):
        """Lazy OpenAI client. Only required by the (paid) gpt-4o-audio
        fallback branch when no audio_reasoning_model is provided."""
        if self._openai_client is None:
            self._openai_client = openai.OpenAI()
        return self._openai_client

    def _get_whisper(self):
        """Lazy-load a faster-whisper model on first use."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise ImportError(
                    "faster-whisper is required for local audio transcription. "
                    "Install with: pip install faster-whisper"
                ) from e
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            kwargs = dict(device=device, compute_type=compute_type)
            if self._whisper_device_index is not None:
                kwargs["device_index"] = self._whisper_device_index
            logger.info(
                f"Loading faster-whisper {self._whisper_model_size} on {device} "
                f"(compute_type={compute_type}, device_index={self._whisper_device_index})..."
            )
            self._whisper_model = WhisperModel(self._whisper_model_size, **kwargs)
        return self._whisper_model

    def _ensure_local_path(self, audio_path: str) -> str:
        """Download URL audio to a temp file under cache_dir; return local path."""
        parsed = urlparse(audio_path)
        if not all([parsed.scheme, parsed.netloc]):
            return audio_path
        import tempfile
        suffix = os.path.splitext(parsed.path)[1] or ".audio"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=self.cache_dir)
        os.close(fd)
        res = requests.get(audio_path)
        res.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(res.content)
        return tmp_path

    def _transcribe_local(self, audio_path: str) -> str:
        """Transcribe audio with faster-whisper (free, local)."""
        model = self._get_whisper()
        segments, _info = model.transcribe(audio_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments)

    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        info = mediainfo(file_path)
        return float(info['duration'])


    def ask_question_about_audio(self, audio_path: str, question: str) -> str:
        r"""Ask any question about the audio and get the answer using
            multimodal model.

        Args:
            audio_path (str): The path to the audio file.
            question (str): The question to ask about the audio.

        Returns:
            str: The answer to the question.
        """

        logger.debug(
            f"Calling ask_question_about_audio method for audio file \
            `{audio_path}` and question `{question}`."
        )

        # Normalize URL → local file once, then reuse for transcription / encoding / duration
        local_audio_path = self._ensure_local_path(audio_path)
        duration = self.get_audio_duration(local_audio_path)

        if self.audio_reasoning_model:
            transcript = self._transcribe_local(local_audio_path)

            reasoning_prompt = f"""
            <speech_transcription_result>{transcript}</speech_transcription_result>

            Please answer the following question based on the speech transcription result above:
            <question>{question}</question>
            """

            audio_reasoning_agent = ChatAgent(
                "You are a helpful assistant that can answer questions about the given speech transcription.",
                model=self.audio_reasoning_model,
            )

            reasoning_result = audio_reasoning_agent.step(reasoning_prompt)
            response: str = str(reasoning_result.msg.content)
            response += f"\n\nAudio duration: {duration} seconds"

            logger.debug(f"Response: {response}")
            return response


        else:
            # ── Paid fallback: gpt-4o-mini-audio-preview ──
            with open(local_audio_path, "rb") as f:
                audio_data = f.read()
            encoded_string = base64.b64encode(audio_data).decode("utf-8")
            file_format = os.path.splitext(local_audio_path)[1][1:]

            text_prompt = f"""Answer the following question based on the given \
            audio information:\n\n{question}"""

            completion = self.client.chat.completions.create(
                # model="gpt-4o-audio-preview",
                model = "gpt-4o-mini-audio-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in \
                        audio analysis.",
                    },
                    {  # type: ignore[list-item, misc]
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_string,
                                    "format": file_format,
                                },
                            },
                        ],
                    },
                ],
            )  # type: ignore[misc]

            response: str = str(completion.choices[0].message.content)
            response += f"\n\nAudio duration: {duration} seconds"

            logger.debug(f"Response: {response}")
            return response
        

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
            in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the
                functions in the toolkit.
        """
        return [FunctionTool(self.ask_question_about_audio)]
