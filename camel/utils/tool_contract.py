"""Replay-facing execution contracts for Owl capabilities.

The public Agent API may expose orchestration helpers and model-backed
capabilities as ``FunctionTool`` objects.  That does not make them replay
tools: only explicit ``deterministic_tool`` and ``external_tool`` primitives
are eligible for direct execution by the replay worker.
"""

from __future__ import annotations

from typing import Literal

ToolExecutionKind = Literal[
    "deterministic_tool",
    "external_tool",
    "orchestration",
    "model",
]


TOOL_EXECUTION_CONTRACTS: dict[str, ToolExecutionKind] = {
    # Agent-facing orchestration boundaries. Their nested model requests and
    # model-free primitives are captured independently.
    "browse_url": "orchestration",
    "extract_document_content": "orchestration",
    "web_search": "orchestration",
    "ask_question_about_audio": "orchestration",
    # These names are model inference capabilities, even though CAMEL exposes
    # them through FunctionTool for normal Agent operation.
    "image_to_text": "model",
    "ask_question_about_image": "model",
    "ask_question_about_video": "model",
    # Replayable, model-free primitives.
    "browser_open": "deterministic_tool",
    "browser_observe": "deterministic_tool",
    "browser_action": "deterministic_tool",
    "browser_close": "deterministic_tool",
    "document_extract_raw": "external_tool",
    "document_select_chunks": "deterministic_tool",
    "audio_transcription": "external_tool",
    "video_download": "external_tool",
    "video_extract_frames": "deterministic_tool",
    "execute_code": "deterministic_tool",
    "extract_excel_content": "deterministic_tool",
    # These contain no model, but their remote observations can change.
    "search_duckduckgo": "external_tool",
    "search_google": "external_tool",
    "search_wiki": "external_tool",
    "search_wiki_revisions": "external_tool",
    "search_archived_webpage": "external_tool",
}


def tool_execution_kind(name: str) -> ToolExecutionKind:
    """Return the declared kind, defaulting unknown tools to fail-safe tool mode."""

    return TOOL_EXECUTION_CONTRACTS.get(name, "deterministic_tool")


def capture_outer_tool(name: str) -> bool:
    """Whether the Agent-facing call itself belongs in replay inventory."""

    return tool_execution_kind(name) in {"deterministic_tool", "external_tool"}


def allows_nested_model_calls(name: str) -> bool:
    """Whether model calls are expected below this Agent-facing boundary."""

    return tool_execution_kind(name) in {"orchestration", "model"}
