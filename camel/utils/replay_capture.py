"""Opt-in exact capture taps for agent-replay Owl recordings."""

from __future__ import annotations

import contextvars
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from camel.utils.tool_contract import capture_outer_tool

T = TypeVar("T")

_actor = contextvars.ContextVar("agent_replay_owl_actor", default="unknown")
_capture_tool = contextvars.ContextVar("agent_replay_owl_capture_tool", default=True)
_tool_name = contextvars.ContextVar("agent_replay_owl_tool_name", default=None)
_allow_nested_models = contextvars.ContextVar(
    "agent_replay_owl_allow_nested_models", default=False
)


def _actor_id(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return normalized or "unknown"


def set_tool_context(
    actor: str,
    capture: bool = True,
    *,
    tool_name: str | None = None,
    allow_nested_models: bool = False,
):
    return (
        _actor.set(actor),
        _capture_tool.set(capture),
        _tool_name.set(tool_name),
        _allow_nested_models.set(allow_nested_models),
    )


def reset_tool_context(tokens) -> None:
    actor_token, capture_token, tool_name_token, allow_models_token = tokens
    _actor.reset(actor_token)
    _capture_tool.reset(capture_token)
    _tool_name.reset(tool_name_token)
    _allow_nested_models.reset(allow_models_token)


def _capture_dir() -> Path | None:
    value = os.environ.get("AGENT_REPLAY_OWL_CAPTURE_DIR")
    return Path(value).resolve() if value else None


def append_record(record: dict[str, Any]) -> None:
    root = _capture_dir()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "owl.capture-event/v1",
        "task_id": os.environ.get("AGENT_REPLAY_OWL_TASK_ID"),
        "pid": os.getpid(),
        "thread_id": threading.get_native_id(),
        **record,
    }
    payload = (json.dumps(record, default=str, separators=(",", ":")) + "\n").encode()
    fd = os.open(root / f"events.{os.getpid()}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def record_tool(
    *,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    error: str | None,
    started_at_ns: int,
    ended_at_ns: int,
) -> None:
    if _capture_dir() is None or not _capture_tool.get():
        return
    owner = getattr(func, "__self__", None)
    name = getattr(func, "__name__", repr(func))
    # Orchestration and model capabilities remain valid FunctionTools for live
    # Agent execution, but their outer calls are not replay tools. Their nested
    # LLM requests and explicit model-free primitives are captured separately.
    if not capture_outer_tool(name):
        return
    append_record(
        {
            "kind": "function_tool.call",
            "capture_id": uuid.uuid4().hex,
            "actor_id": _actor_id(_actor.get()),
            "toolkit": type(owner).__name__ if owner is not None else "function",
            "invocation": {
                "name": name,
                "arguments": kwargs if not args else {"args": list(args), **kwargs},
            },
            "status": "error" if error else "success",
            "output": result,
            "error": error,
            "started_at_ns": started_at_ns,
            "ended_at_ns": ended_at_ns,
        }
    )


def run_tool_primitive(
    *,
    name: str,
    arguments: dict[str, Any],
    function: Callable[[], T],
    toolkit: str,
) -> T:
    """Execute and capture one synchronous model-free replay primitive."""

    if _capture_dir() is None or not _capture_tool.get():
        return function()
    started_at_ns = time.time_ns()
    tool_token = _tool_name.set(name)
    allow_token = _allow_nested_models.set(False)
    result: Any = None
    error: str | None = None
    try:
        result = function()
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _allow_nested_models.reset(allow_token)
        _tool_name.reset(tool_token)
        append_record(
            {
                "kind": "function_tool.call",
                "capture_id": uuid.uuid4().hex,
                "actor_id": _actor_id(_actor.get()),
                "toolkit": toolkit,
                "invocation": {"name": name, "arguments": arguments},
                "status": "error" if error else "success",
                "output": result,
                "error": error,
                "started_at_ns": started_at_ns,
                "ended_at_ns": time.time_ns(),
            }
        )


def record_unreplayable_model_call(
    *, name: str, provider: str, details: dict[str, Any] | None = None
) -> None:
    """Fail-closed marker for model inference lacking an HTTP replay adapter."""

    if _capture_dir() is None or not _capture_tool.get():
        return
    now = time.time_ns()
    append_record(
        {
            "kind": "model.call.unreplayable",
            "capture_id": uuid.uuid4().hex,
            "actor_id": _actor_id(_actor.get()),
            "model_call": {"name": name, "provider": provider, **(details or {})},
            "started_at_ns": now,
            "ended_at_ns": now,
        }
    )


def record_browser_primitive(
    *,
    name: str,
    arguments: dict[str, Any],
    result: Any,
    error: str | None,
    started_at_ns: int,
    ended_at_ns: int,
    replayable: bool = True,
) -> None:
    """Record one model-free browser operation inside ``browse_url``.

    Browser planning remains an ordinary captured LLM request.  Only explicit
    operations whose complete input is present in ``arguments`` are admitted
    as replay tools.  A model-backed browser action is emitted as an explicit
    unsupported record so the recording builder fails closed.
    """

    if _capture_dir() is None or not _capture_tool.get():
        return
    append_record(
        {
            "kind": (
                "function_tool.call"
                if replayable
                else "browser.primitive.unreplayable"
            ),
            "capture_id": uuid.uuid4().hex,
            "actor_id": _actor_id(_actor.get()),
            "toolkit": "AsyncBrowserPrimitive",
            "invocation": {"name": name, "arguments": arguments},
            "status": "error" if error else "success",
            "output": result,
            "error": error,
            "started_at_ns": started_at_ns,
            "ended_at_ns": ended_at_ns,
        }
    )


class OpenAIReplayCapture:
    def __init__(self, owner: Any):
        self.owner = owner

    def _request(self, request: Any) -> None:
        if _capture_dir() is None or not request.url.path.endswith("/chat/completions"):
            return
        capture_id = uuid.uuid4().hex
        started_at_ns = time.time_ns()
        active_tool = _tool_name.get()
        if active_tool and not _allow_nested_models.get():
            append_record(
                {
                    "kind": "tool.model_call.violation",
                    "capture_id": f"violation-{capture_id}",
                    "actor_id": _actor_id(_actor.get()),
                    "tool_name": active_tool,
                    "started_at_ns": started_at_ns,
                    "ended_at_ns": started_at_ns,
                }
            )
        body = bytes(request.content)
        request.extensions["agent_replay_capture_id"] = capture_id
        request.extensions["agent_replay_started_at_ns"] = started_at_ns
        request.extensions["agent_replay_body"] = body
        try:
            request_body = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_body = None
        append_record(
            {
                "kind": "llm.request.start",
                "capture_id": capture_id,
                "actor_id": _actor_id(
                    getattr(self.owner, "_agent_replay_role", "unknown")
                ),
                "target_id": _actor_id(
                    getattr(self.owner, "_agent_replay_role", "unknown")
                ),
                "endpoint_api": "chat.completions",
                "request_body": request_body,
                "started_at_ns": started_at_ns,
            }
        )

    def _response(self, response: Any) -> None:
        capture_id = response.request.extensions.get("agent_replay_capture_id")
        if capture_id is None:
            return
        response.read()
        self._finish(response, capture_id)

    async def _arequest(self, request: Any) -> None:
        self._request(request)

    async def _aresponse(self, response: Any) -> None:
        capture_id = response.request.extensions.get("agent_replay_capture_id")
        if capture_id is None:
            return
        await response.aread()
        self._finish(response, capture_id)

    def _finish(self, response: Any, capture_id: str) -> None:
        try:
            request_body = json.loads(response.request.extensions["agent_replay_body"])
        except (KeyError, json.JSONDecodeError):
            request_body = None
        try:
            response_body = json.loads(response.content)
        except json.JSONDecodeError:
            response_body = None
        append_record(
            {
                "kind": "llm.request",
                "capture_id": capture_id,
                "actor_id": _actor_id(
                    getattr(self.owner, "_agent_replay_role", "unknown")
                ),
                "target_id": _actor_id(
                    getattr(self.owner, "_agent_replay_role", "unknown")
                ),
                "endpoint_api": "chat.completions",
                "request_body": request_body,
                "response_usage": response_body.get("usage")
                if isinstance(response_body, dict)
                else None,
                "response_id": response_body.get("id")
                if isinstance(response_body, dict)
                else None,
                "status_code": response.status_code,
                "started_at_ns": response.request.extensions["agent_replay_started_at_ns"],
                "ended_at_ns": time.time_ns(),
            }
        )

    @property
    def sync_hooks(self) -> dict[str, list[Any]]:
        return {"request": [self._request], "response": [self._response]}

    @property
    def async_hooks(self) -> dict[str, list[Any]]:
        return {"request": [self._arequest], "response": [self._aresponse]}
