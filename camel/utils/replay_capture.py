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
from typing import Any

_actor = contextvars.ContextVar("agent_replay_owl_actor", default="unknown")
_capture_tool = contextvars.ContextVar("agent_replay_owl_capture_tool", default=True)


def _actor_id(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return normalized or "unknown"


def set_tool_context(actor: str, capture: bool = True):
    return _actor.set(actor), _capture_tool.set(capture)


def reset_tool_context(tokens) -> None:
    actor_token, capture_token = tokens
    _actor.reset(actor_token)
    _capture_tool.reset(capture_token)


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
    append_record(
        {
            "kind": "function_tool.call",
            "capture_id": uuid.uuid4().hex,
            "actor_id": _actor_id(_actor.get()),
            "toolkit": type(owner).__name__ if owner is not None else "function",
            "invocation": {
                "name": getattr(func, "__name__", repr(func)),
                "arguments": kwargs if not args else {"args": list(args), **kwargs},
            },
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
