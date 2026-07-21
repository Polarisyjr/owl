from __future__ import annotations

import json
import os

from camel.toolkits.function_tool import _step3_enabled
from camel.utils.replay_capture import record_browser_primitive, run_tool_primitive


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_step3_replaces_orchestration_boundary_with_primitives(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("STEP3_TOOL_LOG", str(tmp_path / "tools.jsonl"))

    def browse_url():
        pass

    def search_wiki():
        pass

    assert _step3_enabled(browse_url) is False
    assert _step3_enabled(search_wiki) is True


def test_browser_primitive_writes_step3_without_replay_capture(
    tmp_path, monkeypatch
):
    output = tmp_path / "tool_events_raw.jsonl"
    monkeypatch.setenv("STEP3_TOOL_LOG", str(output))
    monkeypatch.delenv("AGENT_REPLAY_OWL_CAPTURE_DIR", raising=False)
    monkeypatch.setenv("OWL_T2T_TASK", "task-42")

    record_browser_primitive(
        name="browser_action",
        arguments={"action_code": "scroll_down()"},
        result={"success": True},
        error=None,
        started_at_ns=1_500_000_000,
        ended_at_ns=2_750_000_000,
    )

    assert _events(output) == [
        {
            "tool": "browser_action",
            "chain": f"task-42/AsyncBrowserPrimitive",
            "task_id": "task-42",
            "pid": os.getpid(),
            "call_id": f"{os.getpid()}-1500000000",
            "phase": "end",
            "ts_start": 1.5,
            "ts_end": 2.75,
            "success": True,
        }
    ]


def test_generic_primitive_writes_step3_without_replay_capture(
    tmp_path, monkeypatch
):
    output = tmp_path / "tool_events_raw.jsonl"
    monkeypatch.setenv("STEP3_TOOL_LOG", str(output))
    monkeypatch.delenv("AGENT_REPLAY_OWL_CAPTURE_DIR", raising=False)

    assert run_tool_primitive(
        name="document_extract_raw",
        arguments={"path": "paper.pdf"},
        function=lambda: "content",
        toolkit="DocumentProcessingPrimitive",
    ) == "content"

    event = _events(output)[0]
    assert event["tool"] == "document_extract_raw"
    assert event["chain"].endswith("/DocumentProcessingPrimitive")
    assert event["ts_end"] >= event["ts_start"]
    assert event["success"] is True
