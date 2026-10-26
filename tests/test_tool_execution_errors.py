"""Escaping native tool errors must not become developer logs or Result payloads."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from .helpers import make_model_route

_PRIVATE_MARKERS = ("PRIVATE-EXECUTION-ERROR", "PRIVATE-ARGUMENT", "PRIVATE-TOKEN")


def _install_failing_tool_graph(
    monkeypatch: pytest.MonkeyPatch, *, subagent: bool = False, async_tool: bool = False
) -> Any:
    import deepagents
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import StructuredTool

    class MainModel(FakeMessagesListChatModel):
        def bind_tools(self, _tools: Any, **_kwargs: Any) -> Any:
            return self

    def fail(query: str) -> str:
        raise ValueError(
            f"PRIVATE-EXECUTION-ERROR: {query}; https://mcp.example/?token=PRIVATE-TOKEN"
        )

    async def afail(query: str) -> str:
        return fail(query)

    failing_tool = StructuredTool.from_function(
        name="failing_lookup",
        description="Read external findings.",
        func=None if async_tool else fail,
        coroutine=afail if async_tool else None,
    )
    responses = []
    if subagent:
        responses.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "task-id",
                        "name": "task",
                        "args": {
                            "description": "Look up findings.",
                            "subagent_type": "general-purpose",
                        },
                    }
                ],
            )
        )
    responses.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "type": "tool_call",
                    "id": "failed-id",
                    "name": "failing_lookup",
                    "args": {"query": "PRIVATE-ARGUMENT"},
                }
            ],
        )
    )
    model = MainModel(responses=responses)
    classifier = FakeMessagesListChatModel(
        responses=[AIMessage(content=('{"injectionDetected":false,"category":"none"}'))]
    )
    create_agent = deepagents.create_deep_agent

    def with_tool(**kwargs: Any) -> Any:
        kwargs["tools"] = [failing_tool]
        return create_agent(**kwargs)

    monkeypatch.setattr(deepagents, "create_deep_agent", with_tool)
    return lambda parameters, _reasoning=None: classifier if parameters.max_output_tokens else model


@pytest.mark.asyncio
@pytest.mark.parametrize("subagent", [False, True], ids=["main", "subagent"])
@pytest.mark.parametrize("async_tool", [False, True], ids=["sync-tool", "async-tool"])
@pytest.mark.parametrize("inspection_enabled", [False, True])
async def test_escaping_tool_errors_have_payload_free_operational_failures(
    subagent: bool,
    async_tool: bool,
    inspection_enabled: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    span_exporter: Any,
) -> None:
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.run_agent import run_agent_query

    caplog.set_level("INFO")
    monkeypatch.setenv("HOME", str(tmp_path))
    create_model = _install_failing_tool_graph(
        monkeypatch, subagent=subagent, async_tool=async_tool
    )
    result = await run_agent_query(
        DeepAgentsProvider(make_model_route(create_model)),
        prompt="Look up findings.",
        system_prompt="Report findings.",
        output_schema=None,
        context=None,
        skills_dir=str(tmp_path),
        model="offline-model",
        max_turns=20,
        timeout_seconds=30,
        tool_output_inspection_enabled=inspection_enabled,
    )
    assert result.output == {"success": False, "summary": "Agent error: tool execution failed."}
    for marker in _PRIVATE_MARKERS:
        assert marker not in caplog.text
        assert marker not in repr([record.__dict__ for record in caplog.records])
        assert marker not in json.dumps(result.output)
    failure_records = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(failure_records) == 1
    assert failure_records[0].exc_info is None
    assert "failing_lookup" in failure_records[0].getMessage()
    assert "ValueError" in failure_records[0].getMessage()
    spans = span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "invoke_agent")
    assert root.status.status_code.name == "ERROR"
    assert root.attributes["error.type"] == "ToolExecutionError"
    assert "gen_ai.output.messages" not in root.attributes
    native = next(s for s in spans if s.name == "execute_tool failing_lookup")
    assert native.attributes["error.type"] == "ValueError"
    assert "gen_ai.tool.call.result" not in native.attributes
    assert not native.events
    assert not any(s.name == "tool_result.inspection" for s in spans)


def test_escaping_tool_error_publishes_safe_operational_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from lightspeed_agentic import batch
    from lightspeed_agentic.providers.models import anthropic
    from lightspeed_agentic.publish_results import publish as publisher

    caplog.set_level("INFO")
    create_model = _install_failing_tool_graph(monkeypatch, async_tool=True)
    monkeypatch.setattr(anthropic, "create_direct_model", create_model)
    template = {
        "apiVersion": "agentic.openshift.io/v1alpha1",
        "kind": "AnalysisResult",
        "metadata": {"name": "operational-failure", "namespace": "default"},
        "spec": {"agenticRunName": "run"},
    }
    monkeypatch.setattr(
        batch,
        "read_batch_inputs",
        lambda: batch.BatchInput(
            query="Look up findings.",
            output_schema={"type": "object", "properties": {}},
            context={},
            result_template=template,
        ),
    )
    termination_path = tmp_path / "termination.log"
    monkeypatch.setattr(
        batch,
        "write_termination_log",
        partial(publisher.write_termination_log, path=str(termination_path)),
    )
    published: list[dict[str, Any]] = []

    class RecordingApi:
        def create_namespaced_custom_object(self, **kwargs: Any) -> dict[str, Any]:
            published.append(kwargs["body"])
            return {"metadata": {"resourceVersion": "1"}}

        def replace_namespaced_custom_object_status(self, **kwargs: Any) -> None:
            published.append(kwargs["body"])

    monkeypatch.setattr(publisher, "_default_api", RecordingApi)
    with patch.dict(
        os.environ,
        {
            "HOME": str(tmp_path),
            "LIGHTSPEED_PROVIDER": "anthropic",
            "LIGHTSPEED_MODEL": "offline-model",
            "ANTHROPIC_API_KEY": "offline-key",
            "LIGHTSPEED_SKILLS_DIR": str(tmp_path),
            "LIGHTSPEED_AGENT_TIMEOUT_SECONDS": "30",
            "LIGHTSPEED_AGENT_MAX_TURNS": "20",
            "LIGHTSPEED_AUDIT_ENABLED": "false",
            "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true",
        },
        clear=True,
    ):
        assert batch.main() is None  # Operational failure remains the normal exit-0 path.
    assert not termination_path.exists()
    assert len(published) == 2
    status = published[-1]["status"]
    assert status["failureReason"] == "Agent error: tool execution failed."
    completed = next(c for c in status["conditions"] if c["type"] == "Completed")
    assert completed["reason"] == "Failed"
    for marker in _PRIVATE_MARKERS:
        assert marker not in json.dumps(published)
        assert marker not in caplog.text
        assert marker not in repr([record.__dict__ for record in caplog.records])
