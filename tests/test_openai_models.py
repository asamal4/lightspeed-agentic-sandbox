"""Offline tests for the native OpenAI LangChain model constructor."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from lightspeed_agentic import tls
from lightspeed_agentic.providers.model_router import ModelParameters


@pytest.fixture
def http_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[httpx.Client, httpx.AsyncClient]]:
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
    monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
    monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
    try:
        yield sync_client, async_client
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


@pytest.fixture
def captured_openai_requests(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient]]:
    requests: list[httpx.Request] = []
    response_text = json.dumps(getattr(request, "param", {"status": "ok"}))

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_capture-test",
                "object": "response",
                "created_at": 1_700_000_000,
                "model": "gpt-4.1-mini",
                "output": [
                    {
                        "id": "msg_capture-test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": response_text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        )

    transport = httpx.MockTransport(capture)
    sync_client = httpx.Client(transport=transport)
    async_client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
    monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
    try:
        yield requests, sync_client, async_client
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


def test_openai_model_uses_typed_responses_settings_credentials_and_tls(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_openai import ChatOpenAI

    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-win")
    parameters = ModelParameters(
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key="resolved-openai-key",
    )

    model = create_native_openai_model(
        parameters,
        {"effort": "high", "summary": "auto", "verbosity": "low"},
    )

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4.1-mini"
    assert model.use_responses_api is True
    assert model.output_version == "responses/v1"
    assert model.reasoning == {"effort": "high", "summary": "auto"}
    assert model.verbosity == "low"
    assert model.root_client.api_key == "resolved-openai-key"
    assert str(model.root_client.base_url) == "https://api.openai.com/v1/"
    assert model.root_client._client is http_clients[0]
    assert model.root_async_client.api_key == "resolved-openai-key"
    assert str(model.root_async_client.base_url) == "https://api.openai.com/v1/"
    assert model.root_async_client._client is http_clients[1]
    assert "resolved-openai-key" not in repr(parameters)


def test_openai_model_omits_reasoning_settings_when_unconfigured(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient-endpoint.invalid/v1")
    model = create_native_openai_model(
        ModelParameters(model="gpt-4.1-mini", api_key="resolved-openai-key")
    )

    assert model.reasoning is None
    assert model.verbosity is None
    assert model.use_responses_api is True
    assert str(model.root_client.base_url) == "https://api.openai.com/v1/"
    assert model.root_client._client is http_clients[0]


@pytest.mark.asyncio
async def test_openai_model_sends_responses_request_with_configured_settings(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-win")
    model = create_native_openai_model(
        ModelParameters(
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key="resolved-openai-key",
        ),
        {"effort": "high", "summary": "auto", "verbosity": "low"},
    )

    await model.ainvoke("capture this request")

    requests, _sync_client, async_client = captured_openai_requests
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer resolved-openai-key"
    assert async_client is model.root_async_client._client
    body = json.loads(request.content)
    assert body["model"] == "gpt-4.1-mini"
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["text"]["verbosity"] == "low"
    assert "capture this request" in json.dumps(body["input"])


@pytest.mark.asyncio
async def test_openai_model_omits_reasoning_from_request_when_unconfigured(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
) -> None:
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    model = create_native_openai_model(
        ModelParameters(model="gpt-4.1-mini", api_key="resolved-openai-key")
    )
    await model.ainvoke("use model defaults")

    requests, _sync_client, _async_client = captured_openai_requests
    body = json.loads(requests[0].content)
    assert "reasoning" not in body
    assert "verbosity" not in body.get("text", {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "captured_openai_requests",
    [
        {
            "status": "ok",
            "details": {"count": 1},
            "variant": "example",
            "reference": {"enabled": True},
        }
    ],
    indirect=True,
)
async def test_native_openai_structured_output_sends_strict_schema_in_responses_request(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
) -> None:
    from opentelemetry.context import Context

    from lightspeed_agentic.providers.deepagents import (
        _create_generation_callback,
        _shape_structured_output,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    route = ResolvedModelRoute(
        route="openai",
        model="gpt-4.1-mini",
        create_model=create_native_openai_model,
        model_parameters=ModelParameters(
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key="resolved-openai-key",
        ),
        telemetry_provider="openai",
        structured_output_method="json_schema",
        structured_output_schema_mode="openai_strict_json_schema",
    )
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Overall status"},
            "details": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            "variant": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
                "description": "A string or integer variant",
            },
            "reference": {"$ref": "#/$defs/Reference"},
        },
        "required": ["status"],
        "$defs": {
            "Reference": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
            }
        },
    }

    result, _input_tokens, _output_tokens = await _shape_structured_output(
        route,
        schema,
        "Return a status.",
        "How did it go?",
        "It went well.",
        _create_generation_callback(route, Context(), main_name=None),
    )

    requests, _sync_client, _async_client = captured_openai_requests
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["model"] == "gpt-4.1-mini"
    assert "reasoning" not in body
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"]["additionalProperties"] is False
    strict_schema = body["text"]["format"]["schema"]
    jsonschema.validate(result, strict_schema)
    assert strict_schema["required"] == ["status", "details", "variant", "reference"]
    assert strict_schema["properties"]["details"]["required"] == ["count"]
    assert strict_schema["properties"]["details"]["additionalProperties"] is False
    assert strict_schema["properties"]["variant"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
    ]
    assert "oneOf" not in strict_schema["properties"]["variant"]
    assert strict_schema["properties"]["reference"] == {"$ref": "#/$defs/Reference"}
    assert strict_schema["$defs"]["Reference"]["required"] == ["enabled"]
    assert strict_schema["properties"]["status"]["description"] == "Overall status"
    assert result == {
        "status": "ok",
        "details": {"count": 1},
        "variant": "example",
        "reference": {"enabled": True},
    }
    assert "oneOf" in schema["properties"]["variant"]
    assert "anyOf" not in schema["properties"]["variant"]


def _openai_response(
    response_id: str,
    item: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1_700_000_000,
        "model": "gpt-4.1-mini",
        "output": [item],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _openai_response_stream(
    response_id: str,
    item: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
) -> httpx.Response:
    """Encode scripted native Responses events, including real streaming tool arguments."""
    response = _openai_response(
        response_id, item, input_tokens=input_tokens, output_tokens=output_tokens
    )
    events = [
        {
            "type": "response.created",
            "response": {**response, "output": [], "status": "in_progress", "usage": None},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "arguments": "", "status": "in_progress"}
            if item["type"] == "function_call"
            else {**item, "content": [], "status": "in_progress"},
        },
    ]
    if item["type"] == "function_call":
        events.append(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": 0,
                "delta": item["arguments"],
            }
        )
    else:
        events.extend(
            [
                {
                    "type": "response.content_part.added",
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "part": {**item["content"][0], "text": ""},
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": item["content"][0]["text"],
                    "logprobs": [],
                },
            ]
        )
    events.extend(
        [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
    )
    content = "".join(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index})}\n\n"
        for index, event in enumerate(events)
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


@pytest.mark.asyncio
async def test_native_openai_deepagent_reads_file_and_shapes_structured_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    span_exporter: Any,
) -> None:
    from lightspeed_agentic.config import ResolvedModelFamily
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from lightspeed_agentic.types import (
        ProviderQueryOptions,
        ResultEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    sentinel_file = tmp_path / "sentinel.txt"
    sentinel_file.write_text("native-openai-file-sentinel\n")
    schema = {
        "type": "object",
        "properties": {"sentinel": {"type": "string", "description": "Contents of the file"}},
        "required": ["sentinel"],
    }
    requests: list[dict[str, Any]] = []

    def message(text: str) -> dict[str, Any]:
        return {
            "id": "msg_native-integration",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer resolved-openai-key"
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "gpt-4.1-mini"
        if len(requests) == 1:
            assert body["stream"] is True
            assert "read_file" in {tool["name"] for tool in body["tools"]}
            assert "format" not in body.get("text", {})
            return _openai_response_stream(
                "resp_read-file",
                {
                    "id": "fc_read-file",
                    "type": "function_call",
                    "name": "read_file",
                    "call_id": "call_read-file",
                    "arguments": json.dumps({"file_path": str(sentinel_file)}),
                    "status": "completed",
                },
                input_tokens=10,
                output_tokens=2,
            )
        if len(requests) == 2:
            tool_output = next(
                item for item in body["input"] if item.get("type") == "function_call_output"
            )
            assert tool_output["call_id"] == "call_read-file"
            assert "native-openai-file-sentinel" in tool_output["output"]
            assert '<tool_data source="read_file">' in tool_output["output"]
            return _openai_response_stream(
                "resp_agent-answer",
                message("The file contains native-openai-file-sentinel."),
                input_tokens=20,
                output_tokens=4,
            )
        assert len(requests) == 3, "Unexpected extra model call"
        assert not body.get("tools")
        assert "native-openai-file-sentinel" in json.dumps(body["input"])
        assert body["text"]["format"] == {
            "type": "json_schema",
            "name": "structured_output",
            "strict": True,
            "schema": {**schema, "additionalProperties": False},
        }
        return httpx.Response(
            200,
            json=_openai_response(
                "resp_structured-answer",
                message('{"sentinel": "native-openai-file-sentinel"}'),
                input_tokens=5,
                output_tokens=3,
            ),
        )

    transport = httpx.MockTransport(respond)
    sync_clients: list[httpx.Client] = []
    async_clients: list[httpx.AsyncClient] = []

    def create_http_client() -> httpx.Client:
        client = httpx.Client(transport=transport)
        sync_clients.append(client)
        return client

    def create_async_http_client() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=transport)
        async_clients.append(client)
        return client

    monkeypatch.setattr(tls, "create_http_client", create_http_client)
    monkeypatch.setattr(tls, "create_async_http_client", create_async_http_client)
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-win")
    monkeypatch.setenv("HOME", str(tmp_path))
    sdk, route = resolve_model_route(
        model_family=ResolvedModelFamily(
            name="openai",
            hosting="openai",
            model="gpt-4.1-mini",
            api_key="resolved-openai-key",
            expected_envs=("OPENAI_API_KEY",),
        )
    )
    assert sdk is None
    assert route is not None
    provider = DeepAgentsProvider(route)
    try:
        events = [
            event
            async for event in provider.query(
                ProviderQueryOptions(
                    prompt=f"Read {sentinel_file} and report its contents.",
                    system_prompt="Read the requested file and report its contents unchanged.",
                    model="gpt-4.1-mini",
                    cwd=str(tmp_path),
                    max_turns=10,
                    allowed_tools=["Read"],
                    output_schema=schema,
                    tool_output_inspection_enabled=False,
                )
            )
        ]
    finally:
        for sync_client in sync_clients:
            sync_client.close()
        for async_client in async_clients:
            await async_client.aclose()

    assert len(requests) == 3
    calls = [event for event in events if isinstance(event, ToolCallEvent)]
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].call_id == "call_read-file"
    assert json.loads(calls[0].input) == {"file_path": str(sentinel_file)}
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].call_id == "call_read-file"
    assert results[0].error_type is None
    assert "native-openai-file-sentinel" in results[0].output
    assert isinstance(events[-1], ResultEvent)
    assert json.loads(events[-1].text) == {"sentinel": "native-openai-file-sentinel"}
    # Both streamed generations and the shaping call contribute exactly once.
    assert (events[-1].input_tokens, events[-1].output_tokens) == (35, 9)

    generations = [
        span for span in span_exporter.get_finished_spans() if span.name == "chat gpt-4.1-mini"
    ]
    assert [
        (
            span.attributes["gen_ai.response.id"],
            span.attributes["gen_ai.usage.input_tokens"],
            span.attributes["gen_ai.usage.output_tokens"],
        )
        for span in generations
    ] == [
        ("resp_read-file", 10, 2),
        ("resp_agent-answer", 20, 4),
        ("resp_structured-answer", 5, 3),
    ]
    for span in generations:
        assert span.attributes["gen_ai.provider.name"] == "openai"
        assert span.attributes["gen_ai.response.model"] == "gpt-4.1-mini"


def test_openai_model_requires_resolved_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-be-used")

    with pytest.raises(ValueError, match="resolved OpenAI API key"):
        create_native_openai_model(ModelParameters(model="gpt-4.1-mini"))


@pytest.mark.asyncio
async def test_openai_model_clients_are_closed_by_deepagents_cleanup(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
) -> None:
    from lightspeed_agentic.providers.deepagents import _close_model_clients
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    model = create_native_openai_model(
        ModelParameters(model="gpt-4.1-mini", api_key="resolved-openai-key")
    )

    await _close_model_clients(model)

    assert http_clients[0].is_closed
    assert http_clients[1].is_closed


def test_openai_model_module_import_does_not_load_optional_sdk() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lightspeed_agentic.providers.models.openai; "
            "assert 'langchain_openai' not in sys.modules",
        ],
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": os.path.join(project_root, "src")},
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
