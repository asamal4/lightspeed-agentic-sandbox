"""Native OpenAI chat model construction for DeepAgents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from lightspeed_agentic.config import ModelParameters


def make_openai_json_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply native OpenAI strict-schema requirements without mutating input."""
    if not isinstance(schema, dict):
        return schema

    schema = dict(schema)
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
        schema["properties"] = {
            key: make_openai_json_schema_strict(value)
            for key, value in schema["properties"].items()
        }
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = make_openai_json_schema_strict(schema["items"])
    if "oneOf" in schema and isinstance(schema["oneOf"], list):
        schema["anyOf"] = [*schema.get("anyOf", []), *schema.pop("oneOf")]
    for keyword in ("anyOf", "allOf"):
        if keyword in schema and isinstance(schema[keyword], list):
            schema[keyword] = [make_openai_json_schema_strict(item) for item in schema[keyword]]
    if "not" in schema and isinstance(schema["not"], dict):
        schema["not"] = make_openai_json_schema_strict(schema["not"])
    for defs_key in ("$defs", "definitions"):
        if defs_key in schema and isinstance(schema[defs_key], dict):
            schema[defs_key] = {
                name: make_openai_json_schema_strict(value)
                for name, value in schema[defs_key].items()
            }
    return schema


def native_openai_structured_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a strict raw JSON Schema in LangChain's OpenAI output-format shape."""
    return {
        "name": "structured_output",
        "schema": make_openai_json_schema_strict(schema),
        "strict": True,
    }


def create_native_openai_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct a native Responses API model using resolved credentials and shared TLS."""
    if not parameters.api_key or not parameters.api_key.strip():
        raise ValueError("OpenAI requires a resolved OpenAI API key")

    from langchain_openai import ChatOpenAI

    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    kwargs: dict[str, Any] = {
        "model": parameters.model,
        "api_key": parameters.api_key,
        "base_url": parameters.base_url or "https://api.openai.com/v1",
        "use_responses_api": True,
        "output_version": "responses/v1",
        "http_client": create_http_client(),
        "http_async_client": create_async_http_client(),
    }
    if reasoning_config:
        reasoning = {key: value for key, value in reasoning_config.items() if key != "verbosity"}
        if reasoning:
            kwargs["reasoning"] = reasoning
        if "verbosity" in reasoning_config:
            kwargs["verbosity"] = reasoning_config["verbosity"]

    return ChatOpenAI(**kwargs)
