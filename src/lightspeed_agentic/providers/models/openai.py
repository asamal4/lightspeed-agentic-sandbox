"""Native and OpenAI-compatible chat model construction for DeepAgents."""

from __future__ import annotations

import json
from operator import itemgetter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel, LanguageModelInput
    from langchain_core.runnables import Runnable

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


def openai_compatible_structured_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap an unchanged schema for compatible Chat Completions shaping."""
    return {"name": "structured_output", "schema": schema, "strict": False}


def bind_openai_compatible_output(
    model: BaseChatModel, schema: dict[str, Any]
) -> Runnable[LanguageModelInput, dict[str, Any]]:
    """Bind prepared shaping schema without native model-name capability heuristics."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough

    parsed = RunnablePassthrough.assign(
        parsed=itemgetter("raw") | StrOutputParser() | RunnableLambda(json.loads),
        parsing_error=lambda _: None,
    ).with_fallbacks(
        [RunnablePassthrough.assign(parsed=lambda _: None)], exception_key="parsing_error"
    )
    return (
        RunnableMap(raw=model.bind(response_format={"type": "json_schema", "json_schema": schema}))
        | parsed
    )


def _responses_reasoning_kwargs(reasoning_config: dict[str, Any] | None) -> dict[str, Any]:
    """Map Responses reasoning options without mutating or filtering configuration."""
    if not reasoning_config:
        return {}
    reasoning = dict(reasoning_config)
    kwargs: dict[str, Any] = {}
    if "verbosity" in reasoning:
        kwargs["verbosity"] = reasoning.pop("verbosity")
    if reasoning:
        kwargs["reasoning"] = reasoning
    return kwargs


def _chat_completions_reasoning_kwargs(
    reasoning_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map Chat Completions options, preserving other settings for SDK validation."""
    if not reasoning_config:
        return {}
    reasoning = dict(reasoning_config)
    kwargs: dict[str, Any] = {}
    if "verbosity" in reasoning:
        kwargs["verbosity"] = reasoning.pop("verbosity")
    if "effort" in reasoning:
        kwargs["reasoning_effort"] = reasoning.pop("effort")
    if reasoning:
        kwargs["reasoning"] = reasoning
    return kwargs


def create_native_openai_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct a native Responses API model using resolved credentials and shared TLS."""
    from langchain_openai import ChatOpenAI

    return _create_openai_model(
        parameters,
        model_type=ChatOpenAI,
        use_responses_api=True,
        model_options=_responses_reasoning_kwargs(reasoning_config),
    )


def create_openai_compatible_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct a Chat Completions model for an explicitly resolved custom endpoint."""
    if not parameters.base_url or not parameters.base_url.strip():
        raise ValueError("OpenAI-compatible endpoints require a resolved base URL")

    from langchain_core.outputs import ChatGenerationChunk
    from langchain_openai import ChatOpenAI

    class CompatibleChatOpenAI(ChatOpenAI):
        """Retain server completion IDs and reasoning omitted by SDK stream conversion."""

        def _convert_chunk_to_generation_chunk(
            self,
            chunk: dict[str, Any],
            default_chunk_class: type[Any],
            base_generation_info: dict[str, Any] | None,
        ) -> ChatGenerationChunk | None:
            generation = super()._convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, base_generation_info
            )
            if generation is None:
                return None
            response_id = chunk.get("id")
            if isinstance(response_id, str) and response_id:
                generation.message.response_metadata["id"] = response_id
            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            reasoning = delta.get("reasoning_content")
            if not isinstance(reasoning, str) or not reasoning:
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                generation.message.content = [
                    {"type": "reasoning", "reasoning": reasoning},
                    *(dict(block) for block in generation.message.content_blocks),
                ]
            return generation

    return _create_openai_model(
        parameters,
        model_type=CompatibleChatOpenAI,
        use_responses_api=False,
        model_options=_chat_completions_reasoning_kwargs(reasoning_config),
    )


def _create_openai_model(
    parameters: ModelParameters,
    *,
    model_type: type[BaseChatModel],
    use_responses_api: bool,
    model_options: dict[str, Any],
) -> BaseChatModel:
    """Construct from prepared SDK options using resolved credentials and shared TLS."""
    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    kwargs: dict[str, Any] = {
        "model": parameters.model,
        # Readiness validates presence; an explicit empty value prevents SDK env fallback.
        "api_key": parameters.api_key or "",
        "base_url": parameters.base_url or "https://api.openai.com/v1",
        "use_responses_api": use_responses_api,
        "output_version": "responses/v1" if use_responses_api else "v1",
        "http_client": create_http_client(),
        "http_async_client": create_async_http_client(),
    }
    if parameters.max_output_tokens is not None:
        kwargs["max_tokens"] = parameters.max_output_tokens
    if not use_responses_api:
        # Custom clients disable LangChain's default streaming-usage request.
        kwargs["stream_usage"] = True

    return model_type(**kwargs, **model_options)
