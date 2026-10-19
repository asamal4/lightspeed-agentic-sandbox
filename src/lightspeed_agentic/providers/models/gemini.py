"""Vertex Gemini chat model construction for DeepAgents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from lightspeed_agentic.providers.model_router import ModelParameters


def create_vertex_gemini_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct a Vertex Gemini model with shared TLS and optional impersonation."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials
    from lightspeed_agentic.tls import get_ssl_context

    kwargs: dict[str, Any] = {
        "model": parameters.model,
        "vertexai": True,
        "project": parameters.project,
        "location": parameters.region,
        "client_args": {"verify": get_ssl_context()},
    }
    if parameters.base_url is not None:
        kwargs["base_url"] = parameters.base_url
    credentials = create_vertex_credentials()
    if credentials is not None:
        kwargs["credentials"] = credentials
    if reasoning_config:
        kwargs["thinking_config"] = reasoning_config
    return ChatGoogleGenerativeAI(**kwargs)
