"""Offline tests for Vertex Gemini model construction."""

from __future__ import annotations

import ssl
from pathlib import Path
from typing import Any

import pytest
from google.auth.credentials import AnonymousCredentials
from langchain_google_genai import ChatGoogleGenerativeAI

from lightspeed_agentic import tls
from lightspeed_agentic.config import (
    VertexCredentialSource,
    VertexProviderParameters,
)
from lightspeed_agentic.providers import cloud_credentials
from lightspeed_agentic.providers.model_router import ModelParameters
from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model


def test_vertex_gemini_model_uses_impersonated_credentials_and_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = AnonymousCredentials()
    ssl_context = ssl.create_default_context()
    reasoning_config: dict[str, Any] = {"thinking_budget": 1024, "include_thoughts": True}
    monkeypatch.setattr(cloud_credentials, "create_vertex_credentials", lambda _source: credentials)
    monkeypatch.setattr(tls, "get_ssl_context", lambda: ssl_context)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    model = create_vertex_gemini_model(
        ModelParameters(
            model="gemini-2.5-flash",
            base_url="https://vertex-proxy.example.test/custom",
            provider=VertexProviderParameters(project="test-project", region="us-central1"),
            credentials=VertexCredentialSource(Path("/unused/credentials.json")),
        ),
        reasoning_config,
    )

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "gemini-2.5-flash"
    assert model.vertexai is True
    assert model.project == "test-project"
    assert model.location == "us-central1"
    assert model.base_url == "https://vertex-proxy.example.test/custom"
    assert model.credentials is credentials
    assert model.thinking_config == reasoning_config
    http_options = model.client._api_client._http_options
    assert http_options.base_url == "https://vertex-proxy.example.test/custom"
    assert http_options.client_args == {"verify": ssl_context}
    assert http_options.async_client_args == {"verify": ssl_context}


def test_vertex_gemini_model_leaves_adc_to_sdk_without_impersonation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cloud_credentials, "create_vertex_credentials", lambda _source: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    model = create_vertex_gemini_model(
        ModelParameters(
            model="gemini-2.5-flash",
            provider=VertexProviderParameters(project="test-project", region="us-central1"),
            credentials=VertexCredentialSource(Path("/unused/credentials.json")),
        )
    )

    assert model.vertexai is True
    assert model.credentials is None
    assert model.thinking_config is None
