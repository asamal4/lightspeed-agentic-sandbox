"""Anthropic chat model constructors for direct, Bedrock, and Vertex hosting.

Routing inputs are explicit; native SDKs retain their credential loading and
refresh behavior. Provider SDK imports stay local to the selected constructor.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from lightspeed_agentic.config import ModelParameters


def _tls_model_class(
    model_class: Any,
    client_class: Any,
    async_client_class: Any,
    **client_kwargs: Any,
) -> type[BaseChatModel]:
    """Wrap lazy native clients with shared TLS-configured HTTP clients."""
    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    class TLSChatModel(model_class):  # type: ignore[misc]  # Dynamic optional SDK base class.
        @cached_property
        def _client(self) -> Any:
            return client_class(
                **self._client_params,
                **client_kwargs,
                http_client=create_http_client(),
            )

        @cached_property
        def _async_client(self) -> Any:
            return async_client_class(
                **self._client_params,
                **client_kwargs,
                http_client=create_async_http_client(),
            )

    return cast("type[BaseChatModel]", TLSChatModel)


def create_direct_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct a direct Anthropic client with shared TLS and optional bearer auth."""
    from anthropic import Anthropic, AsyncAnthropic
    from langchain_anthropic import ChatAnthropic

    from lightspeed_agentic.config import AnthropicCredentialSource

    model_class = _tls_model_class(ChatAnthropic, Anthropic, AsyncAnthropic)
    credential_source = cast(AnthropicCredentialSource, parameters.credentials)
    kwargs: dict[str, Any] = {
        "model": parameters.model,
        "api_key": credential_source.api_key,
    }
    if parameters.base_url is not None:
        kwargs["base_url"] = parameters.base_url
    thinking = reasoning_config.get("thinking") if reasoning_config else None
    if thinking:
        kwargs["thinking"] = thinking
    if credential_source.auth_token:
        kwargs["default_headers"] = {"Authorization": f"Bearer {credential_source.auth_token}"}
    return model_class(**kwargs)


def create_bedrock_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct an Anthropic Bedrock model with static or refreshable AWS credentials."""
    from lightspeed_agentic.config import BedrockCredentialSource, BedrockProviderParameters
    from lightspeed_agentic.providers.cloud_credentials import configure_bedrock_credentials

    provider = cast(BedrockProviderParameters, parameters.provider)
    credential_source = cast(BedrockCredentialSource, parameters.credentials)

    from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
    from langchain_aws import ChatAnthropicBedrock

    client_class, async_client_class, client_kwargs, credential_kwargs = (
        configure_bedrock_credentials(
            credential_source,
            provider.region,
            AnthropicBedrock,
            AsyncAnthropicBedrock,
            client_kwargs={"base_url": parameters.base_url},
        )
    )

    # The integration has no stable HTTPX injection point; keep the native-client
    # override aligned with the installed Anthropic SDK.
    model_class = _tls_model_class(
        ChatAnthropicBedrock,
        client_class,
        async_client_class,
        **client_kwargs,
    )
    kwargs: dict[str, Any] = {
        "model": parameters.model,
        "region_name": provider.region,
        **credential_kwargs,
    }
    thinking = reasoning_config.get("thinking") if reasoning_config else None
    if thinking:
        kwargs["thinking"] = thinking
    return model_class(**kwargs)


def create_vertex_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct an Anthropic Vertex client for the supplied project and region."""
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials
    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    provider = cast(VertexProviderParameters, parameters.provider)
    credential_source = cast(VertexCredentialSource, parameters.credentials)
    credentials = create_vertex_credentials(credential_source)
    kwargs: dict[str, Any] = {
        "model_name": parameters.model,
        "project": provider.project,
        "location": provider.region,
        "http_client": create_http_client(),
        "async_http_client": create_async_http_client(),
    }
    kwargs["credentials"] = credentials
    if parameters.base_url is not None:
        # Vertex SDK overrides replace its versioned default base URL entirely.
        endpoint = urlsplit(parameters.base_url)
        if endpoint.path in ("", "/"):
            endpoint = endpoint._replace(path="/v1")
        kwargs["base_url"] = endpoint.geturl()
    thinking = reasoning_config.get("thinking") if reasoning_config else None
    if thinking:
        kwargs["thinking"] = thinking
    return ChatAnthropicVertex(**kwargs)
