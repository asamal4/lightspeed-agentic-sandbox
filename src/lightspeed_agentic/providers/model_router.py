"""Resolve model construction or return an unmigrated adapter's SDK config.

Routing happens at startup, without importing provider SDKs or creating clients.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from lightspeed_agentic.config import (
    DEFAULT_AZURE_API_VERSION,
    ModelParameters,
    ResolvedModelFamily,
    ResolvedSDK,
    VertexCredentialSource,
    _is_native_openai_endpoint,
    resolve_model_family,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel, LanguageModelInput
    from langchain_core.runnables import Runnable


ModelCreator = Callable[[ModelParameters, dict[str, Any] | None], "BaseChatModel"]


def _uses_native_openai_endpoint(model_family: ResolvedModelFamily) -> bool:
    """Select the native OpenAI route only for its public API endpoint."""
    return model_family.hosting == "openai" and _is_native_openai_endpoint(model_family.base_url)


@dataclass(frozen=True)
class ResolvedModelRoute:
    """Hosting identity and typed construction inputs for a migrated model route."""

    route: str
    model: str
    create_model: ModelCreator
    model_parameters: ModelParameters = field(repr=False)
    telemetry_provider: str
    structured_output_method: Literal["function_calling", "json_schema"]
    structured_output_schema_mode: Literal["pydantic", "raw_json_schema"] = "pydantic"
    translate_output_schema: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    bind_structured_output: (
        Callable[[BaseChatModel, dict[str, Any]], Runnable[LanguageModelInput, dict[str, Any]]]
        | None
    ) = None


def resolve_model_route(
    model: str | None = None,
    *,
    model_family: ResolvedModelFamily | None = None,
) -> tuple[ResolvedSDK, None] | tuple[None, ResolvedModelRoute]:
    """Return SDK config for legacy adapters or a typed route for migrated models.

    Route model precedence is an explicit override, then the resolved family model.
    Every model family must resolve a model; model clients remain unconstructed and
    readiness is checked by the caller.
    """
    if model_family is None:
        model_family = resolve_model_family()
    effective_model = model or model_family.model
    if not effective_model:
        raise ValueError(f"No model configured for {model_family.name}")
    match model_family.name:
        case "openai":
            if model_family.hosting == "openai":
                from lightspeed_agentic.providers.models.openai import (
                    bind_openai_compatible_output,
                    create_native_openai_model,
                    create_openai_compatible_model,
                    native_openai_structured_output_schema,
                    openai_compatible_structured_output_schema,
                )

                is_native = _uses_native_openai_endpoint(model_family)
                return None, ResolvedModelRoute(
                    route="openai" if is_native else "openai-compatible",
                    model=effective_model,
                    create_model=(
                        create_native_openai_model if is_native else create_openai_compatible_model
                    ),
                    model_parameters=ModelParameters(
                        model=effective_model,
                        base_url=model_family.base_url,
                        api_key=model_family.api_key,
                    ),
                    telemetry_provider="openai",
                    structured_output_method="json_schema",
                    structured_output_schema_mode="raw_json_schema",
                    translate_output_schema=(
                        native_openai_structured_output_schema
                        if is_native
                        else openai_compatible_structured_output_schema
                    ),
                    bind_structured_output=None if is_native else bind_openai_compatible_output,
                )
            # The legacy OpenAI adapter reads these SDK-specific settings from env.
            if model_family.hosting == "vertex":
                if model_family.model:
                    os.environ["OPENAI_MODEL"] = model_family.model
                if model_family.base_url:
                    os.environ["OPENAI_BASE_URL"] = model_family.base_url
            elif model_family.hosting == "azure":
                if model_family.model:
                    os.environ["OPENAI_MODEL"] = model_family.model
                if model_family.azure_api_key:
                    os.environ["AZURE_OPENAI_API_KEY"] = model_family.azure_api_key
                if model_family.base_url:
                    os.environ["AZURE_OPENAI_ENDPOINT"] = model_family.base_url
                os.environ["AZURE_OPENAI_API_VERSION"] = (
                    model_family.azure_api_version or DEFAULT_AZURE_API_VERSION
                )
            if model_family.hosting == "vertex" and isinstance(
                model_family.credentials, VertexCredentialSource
            ):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
                    model_family.credentials.credentials_file
                )
            return ResolvedSDK(
                name="openai",
                expected_envs=model_family.expected_envs,
                credential_file_envs=model_family.credential_file_envs,
                azure_auth_mode=model_family.azure_auth_mode,
                azure_credentials=model_family.azure_credentials,
            ), None
        case "google":
            from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model

            return None, ResolvedModelRoute(
                route="vertex/google",
                model=effective_model,
                create_model=create_vertex_gemini_model,
                model_parameters=ModelParameters(
                    model=effective_model,
                    base_url=model_family.base_url,
                    provider=model_family.provider,
                    credentials=model_family.credentials,
                ),
                telemetry_provider="gcp.vertex_ai",
                structured_output_method="json_schema",
                structured_output_schema_mode="raw_json_schema",
            )
        case "anthropic":
            from lightspeed_agentic.providers.models.anthropic import (
                create_bedrock_model,
                create_direct_model,
                create_vertex_model,
            )

            # The family snapshots model and host settings; explicit route overrides win.
            base_url = model_family.base_url
            match model_family.hosting:
                case "vertex":
                    return None, ResolvedModelRoute(
                        route="vertex/anthropic",
                        model=effective_model,
                        create_model=create_vertex_model,
                        model_parameters=ModelParameters(
                            model=effective_model,
                            base_url=base_url,
                            provider=model_family.provider,
                            credentials=model_family.credentials,
                        ),
                        telemetry_provider="gcp.vertex_ai",
                        structured_output_method="json_schema",
                    )
                case "bedrock":
                    return None, ResolvedModelRoute(
                        route="bedrock/anthropic",
                        model=effective_model,
                        create_model=create_bedrock_model,
                        model_parameters=ModelParameters(
                            model=effective_model,
                            base_url=base_url,
                            provider=model_family.provider,
                            credentials=model_family.credentials,
                        ),
                        telemetry_provider="aws.bedrock",
                        structured_output_method="function_calling",
                    )
                case _:
                    # Configuration guarantees the remaining hosting is direct Anthropic.
                    method: Literal["function_calling", "json_schema"] = "function_calling"
                    if (
                        base_url
                        and (urlparse(base_url).hostname or "").lower().rstrip(".")
                        != "api.anthropic.com"
                    ):
                        method = "json_schema"
                    return None, ResolvedModelRoute(
                        route="anthropic",
                        model=effective_model,
                        create_model=create_direct_model,
                        model_parameters=ModelParameters(
                            model=effective_model,
                            base_url=base_url,
                            credentials=model_family.credentials,
                        ),
                        telemetry_provider="anthropic",
                        structured_output_method=method,
                    )
