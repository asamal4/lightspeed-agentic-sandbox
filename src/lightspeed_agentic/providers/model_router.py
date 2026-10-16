"""Resolve Anthropic model construction or return an unmigrated adapter's SDK config.

Routing happens at startup, without importing provider SDKs or creating clients.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from lightspeed_agentic.config import ResolvedModelFamily, ResolvedSDK, resolve_model_family
from lightspeed_agentic.types import DEFAULT_MODEL

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class ModelParameters:
    """Resolved connection/model settings; SDKs retain credential lifecycle ownership."""

    model: str
    base_url: str | None = None
    project: str | None = None
    region: str | None = None
    auth_token: str | None = field(default=None, repr=False)


ModelCreator = Callable[[ModelParameters, dict[str, Any] | None], "BaseChatModel"]


@dataclass(frozen=True)
class ResolvedModelRoute:
    """Hosting identity and typed construction inputs for a migrated model route."""

    route: str
    model: str
    create_model: ModelCreator
    model_parameters: ModelParameters = field(repr=False)
    telemetry_provider: str
    structured_output_method: Literal["function_calling", "json_schema"]


def resolve_model_route(
    model: str | None = None,
    *,
    model_family: ResolvedModelFamily | None = None,
) -> tuple[ResolvedSDK, None] | tuple[None, ResolvedModelRoute]:
    """Return (sdk, None) for legacy adapters or (None, model_route) for Anthropic.

    Anthropic model precedence is an explicit override, the resolved family model,
    then the default. Legacy callers retain their existing model resolution. Model
    clients remain unconstructed; readiness is checked by the caller.
    """
    if model_family is None:
        model_family = resolve_model_family()
    match model_family.name:
        case "openai" | "google":
            return ResolvedSDK(
                name="gemini" if model_family.name == "google" else "openai",
                expected_envs=model_family.expected_envs,
                credential_file_envs=model_family.credential_file_envs,
                azure_auth_mode=model_family.azure_auth_mode,
                azure_credentials=model_family.azure_credentials,
            ), None
        case "anthropic":
            from lightspeed_agentic.providers.models.anthropic import (
                create_bedrock_model,
                create_direct_model,
                create_vertex_model,
            )

            # The family snapshots model and host settings; explicit route overrides win.
            effective_model = model or model_family.model or DEFAULT_MODEL
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
                            project=model_family.project or "",
                            region=model_family.region or "us-east5",
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
                            region=model_family.region or "us-east-1",
                        ),
                        telemetry_provider="aws.bedrock",
                        structured_output_method="function_calling",
                    )
                case "anthropic":
                    # Direct Anthropic.
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
                            auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN") or None,
                        ),
                        telemetry_provider="anthropic",
                        structured_output_method=method,
                    )
                case "openai" | "azure":
                    raise ValueError(
                        f"Unsupported hosting provider for Anthropic: {model_family.hosting}"
                    )
