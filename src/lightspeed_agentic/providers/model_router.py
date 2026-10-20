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
    DEFAULT_VERTEX_REGION,
    ResolvedModelFamily,
    ResolvedSDK,
    resolve_model_family,
)

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
    structured_output_schema_mode: Literal["pydantic", "raw_json_schema"] = "pydantic"


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
                    project=model_family.project,
                    region=model_family.region or DEFAULT_VERTEX_REGION,
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
                            project=model_family.project or "",
                            region=model_family.region or DEFAULT_VERTEX_REGION,
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
                            auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN") or None,
                        ),
                        telemetry_provider="anthropic",
                        structured_output_method=method,
                    )
