"""Shared helpers for tests that don't need pytest fixtures."""

from lightspeed_agentic.providers.model_router import (
    ModelCreator,
    ModelParameters,
    ResolvedModelRoute,
)

# Workflow budgets include SDK middleware graph steps, not just model calls.
TEST_WORKFLOW_MAX_TURNS = 20


def make_model_route(
    create_model: ModelCreator, *, model: str = "claude-sonnet-4-6"
) -> ResolvedModelRoute:
    """Build an explicit route around an offline test model constructor."""
    return ResolvedModelRoute(
        route="anthropic",
        model=model,
        create_model=create_model,
        model_parameters=ModelParameters(model=model),
        telemetry_provider="anthropic",
        structured_output_method="function_calling",
    )
