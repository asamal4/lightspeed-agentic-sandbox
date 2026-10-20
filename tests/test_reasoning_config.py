"""Tests for reasoning config parsing and provider adapter wiring."""

# The test doubles intentionally model optional SDKs without complete type stubs.
# mypy: ignore-errors

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lightspeed_agentic.types import ProviderQueryOptions


def _base_options(**overrides) -> ProviderQueryOptions:
    defaults = {
        "prompt": "test prompt",
        "system_prompt": "test system",
        "model": "test-model",
        "max_turns": 10,
        "allowed_tools": ["Bash"],
        "cwd": "/tmp/test",  # noqa: S108
    }
    defaults.update(overrides)
    return ProviderQueryOptions(**defaults)


class TestOpenAIReasoningConfig:
    @pytest.mark.asyncio
    async def test_no_reasoning_config(self) -> None:
        """When reasoning_config is None, no model_settings is passed."""
        agent_kwargs = await self._run_openai(reasoning_config=None)
        assert "model_settings" not in agent_kwargs

    @pytest.mark.asyncio
    async def test_reasoning_keys(self) -> None:
        """Recognized keys build ModelSettings with Reasoning."""
        reasoning = {"effort": "high", "mode": "chain_of_thought", "verbosity": "high"}
        agent_kwargs = await self._run_openai(reasoning_config=reasoning)
        ms = agent_kwargs["model_settings"]
        assert ms.reasoning.effort == "high"
        assert ms.reasoning.mode == "chain_of_thought"
        assert ms.verbosity == "high"

    @pytest.mark.asyncio
    async def test_all_keys_passed_through(self) -> None:
        """All non-verbosity keys are forwarded to Reasoning."""
        reasoning = {"effort": "low", "unknown_stuff": True}
        agent_kwargs = await self._run_openai(reasoning_config=reasoning)
        ms = agent_kwargs["model_settings"]
        assert ms.reasoning.effort == "low"
        assert ms.reasoning.unknown_stuff is True

    @staticmethod
    async def _run_openai(reasoning_config=None):
        """Run OpenAIProvider.query() with mocks and return the SandboxAgent kwargs."""
        from lightspeed_agentic.providers.openai import OpenAIProvider

        class FakeReasoning:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class FakeModelSettings:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        captured_kwargs = {}

        mock_result = MagicMock()

        async def _empty_events():
            return
            yield

        mock_result.stream_events = _empty_events
        mock_result.final_output = "done"
        mock_result.context_wrapper.usage.input_tokens = 10
        mock_result.context_wrapper.usage.output_tokens = 5

        def _capture_sandbox_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        mock_sandbox_agent_cls = MagicMock(side_effect=_capture_sandbox_agent)

        with (
            patch("agents.sandbox.SandboxAgent", mock_sandbox_agent_cls),
            patch("agents.Runner.run_streamed", return_value=mock_result),
            patch("agents.models.openai_responses.OpenAIResponsesModel", MagicMock()),
            patch("openai.AsyncOpenAI", MagicMock()),
            patch("agents.model_settings.ModelSettings", FakeModelSettings),
            patch("openai.types.shared.Reasoning", FakeReasoning),
        ):
            provider = OpenAIProvider()
            provider._client = MagicMock()
            options = _base_options(reasoning_config=reasoning_config)
            events = provider.query(options)
            async for _ in events:
                pass

        return captured_kwargs
