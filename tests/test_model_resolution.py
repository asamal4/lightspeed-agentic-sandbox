"""Tests for model resolution from environment variables."""

from __future__ import annotations

import pytest

from lightspeed_agentic.config import resolve_router_model, resolve_startup_model


def test_resolve_router_model_prefers_explicit_model() -> None:
    assert resolve_router_model("openai", "custom-model") == "custom-model"


def test_resolve_startup_model_prefers_lightspeed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    assert resolve_startup_model("openai") == "gpt-5-mini"


def test_resolve_startup_model_uses_sdk_env_when_lightspeed_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIGHTSPEED_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    assert resolve_startup_model("openai") == "gpt-4.1"


def test_resolve_startup_model_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGHTSPEED_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_startup_model("openai") is None


def test_resolve_router_model_prefers_lightspeed_when_both_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    assert resolve_router_model("openai") == "gpt-5-mini"


def test_resolve_router_model_uses_sdk_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGHTSPEED_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    assert resolve_router_model("openai") == "gpt-4.1"


def test_resolve_router_model_falls_back_to_lightspeed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-5-mini")
    assert resolve_router_model("openai") == "gpt-5-mini"


def test_resolve_router_model_requires_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LIGHTSPEED_MODEL", raising=False)
    with pytest.raises(ValueError, match="No model configured"):
        resolve_router_model("openai")


def test_resolve_router_model_raises_for_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown SDK provider: 'bogus'"):
        resolve_router_model("bogus")


def test_resolve_startup_model_raises_for_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGHTSPEED_MODEL", raising=False)
    with pytest.raises(ValueError, match="Unknown SDK provider: 'bogus'"):
        resolve_startup_model("bogus")
