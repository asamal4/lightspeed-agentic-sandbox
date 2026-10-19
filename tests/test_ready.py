"""Tests for provider readiness checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from lightspeed_agentic.config import ResolvedModelFamily, ResolvedSDK
from lightspeed_agentic.readiness import (
    check_provider_env,
    read_first_mounted_secret_in_dir,
    read_mounted_secret,
    run_readiness_checks,
)

_ANTHROPIC_DIRECT = ResolvedSDK(
    "deepagents",
    ("ANTHROPIC_API_KEY",),
)

_VERTEX_ANTHROPIC = ResolvedSDK(
    "deepagents",
    ("GOOGLE_APPLICATION_CREDENTIALS",),
    ("GOOGLE_APPLICATION_CREDENTIALS",),
)

_VERTEX_GOOGLE = ResolvedSDK(
    "gemini",
    ("GOOGLE_APPLICATION_CREDENTIALS",),
    ("GOOGLE_APPLICATION_CREDENTIALS",),
)

_OPENAI_DIRECT = ResolvedSDK(
    "openai",
    ("OPENAI_API_KEY",),
)

_BEDROCK = ResolvedSDK(
    "deepagents",
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
)


def test_check_provider_env_anthropic_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert check_provider_env(_ANTHROPIC_DIRECT.expected_envs) == "ok"


def test_check_provider_env_anthropic_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "error: missing" in check_provider_env(_ANTHROPIC_DIRECT.expected_envs)


def test_check_provider_env_anthropic_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/some/path")
    assert "error: missing" in check_provider_env(_ANTHROPIC_DIRECT.expected_envs)


def test_check_provider_env_vertex_anthropic_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "gac.json"
    cred.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    assert (
        check_provider_env(_VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs)
        == "ok"
    )


def test_check_provider_env_vertex_anthropic_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert "error: missing" in check_provider_env(
        _VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs
    )


def test_check_provider_env_vertex_google_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "gac.json"
    cred.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    assert (
        check_provider_env(_VERTEX_GOOGLE.expected_envs, _VERTEX_GOOGLE.credential_file_envs)
        == "ok"
    )


def test_check_provider_env_vertex_google_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert "error: missing" in check_provider_env(
        _VERTEX_GOOGLE.expected_envs, _VERTEX_GOOGLE.credential_file_envs
    )


def test_check_provider_env_credential_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/gac.json")
    result = check_provider_env(
        _VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs
    )
    assert "file not found" in result


def test_check_provider_env_credential_file_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "gac.json"
    cred.write_text("", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    result = check_provider_env(
        _VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs
    )
    assert "file is empty or unreadable" in result


def test_check_provider_env_credential_file_whitespace_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "gac.json"
    cred.write_text("   \n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    result = check_provider_env(
        _VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs
    )
    assert "file is empty or unreadable" in result


def test_check_provider_env_openai_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert check_provider_env(_OPENAI_DIRECT.expected_envs) == "ok"


def test_check_provider_env_openai_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "error: missing" in check_provider_env(_OPENAI_DIRECT.expected_envs)


def test_check_provider_env_bedrock_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert check_provider_env(_BEDROCK.expected_envs) == "ok"


def test_check_provider_env_bedrock_partial_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    assert "error: missing" in check_provider_env(_BEDROCK.expected_envs)


def test_run_readiness_checks_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ok, checks = run_readiness_checks(_OPENAI_DIRECT)
    assert ok is True
    assert checks == {"provider_env": "ok"}


def test_run_readiness_checks_provider_env_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ok, checks = run_readiness_checks(_ANTHROPIC_DIRECT)
    assert ok is False
    assert checks["provider_env"].startswith("error: ")


@pytest.mark.parametrize(
    ("project", "region", "expected_route_status"),
    [
        ("", "us-east5", "error: missing project for vertex/anthropic"),
        ("test-project", "", "error: missing region for vertex/anthropic"),
    ],
)
def test_run_readiness_checks_rejects_vertex_route_without_parameters(
    project: str, region: str, expected_route_status: str
) -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.anthropic import create_vertex_model

    family = ResolvedModelFamily("anthropic", (), hosting="vertex")
    route = ResolvedModelRoute(
        route="vertex/anthropic",
        model="test-model",
        create_model=create_vertex_model,
        model_parameters=ModelParameters(model="test-model", project=project, region=region),
        telemetry_provider="gcp.vertex_ai",
        structured_output_method="json_schema",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {"provider_env": "ok", "model_route": expected_route_status}


@pytest.mark.parametrize(
    ("project", "region", "expected_route_status"),
    [
        (None, "us-central1", "error: missing project for vertex/google"),
        ("test-project", None, "error: missing region for vertex/google"),
        ("", "", "error: missing project, region for vertex/google"),
        ("test-project", "us-central1", "ok"),
    ],
)
def test_run_readiness_checks_validates_vertex_gemini_route(
    project: str | None,
    region: str | None,
    expected_route_status: str,
) -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model

    family = ResolvedModelFamily("google", (), hosting="vertex")
    route = ResolvedModelRoute(
        route="vertex/google",
        model="gemini-2.5-flash",
        create_model=create_vertex_gemini_model,
        model_parameters=ModelParameters(
            model="gemini-2.5-flash",
            project=project,
            region=region,
        ),
        telemetry_provider="gcp.vertex_ai",
        structured_output_method="json_schema",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is (expected_route_status == "ok")
    assert checks == {"provider_env": "ok", "model_route": expected_route_status}


def test_run_readiness_checks_rejects_bedrock_route_without_region() -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.anthropic import create_bedrock_model

    family = ResolvedModelFamily("anthropic", (), hosting="bedrock")
    route = ResolvedModelRoute(
        route="bedrock/anthropic",
        model="test-model",
        create_model=create_bedrock_model,
        model_parameters=ModelParameters(model="test-model"),
        telemetry_provider="aws.bedrock",
        structured_output_method="function_calling",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {
        "provider_env": "ok",
        "model_route": "error: missing region for bedrock/anthropic",
    }


def test_run_readiness_checks_accepts_default_bedrock_region() -> None:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    family = ResolvedModelFamily("anthropic", (), hosting="bedrock", model="test-model")
    _, route = resolve_model_route(model_family=family)

    assert isinstance(route, ResolvedModelRoute)
    ready, checks = run_readiness_checks(family, route)

    assert ready is True
    assert checks == {"provider_env": "ok", "model_route": "ok"}


def test_read_mounted_secret_returns_content(tmp_path: Path) -> None:
    secret = tmp_path / "key"
    secret.write_text("  secret-value  \n", encoding="utf-8")
    assert read_mounted_secret(secret) == "secret-value"


def test_read_mounted_secret_missing_returns_none(tmp_path: Path) -> None:
    assert read_mounted_secret(tmp_path / "missing") is None


def test_read_mounted_secret_empty_returns_none(tmp_path: Path) -> None:
    secret = tmp_path / "empty"
    secret.write_text("", encoding="utf-8")
    assert read_mounted_secret(secret) is None


def test_read_first_mounted_secret_in_dir_sorted(tmp_path: Path) -> None:
    secret_dir = tmp_path / "mount"
    secret_dir.mkdir()
    (secret_dir / "b").write_text("second", encoding="utf-8")
    (secret_dir / "a").write_text("first", encoding="utf-8")
    assert read_first_mounted_secret_in_dir(secret_dir) == "first"


# ── Azure readiness ──

_AZURE_ENTRA = ResolvedSDK(
    "openai",
    (),  # no env vars needed — credentials are file-based
    azure_auth_mode="entra_id",
    azure_credentials={
        "client_id": "cid",
        "tenant_id": "tid",
        "client_secret": "csec",
    },
)

_AZURE_API_KEY = ResolvedSDK(
    "openai",
    ("AZURE_OPENAI_API_KEY",),
    azure_auth_mode="api_key",
)


def test_run_readiness_checks_azure_entra_ok() -> None:
    """Entra ID mode with empty expected_envs passes readiness."""
    ok, checks = run_readiness_checks(_AZURE_ENTRA)
    assert ok is True
    assert checks["provider_env"] == "ok"


def test_run_readiness_checks_azure_api_key_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """API key mode passes when AZURE_OPENAI_API_KEY is set."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    ok, checks = run_readiness_checks(_AZURE_API_KEY)
    assert ok is True
    assert checks["provider_env"] == "ok"


def test_run_readiness_checks_azure_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """API key mode fails when AZURE_OPENAI_API_KEY is missing."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    ok, checks = run_readiness_checks(_AZURE_API_KEY)
    assert ok is False
    assert "error: missing" in checks["provider_env"]
