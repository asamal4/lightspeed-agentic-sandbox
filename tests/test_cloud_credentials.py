"""Tests for Vertex ADC and service-account impersonation credentials."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import pytest


def _mounted_credentials(tmp_path: Any, target: str | None = None) -> None:
    source_file = tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    source_file.write_text("{}", encoding="utf-8")
    if target is not None:
        (tmp_path / "target_service_account").write_text(target, encoding="utf-8")


def test_without_target_leaves_adc_resolution_to_the_provider_sdk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth

    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path)
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )

    def unexpected_adc_lookup(**_kwargs: Any) -> Any:
        raise AssertionError("ADC must remain owned by the provider SDK without a target")

    monkeypatch.setattr(google.auth, "default", unexpected_adc_lookup)

    assert create_vertex_credentials() is None


def test_target_creates_impersonated_credentials_from_adc_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import credentials, impersonated_credentials

    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    target = "vertex-agent@example.iam.gserviceaccount.com"
    _mounted_credentials(tmp_path, target)
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )
    source = credentials.AnonymousCredentials()
    monkeypatch.setattr(google.auth, "default", lambda **_kwargs: (source, "source-project"))

    impersonated = create_vertex_credentials()

    assert isinstance(impersonated, impersonated_credentials.Credentials)
    assert impersonated.service_account_email == target


def test_empty_target_fails_instead_of_using_source_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth

    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "   \n")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )

    def unexpected_adc_lookup(**_kwargs: Any) -> Any:
        raise AssertionError("invalid configured impersonation must not fall back to ADC")

    monkeypatch.setattr(google.auth, "default", unexpected_adc_lookup)

    with pytest.raises(ValueError, match=r"target_service_account.*empty"):
        create_vertex_credentials()


def test_adc_failure_propagates_when_target_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth

    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )
    failure = RuntimeError("source credentials unavailable")

    def fail_adc_lookup(**_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(google.auth, "default", fail_adc_lookup)

    with pytest.raises(RuntimeError, match="source credentials unavailable"):
        create_vertex_credentials()


def test_google_auth_refreshes_the_returned_impersonated_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import impersonated_credentials
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )
    source = Credentials(
        token="source-token",  # noqa: S106 - Fake source credential for offline refresh test.
        expiry=datetime.now() + timedelta(hours=1),
    )
    monkeypatch.setattr(google.auth, "default", lambda **_kwargs: (source, "source-project"))
    requests: list[tuple[str, str, bytes, dict[str, str]]] = []

    class Response:
        status = 200
        data = json.dumps(
            {
                "accessToken": "target-token",
                "expireTime": (datetime.now(UTC) + timedelta(hours=1)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
        ).encode("utf-8")
        headers: ClassVar[dict[str, str]] = {}

    def fake_request(url: str, method: str, *, body: bytes, headers: dict[str, str]) -> Response:
        requests.append((url, method, body, headers))
        return Response()

    impersonated = create_vertex_credentials()
    assert isinstance(impersonated, impersonated_credentials.Credentials)

    impersonated.refresh(fake_request)

    assert impersonated.token == "target-token"  # noqa: S105 - Fake token from the mock response.
    assert len(requests) == 1
    url, method, body, headers = requests[0]
    assert method == "POST"
    assert url.endswith(
        "/v1/projects/-/serviceAccounts/vertex-agent@example.iam.gserviceaccount.com:generateAccessToken"
    )
    assert json.loads(body) == {
        "delegates": None,
        "scope": ["https://www.googleapis.com/auth/cloud-platform"],
        "lifetime": "3600s",
    }
    assert headers["authorization"] == "Bearer source-token"


def test_impersonation_refresh_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import exceptions, impersonated_credentials
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )
    source = Credentials(
        token="source-token",  # noqa: S106 - Fake source credential in an offline failure test.
        expiry=datetime.now() + timedelta(hours=1),
    )
    monkeypatch.setattr(google.auth, "default", lambda **_kwargs: (source, "source-project"))

    class Response:
        status = 403
        data = b'{"error":"permissionDenied"}'
        headers: ClassVar[dict[str, str]] = {}

    def forbidden_request(*_args: Any, **_kwargs: Any) -> Response:
        return Response()

    impersonated = create_vertex_credentials()
    assert isinstance(impersonated, impersonated_credentials.Credentials)

    with pytest.raises(exceptions.RefreshError):
        impersonated.refresh(forbidden_request)
