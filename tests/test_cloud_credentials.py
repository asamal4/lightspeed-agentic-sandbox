"""Tests for cloud credential loading and refresh."""

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


def test_vertex_credentials_load_only_when_the_vertex_route_constructs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    source = Credentials(token="source-token")  # noqa: S106 - Fake offline credential.
    credential_file = tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    credential_file.write_text("invalid JSON is not parsed during routing", encoding="utf-8")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    credentials = create_vertex_credentials(VertexCredentialSource(credential_file))

    assert credentials is source


def test_without_target_loads_the_configured_adc_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path)
    source = Credentials(token="source-token")  # noqa: S106 - Fake offline credential.
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    credentials = create_vertex_credentials(
        VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )

    assert credentials is source


def test_target_creates_impersonated_credentials_from_adc_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import credentials, impersonated_credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    target = "vertex-agent@example.iam.gserviceaccount.com"
    _mounted_credentials(tmp_path, target)
    source = credentials.AnonymousCredentials()
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    impersonated = create_vertex_credentials(
        VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )

    assert isinstance(impersonated, impersonated_credentials.Credentials)
    assert impersonated.service_account_email == target


def test_empty_target_fails_instead_of_using_source_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "   \n")
    source = credentials.AnonymousCredentials()
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    with pytest.raises(ValueError, match=r"target_service_account.*empty"):
        create_vertex_credentials(
            VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
        )


def test_adc_failure_propagates_when_target_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    failure = RuntimeError("source credentials unavailable")

    def fail_file_load(_filename: str, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(google.auth, "load_credentials_from_file", fail_file_load)

    with pytest.raises(RuntimeError, match="source credentials unavailable"):
        create_vertex_credentials(
            VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
        )


def test_google_auth_refreshes_the_returned_impersonated_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import impersonated_credentials
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    source = Credentials(
        token="source-token",  # noqa: S106 - Fake source credential for offline refresh test.
        expiry=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
    )
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )
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

    impersonated = create_vertex_credentials(
        VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )
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

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    source = Credentials(
        token="source-token",  # noqa: S106 - Fake source credential in an offline failure test.
        expiry=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
    )
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    class Response:
        status = 403
        data = b'{"error":"permissionDenied"}'
        headers: ClassVar[dict[str, str]] = {}

    def forbidden_request(*_args: Any, **_kwargs: Any) -> Response:
        return Response()

    impersonated = create_vertex_credentials(
        VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )
    assert isinstance(impersonated, impersonated_credentials.Credentials)

    with pytest.raises(exceptions.RefreshError):
        impersonated.refresh(forbidden_request)


def test_bedrock_credentials_without_role_return_static_botocore_credentials() -> None:
    from botocore.credentials import Credentials, RefreshableCredentials

    from lightspeed_agentic.config import BedrockCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_bedrock_credentials

    credentials = create_bedrock_credentials(
        BedrockCredentialSource(
            access_key_id="static-access",
            secret_access_key="static-secret",  # noqa: S106 - test credential
            session_token="static-session",  # noqa: S106 - test credential
        ),
        "us-east-1",
    )

    assert isinstance(credentials, Credentials)
    assert not isinstance(credentials, RefreshableCredentials)
    frozen = credentials.get_frozen_credentials()
    assert frozen.access_key == "static-access"
    assert frozen.secret_key == "static-secret"  # noqa: S105
    assert frozen.token == "static-session"  # noqa: S105


def test_bedrock_credentials_with_role_assume_and_refresh(monkeypatch: Any) -> None:
    import botocore.credentials
    import botocore.session

    from lightspeed_agentic.config import BedrockCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_bedrock_credentials

    now = datetime.now(UTC)
    clock = [now]
    assume_calls: list[dict[str, Any]] = []

    class FakeSTS:
        def assume_role(self, **kwargs: Any) -> dict[str, Any]:
            assume_calls.append(kwargs)
            suffix = str(len(assume_calls))
            return {
                "Credentials": {
                    "AccessKeyId": f"assumed-access-{suffix}",
                    "SecretAccessKey": f"assumed-secret-{suffix}",
                    "SessionToken": f"assumed-token-{suffix}",
                    "Expiration": clock[0] + timedelta(hours=1),
                }
            }

    client_calls: list[tuple[str, dict[str, Any]]] = []

    def create_client(_session: Any, service_name: str, **kwargs: Any) -> FakeSTS:
        client_calls.append((service_name, kwargs))
        return FakeSTS()

    monkeypatch.setattr(botocore.session.Session, "create_client", create_client)
    monkeypatch.setattr(botocore.credentials, "_local_now", lambda: clock[0])
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-access-must-not-win")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret-must-not-win")
    source = BedrockCredentialSource(
        access_key_id="source-access",
        secret_access_key="source-secret",  # noqa: S106 - test credential
        session_token="source-session",  # noqa: S106 - test credential
        role_arn="arn:aws:iam::123456789012:role/target",
    )

    credentials = create_bedrock_credentials(source, "us-west-2")
    credentials._time_fetcher = lambda: clock[0]
    first = credentials.get_frozen_credentials()
    clock[0] += timedelta(minutes=50)
    second = credentials.get_frozen_credentials()

    assert first.access_key == "assumed-access-1"
    assert first.token == "assumed-token-1"  # noqa: S105
    assert second.access_key == "assumed-access-2"
    assert second.token == "assumed-token-2"  # noqa: S105
    assert len(assume_calls) == 2
    assert assume_calls[0]["RoleArn"] == "arn:aws:iam::123456789012:role/target"
    assert client_calls[0] == (
        "sts",
        {
            "aws_access_key_id": "source-access",
            "aws_secret_access_key": "source-secret",
            "aws_session_token": "source-session",
            "region_name": "us-west-2",
        },
    )


def test_bedrock_role_assumption_error_propagates_without_fallback(
    monkeypatch: Any,
) -> None:
    import botocore.session
    from botocore.exceptions import ClientError

    from lightspeed_agentic.config import BedrockCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_bedrock_credentials

    assume_calls: list[dict[str, Any]] = []

    class FakeSTS:
        def assume_role(self, **kwargs: Any) -> dict[str, Any]:
            assume_calls.append(kwargs)
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "role assumption denied"}},
                "AssumeRole",
            )

    def create_client(_session: Any, _service_name: str, **_kwargs: Any) -> FakeSTS:
        return FakeSTS()

    monkeypatch.setattr(botocore.session.Session, "create_client", create_client)
    source = BedrockCredentialSource(
        access_key_id="source-access",
        secret_access_key="source-secret",  # noqa: S106 - test credential
        role_arn="arn:aws:iam::123456789012:role/target",
    )
    credentials = create_bedrock_credentials(source, "us-west-2")

    with pytest.raises(ClientError) as exc_info:
        credentials.get_frozen_credentials()

    assert exc_info.value.response["Error"]["Code"] == "AccessDenied"
    assert len(assume_calls) == 1
    assert assume_calls[0]["RoleArn"] == "arn:aws:iam::123456789012:role/target"
