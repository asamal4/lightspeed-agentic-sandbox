"""Google ADC and service-account impersonation for Vertex integrations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TARGET_SERVICE_ACCOUNT_FILE = "target_service_account"


def _read_target_service_account() -> str | None:
    """Read the optional target account next to the mounted ADC credential file."""
    source_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not source_path:
        return None

    target_path = Path(source_path).parent / _TARGET_SERVICE_ACCOUNT_FILE
    try:
        target = target_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Unable to read Vertex target_service_account credential") from exc

    if not target:
        raise ValueError("Vertex target_service_account credential is empty")
    return target


def create_vertex_credentials() -> Credentials | None:
    """Return impersonated ADC credentials when a target account is configured.

    When no target is configured, return ``None`` so the provider SDK retains
    its current ADC behavior, including ADC-configured service-account
    impersonation. With a target, Google's auth library owns token refresh.
    """
    target_service_account = _read_target_service_account()
    if target_service_account is None:
        return None

    import google.auth
    from google.auth import impersonated_credentials

    source_credentials, _project_id = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    return impersonated_credentials.Credentials(  # type: ignore[no-untyped-call]
        source_credentials=source_credentials,
        target_principal=target_service_account,
        target_scopes=[_CLOUD_PLATFORM_SCOPE],
    )
