from __future__ import annotations

import pytest
from pydantic import ValidationError

from lightspeed_agentic.inspection.models import (
    ClassifierDecision,
    ClassifierRequest,
)


def _request(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "toolName": "get_pod_logs",
        "resultType": "result",
        "chunkIndex": 1,
        "chunkCount": 2,
        "content": "pod output",
    }
    value.update(overrides)
    return value


def test_classifier_request_accepts_contract_shape() -> None:
    request = ClassifierRequest.model_validate(_request())

    assert request.model_dump(by_alias=True) == _request()


def test_classifier_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ClassifierRequest.model_validate(_request(history="secret"))


@pytest.mark.parametrize("field", ["chunkIndex", "chunkCount"])
def test_classifier_request_rejects_boolean_integer_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        ClassifierRequest.model_validate(_request(**{field: True}))


def test_classifier_decision_accepts_benign_none() -> None:
    decision = ClassifierDecision.model_validate({"injectionDetected": False, "category": "none"})

    assert decision.injection_detected is False
    assert decision.category == "none"


def test_classifier_decision_accepts_malicious_unknown() -> None:
    decision = ClassifierDecision.model_validate({"injectionDetected": True, "category": "unknown"})

    assert decision.injection_detected is True
    assert decision.category == "unknown"


def test_classifier_decision_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ClassifierDecision.model_validate(
            {"injectionDetected": False, "category": "none", "reason": "secret"}
        )


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_classifier_decision_requires_strict_boolean(value: object) -> None:
    with pytest.raises(ValidationError):
        ClassifierDecision.model_validate({"injectionDetected": value, "category": "none"})


@pytest.mark.parametrize(
    "category",
    [
        "instruction_override",
        "role_change",
        "prompt_extraction",
        "data_exfiltration",
        "tool_manipulation",
        "unknown",
    ],
)
def test_malicious_categories_are_allowed(category: str) -> None:
    decision = ClassifierDecision.model_validate({"injectionDetected": True, "category": category})

    assert decision.category == category


def test_benign_decision_requires_none_category() -> None:
    with pytest.raises(ValidationError):
        ClassifierDecision.model_validate({"injectionDetected": False, "category": "unknown"})


def test_malicious_decision_requires_non_none_category() -> None:
    with pytest.raises(ValidationError):
        ClassifierDecision.model_validate({"injectionDetected": True, "category": "none"})


def test_decision_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        ClassifierDecision.model_validate({"injectionDetected": True, "category": "other"})
