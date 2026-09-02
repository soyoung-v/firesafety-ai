"""app/schema/explain.py 검증 규칙 테스트 - Synthetic Dataset 정답 메타데이터가 섞여도 거절되는지 확인."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schema.explain import ExplainRequest


def _valid_payload(**overrides) -> dict:
    payload = {
        "circuit": 1,
        "prediction": {"pred": 0, "proba": 0.08},
        "sensorEvidence": {"current": 10.9},
    }
    payload.update(overrides)
    return payload


def test_valid_request_parses():
    ExplainRequest(**_valid_payload())


@pytest.mark.parametrize(
    "forbidden_field",
    ["scenario", "risk_level", "run_id", "aerror", "ground_truth_risk_level", "split"],
)
def test_top_level_synthetic_metadata_field_is_rejected(forbidden_field):
    with pytest.raises(ValidationError):
        ExplainRequest(**_valid_payload(**{forbidden_field: "OVERHEATING"}))


def test_prediction_extra_field_is_rejected():
    payload = _valid_payload()
    payload["prediction"]["scenario"] = "ARC"
    with pytest.raises(ValidationError):
        ExplainRequest(**payload)


def test_prediction_pred_out_of_range_rejected():
    payload = _valid_payload()
    payload["prediction"]["pred"] = 2
    with pytest.raises(ValidationError):
        ExplainRequest(**payload)


def test_prediction_risk_score_out_of_range_rejected():
    payload = _valid_payload(prediction={"pred": 0, "proba": 0.08, "riskScore": 1.5})
    with pytest.raises(ValidationError):
        ExplainRequest(**payload)
