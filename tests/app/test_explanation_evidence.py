"""app/service/explanation_evidence.py 단위 테스트. deterministic 순수 함수라 LLM 호출 없이 검증한다."""

from __future__ import annotations

from app.schema.explain import ExplainRequest, PredictionEvidence, SensorEvidence, TrendEvidence
from app.service.explanation_evidence import build_evidence


def _prediction(**overrides) -> PredictionEvidence:
    base = dict(
        pred=0, proba=0.08, riskLevel="WARNING", riskScore=0.63,
        anomaly=True, anomalyScore=0.78, predictedCurrent=12.4,
    )
    base.update(overrides)
    return PredictionEvidence(**base)


def test_build_evidence_includes_circuit_and_prediction_as_is():
    request = ExplainRequest(
        circuit=3,
        prediction=_prediction(),
        sensorEvidence=SensorEvidence(current=10.9, arcCount=0, temperature=72.3,
                                       leakageCurrent=11.0, totalCurrent=18.2),
    )

    evidence = build_evidence(request)

    assert evidence["circuit"] == 3
    # ML 결과 값이 손실/변형 없이 그대로 들어가야 한다 - Evidence Builder는 재계산하지 않는다
    assert evidence["prediction"]["riskLevel"] == "WARNING"
    assert evidence["prediction"]["riskScore"] == 0.63
    assert evidence["prediction"]["anomaly"] is True
    assert evidence["prediction"]["predictedCurrent"] == 12.4


def test_build_evidence_drops_none_fields_instead_of_faking_values():
    request = ExplainRequest(
        circuit=1,
        prediction=_prediction(riskLevel=None, riskScore=None, anomaly=None, anomalyScore=None),
        sensorEvidence=SensorEvidence(current=5.0),
    )

    evidence = build_evidence(request)

    assert "riskLevel" not in evidence["prediction"]
    assert "riskScore" not in evidence["prediction"]
    assert "arcCount" not in evidence["sensorEvidence"]
    assert "temperature" not in evidence["sensorEvidence"]


def test_build_evidence_omits_trend_when_not_provided():
    request = ExplainRequest(
        circuit=1, prediction=_prediction(), sensorEvidence=SensorEvidence(current=5.0),
    )

    evidence = build_evidence(request)

    assert "trendEvidence" not in evidence


def test_build_evidence_includes_trend_when_provided():
    request = ExplainRequest(
        circuit=1, prediction=_prediction(), sensorEvidence=SensorEvidence(current=5.0),
        trendEvidence=TrendEvidence(currentSlope=0.12, temperatureSlope=0.81),
    )

    evidence = build_evidence(request)

    assert evidence["trendEvidence"] == {"currentSlope": 0.12, "temperatureSlope": 0.81}


def test_evidence_never_contains_synthetic_ground_truth_keys():
    # 스키마 자체가 이런 필드를 받지 않으므로(extra=forbid) 여기서는 evidence dict에도 절대 등장하지 않음을 재확인한다
    request = ExplainRequest(
        circuit=1, prediction=_prediction(), sensorEvidence=SensorEvidence(current=5.0),
    )

    evidence = build_evidence(request)

    forbidden_keys = {"scenario", "risk_level", "run_id", "aerror", "ground_truth", "split"}
    import json

    serialized = json.dumps(evidence)
    for key in forbidden_keys:
        assert key not in serialized
