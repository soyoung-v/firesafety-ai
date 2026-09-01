"""circuit 단위로 raw sample -> Feature -> 4-model inference -> 결과 조합.

ARC Classifier는 기존 계약이라 항상 계산한다. Risk/Anomaly는 context가 충분할 때만, Current
Regressor는 current 이력만 있으면(=circuit 샘플만으로) 계산한다 - 계산 불가능한 항목은 가짜 값을
채우지 않고 null로 둔다(Phase 8 명세 4절).
"""

from __future__ import annotations

import pandas as pd

from training.features.current_prediction import compute_current_features
from training.features.legacy_arc import compute_legacy_arc_features
from training.features.risk import compute_risk_features

from . import feature_service
from .model_registry import ModelRegistry


class InsufficientSamplesError(ValueError):
    """circuit의 샘플 수가 최소 기준(30개) 미만 - 기존 계약과 동일하게 요청 전체를 거절한다."""

    def __init__(self, circuit: int, n_samples: int) -> None:
        self.circuit = circuit
        self.n_samples = n_samples
        super().__init__(
            f"circuit {circuit}: 샘플 부족(최소 {feature_service.MIN_SAMPLES}개 필요, 현재 {n_samples}개)"
        )


# circuit 1개에 대해 4-model 추론을 수행해 결과 dict 생성
def predict_circuit(registry: ModelRegistry, circuit_req, context_samples=None) -> dict:
    circuit_frame = feature_service.circuit_samples_to_frame(circuit_req.samples)
    n_samples = len(circuit_frame)

    if n_samples < feature_service.MIN_SAMPLES:
        raise InsufficientSamplesError(circuit_req.circuit, n_samples)

    warning = None
    if n_samples < feature_service.RECOMMENDED_SAMPLES:
        warning = f"샘플 수({n_samples})가 권장치({feature_service.RECOMMENDED_SAMPLES}) 미만입니다"

    result: dict = {
        "circuit": circuit_req.circuit,
        "n_samples": n_samples,
        "warning": warning,
        "riskLevel": None,
        "riskScore": None,
        "anomaly": None,
        "anomalyScore": None,
        "predictedCurrent": None,
    }

    # ARC Classifier - 기존 계약, current+arc_count만 있으면 항상 계산
    arc_model = registry.get("arcClassifier")
    arc_features = pd.DataFrame([compute_legacy_arc_features(circuit_frame)])
    arc_out = arc_model.predict_arc(arc_features).iloc[0]
    result["pred"] = int(arc_out["pred"])
    result["proba"] = float(arc_out["proba"])

    # Current Regressor - current 이력만 있으면 계산 가능(context 불필요)
    current_model = registry.get("currentRegressor")
    if current_model is not None:
        current_features = pd.DataFrame([compute_current_features(circuit_frame)])
        predicted = current_model.predict_next_current(current_features).iloc[0]
        result["predictedCurrent"] = float(predicted)

    # Risk/Anomaly - context가 충분히 제공된 경우에만 계산
    if feature_service.context_is_sufficient(context_samples, n_samples):
        risk_input_frame = feature_service.build_risk_input_frame(circuit_frame, context_samples)
        risk_features = pd.DataFrame([compute_risk_features(risk_input_frame)])

        risk_model = registry.get("riskClassifier")
        if risk_model is not None:
            risk_out = risk_model.predict_risk(risk_features).iloc[0]
            result["riskLevel"] = str(risk_out["riskLevel"])
            result["riskScore"] = float(risk_out["riskScore"])

        anomaly_model = registry.get("anomalyDetector")
        if anomaly_model is not None:
            anomaly_out = anomaly_model.predict(risk_features).iloc[0]
            result["anomaly"] = bool(anomaly_out["anomaly"])
            result["anomalyScore"] = float(anomaly_out["anomalyScore"])

    return result
