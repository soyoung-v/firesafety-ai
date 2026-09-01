"""저장된 Risk Classifier artifact를 로드해 추론하는 재사용 가능한 wrapper.

향후 FastAPI에서 riskLevel/riskScore 확장 필드를 제공할 수 있도록 설계했다 - 이번 Phase에서는
endpoint에 연결하지 않는다.

riskScore 정의(이 Phase에서 처음 확정): predict_proba에 심각도 가중치(NORMAL=0, WARNING=0.5,
DANGER=1.0)를 곱한 기댓값이다. anomalyScore와 동일하게 "0에 가까움=낮은 위험, 1에 가까움=높은
위험"이 되도록 방향을 맞췄다. Phase 4까지는 riskScore 계산식이 문서화된 적이 없었다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_artifact, load_metadata

SEVERITY_WEIGHTS = {"NORMAL": 0.0, "WARNING": 0.5, "DANGER": 1.0}


class RiskClassifier:
    def __init__(self, model, feature_names: list[str], label_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        self.label_names = label_names

    # 저장된 artifact 경로에서 RiskClassifier 인스턴스 생성
    @classmethod
    def load(cls, model_path: Path) -> "RiskClassifier":
        model = load_artifact(model_path)
        metadata = load_metadata(model_path)
        return cls(model=model, feature_names=metadata["feature_names"], label_names=metadata["label_names"])

    # 위험도 예측 - Feature Matrix(1행 이상)에 대해 riskLevel/riskScore를 계산
    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        ordered = X[self.feature_names]
        proba = self.model.predict_proba(ordered)
        classes = list(self.model.classes_)
        risk_level = self.model.predict(ordered)

        weights = np.array([SEVERITY_WEIGHTS[label] for label in classes])
        risk_score = proba @ weights

        return pd.DataFrame({"riskLevel": risk_level, "riskScore": risk_score}, index=ordered.index)
