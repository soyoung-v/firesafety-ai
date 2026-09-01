"""저장된 Legacy ARC Classifier artifact를 로드해 추론하는 재사용 가능한 wrapper.

향후 FastAPI에서 기존 Spring Boot 계약(pred/proba)을 그대로 제공할 수 있도록 설계했다 - 이번
Phase에서는 endpoint에 연결하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .arc_io import load_arc_artifact, load_arc_metadata
from .arc_metrics import resolve_pred


class ArcClassifier:
    def __init__(self, model, feature_names: list[str], threshold: float) -> None:
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold

    # 저장된 artifact 경로에서 ArcClassifier 인스턴스 생성
    @classmethod
    def load(cls, model_path: Path) -> "ArcClassifier":
        model = load_arc_artifact(model_path)
        metadata = load_arc_metadata(model_path)
        return cls(model=model, feature_names=metadata["feature_names"], threshold=metadata["threshold"])

    # 아크 위험 예측 - Feature Matrix(1행 이상)에 대해 pred(0/1)/proba(ARC 확률)를 계산
    def predict_arc(self, X: pd.DataFrame) -> pd.DataFrame:
        ordered = X[self.feature_names]
        proba = self.model.predict_proba(ordered)[:, 1]
        pred = resolve_pred(proba, self.threshold)
        return pd.DataFrame({"pred": pred, "proba": proba}, index=ordered.index)
