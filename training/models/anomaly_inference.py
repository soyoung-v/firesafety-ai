"""저장된 Anomaly Detector artifact를 로드해 추론하는 재사용 가능한 wrapper.

향후 FastAPI에서 그대로 재사용할 수 있도록 설계했다 - 이번 Phase에서는 endpoint에 연결하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .anomaly_io import load_anomaly_artifact, load_anomaly_metadata
from .anomaly_metrics import normalize_score, raw_to_anomaly_score, resolve_anomaly


class AnomalyDetector:
    def __init__(
        self,
        model,
        feature_names: list[str],
        score_min: float,
        score_max: float,
        threshold: float,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.score_min = score_min
        self.score_max = score_max
        self.threshold = threshold

    # 저장된 artifact 경로에서 AnomalyDetector 인스턴스 생성
    @classmethod
    def load(cls, model_path: Path) -> "AnomalyDetector":
        model = load_anomaly_artifact(model_path)
        metadata = load_anomaly_metadata(model_path)
        transform = metadata["score_transform"]
        return cls(
            model=model,
            feature_names=metadata["feature_names"],
            score_min=transform["train_score_min"],
            score_max=transform["train_score_max"],
            threshold=metadata["anomaly_threshold"],
        )

    # Feature Matrix(1행 이상)에 대해 raw_score / anomalyScore / anomaly를 계산
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ordered = X[self.feature_names]
        raw_score = raw_to_anomaly_score(self.model, ordered)
        anomaly_score = normalize_score(raw_score, self.score_min, self.score_max)
        anomaly = resolve_anomaly(anomaly_score, self.threshold)
        return pd.DataFrame(
            {"raw_score": raw_score, "anomalyScore": anomaly_score, "anomaly": anomaly},
            index=ordered.index,
        )
