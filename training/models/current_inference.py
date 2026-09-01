"""저장된 Current Regressor artifact를 로드해 추론하는 재사용 가능한 wrapper.

향후 FastAPI에서 predictedCurrent를 그대로 제공할 수 있도록 설계했다 - 이번 Phase에서는
endpoint에 연결하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .current_io import load_current_artifact, load_current_metadata


class CurrentRegressor:
    def __init__(self, model, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names

    # 저장된 artifact 경로에서 CurrentRegressor 인스턴스 생성
    @classmethod
    def load(cls, model_path: Path) -> "CurrentRegressor":
        model = load_current_artifact(model_path)
        metadata = load_current_metadata(model_path)
        return cls(model=model, feature_names=metadata["feature_names"])

    # 다음 전류 예측 - Feature Matrix(1행 이상)에 대해 predictedCurrent를 계산
    def predict_next_current(self, X: pd.DataFrame) -> pd.Series:
        ordered = X[self.feature_names]
        predictions = self.model.predict(ordered)
        return pd.Series(predictions, index=ordered.index, name="predictedCurrent")
