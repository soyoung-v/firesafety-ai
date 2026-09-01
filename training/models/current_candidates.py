"""Current Regressor 후보. Persistence Baseline과 ML 회귀 모델 3종을 비교한다.

Ridge는 스케일에 민감하므로 StandardScaler를 앞에 둔 Pipeline으로 감싼다. RandomForest/
HistGradientBoosting은 트리 기반이라 스케일링을 강제하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class PersistenceBaseline:
    """다음 전류값을 현재 window의 마지막 전류값과 같다고 예측하는 최소 baseline."""

    # 학습할 파라미터가 없다
    def fit(self, X, y=None):
        return self

    # current_last를 그대로 다음 전류 예측치로 사용
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["current_last"].to_numpy()


# 비교할 후보 dict 생성 (Persistence Baseline 포함)
def build_current_candidate_models() -> dict:
    return {
        "PersistenceBaseline": PersistenceBaseline(),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }
