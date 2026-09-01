"""Anomaly Detector 후보 모델. 정상(NORMAL) 데이터만으로 학습하는 비지도 이상치 탐지 모델이다.

두 모델 모두 `score_samples()`를 공통 인터페이스로 지원해(값이 낮을수록 비정상 - sklearn 규약)
`anomaly_metrics.py`에서 동일한 방식으로 방향을 통일하고 정규화한다. LocalOutlierFactor는 거리
기반이라 StandardScaler를 앞에 둔다. IsolationForest는 트리 기반이라 스케일링을 강제하지 않는다.
"""

from __future__ import annotations

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# 비교할 후보 모델 dict 생성 (전부 정상 데이터만으로 fit)
def build_anomaly_candidate_models() -> dict:
    return {
        "IsolationForest": IsolationForest(n_estimators=200, random_state=RANDOM_STATE),
        "LocalOutlierFactor": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LocalOutlierFactor(n_neighbors=20, novelty=True)),
            ]
        ),
    }
