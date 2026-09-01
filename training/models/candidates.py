"""Risk Classifier 후보 모델. random_state를 고정해 재현 가능하게 한다.

LogisticRegression은 스케일에 민감하므로 StandardScaler를 앞에 두는 Pipeline으로 감싼다.
RandomForest/HistGradientBoosting은 트리 기반이라 스케일링을 강제하지 않는다.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# 비교할 후보 모델 dict 생성 (이름 -> fit 가능한 estimator/Pipeline)
def build_candidate_models() -> dict:
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
