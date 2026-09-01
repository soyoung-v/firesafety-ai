"""Current Regressor용 feature - 회로 전류(`current`) 시계열 자체의 통계량만 사용한다.

`risk.py`의 `cur_*` feature와 계산식을 공유한다(`endpoint_slope`를 그대로 재사용) - 같은 통계를
두 파일에 다르게 구현하지 않는다. min/max/last는 risk.py에는 없어 여기서 새로 추가했다.
"""

from __future__ import annotations

import pandas as pd

from .risk import endpoint_slope

# 모델 입력 순서 고정
CURRENT_PREDICTION_FEATURE_NAMES = [
    "current_mean",
    "current_std",
    "current_min",
    "current_max",
    "current_range",
    "current_diff_abs",
    "current_slope",
    "current_last",
]


# 전류 시계열 feature 계산 (window_rows는 current 컬럼을 가진 60행 슬라이스)
def compute_current_features(window_rows: pd.DataFrame) -> dict:
    cur = window_rows["current"]
    return {
        "current_mean": cur.mean(),
        "current_std": cur.std(),
        "current_min": cur.min(),
        "current_max": cur.max(),
        "current_range": cur.max() - cur.min(),
        "current_diff_abs": cur.diff().abs().mean(),
        "current_slope": endpoint_slope(cur),
        "current_last": cur.iloc[-1],
    }
