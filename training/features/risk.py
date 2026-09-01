"""신규 Risk Classifier용 feature.

원천은 current/arc_count/leakage_current/temperature/total_current 5개만 사용한다.
voltage/humidity/gas_raw/fire_raw/door_open/total_power는 제외했다 - Dataset Generator
(`training/scenario/config.py`)가 이 필드들을 어떤 시나리오/risk_level에서도 baseline에서 변화시키지
않으므로 구성상 판별력이 없다(Phase 3 완료 보고에 실측 분산 수치 기록). 데이터를 늘리기 위해
기계적으로 모든 센서에 모든 통계를 복제하지 않는다 - feature-spec.md "2. 확장 feature" 참고.
"""

from __future__ import annotations

import pandas as pd

# 모델 입력 순서 고정 (feature-spec.md 2절과 동일)
RISK_FEATURE_NAMES = [
    "cur_mean",
    "cur_std",
    "cur_range",
    "cur_diff_abs",
    "curendpoint_slope",
    "cnt_mean",
    "cnt_std",
    "cnt_range",
    "cnt_diff_abs",
    "leak_mean",
    "leak_std",
    "leak_max",
    "leakendpoint_slope",
    "temp_mean",
    "temp_std",
    "temp_max",
    "tempendpoint_slope",
    "tcur_mean",
    "tcur_std",
    "tcur_max",
    "tcurendpoint_slope",
]


# 윈도우 끝값-시작값 기반 단순 기울기 (선형회귀 대신 endpoint 기반 - 재현성/설명력 우선, feature-spec.md 참고)
def endpoint_slope(series: pd.Series) -> float:
    n = len(series)
    if n < 2:
        return 0.0
    return (series.iloc[-1] - series.iloc[0]) / (n - 1)


# Risk Classifier용 feature 계산 (window_rows는 원천 5개 컬럼을 가진 60행 슬라이스)
def compute_risk_features(window_rows: pd.DataFrame) -> dict:
    cur = window_rows["current"]
    cnt = window_rows["arc_count"]
    leak = window_rows["leakage_current"]
    temp = window_rows["temperature"]
    tcur = window_rows["total_current"]

    return {
        "cur_mean": cur.mean(),
        "cur_std": cur.std(),
        "cur_range": cur.max() - cur.min(),
        "cur_diff_abs": cur.diff().abs().mean(),
        "curendpoint_slope": endpoint_slope(cur),
        "cnt_mean": cnt.mean(),
        "cnt_std": cnt.std(),
        "cnt_range": cnt.max() - cnt.min(),
        "cnt_diff_abs": cnt.diff().abs().mean(),
        "leak_mean": leak.mean(),
        "leak_std": leak.std(),
        "leak_max": leak.max(),
        "leakendpoint_slope": endpoint_slope(leak),
        "temp_mean": temp.mean(),
        "temp_std": temp.std(),
        "temp_max": temp.max(),
        "tempendpoint_slope": endpoint_slope(temp),
        "tcur_mean": tcur.mean(),
        "tcur_std": tcur.std(),
        "tcur_max": tcur.max(),
        "tcurendpoint_slope": endpoint_slope(tcur),
    }
