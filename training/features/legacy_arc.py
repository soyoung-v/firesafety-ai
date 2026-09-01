"""기존 레거시 AI의 7개 Legacy ARC feature를 동일 계산식으로 재현한다 (Spring Boot 하위호환 목적).

레거시 참고 구현의 계산식을 Phase 3 착수 전에 직접 확인하고 이식했다 - std의 ddof(pandas 기본
ddof=1), diff의 NaN 처리(첫 행 NaN, `.mean()` 기본 skipna=True), range 계산(max-min) 모두 pandas
기본 동작을 그대로 따르며 추측하지 않았다.
"""

from __future__ import annotations

import pandas as pd

# 모델 입력 순서 고정 (feature-spec.md 1절과 동일, 레거시 feature 순서와 동일)
FEATURE_NAMES = [
    "cnt_std",
    "cnt_mean",
    "cnt_range",
    "cnt_diff_abs",
    "cur_std",
    "cur_mean",
    "cur_range",
]


# 레거시 7개 feature 계산 (window_rows는 current/arc_count 컬럼을 가진 60행 슬라이스)
def compute_legacy_arc_features(window_rows: pd.DataFrame) -> dict:
    cnt = window_rows["arc_count"]
    cur = window_rows["current"]
    return {
        "cnt_std": cnt.std(),
        "cnt_mean": cnt.mean(),
        "cnt_range": cnt.max() - cnt.min(),
        "cnt_diff_abs": cnt.diff().abs().mean(),
        "cur_std": cur.std(),
        "cur_mean": cur.mean(),
        "cur_range": cur.max() - cur.min(),
    }
