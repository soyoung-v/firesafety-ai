"""Current Prediction Feature Dataset 생성 파이프라인.

Phase 3의 window.py(run_id 경계 보존)와 split.py(train/val/test 계층화 분할)를 그대로 재사용한다.
target(다음 sample의 current)이 같은 run 안에 존재하지 않는 window(각 run의 마지막 window)는
Dataset에서 제외한다 - target leakage 및 run 경계 교차 방지 (Phase 6 명세 2절).
"""

from __future__ import annotations

import pandas as pd

from .current_columns import ALL_COLUMNS
from .current_prediction import compute_current_features
from .current_target import build_current_lookup, next_sample_current
from .split import assign_splits
from .window import DEFAULT_STRIDE, WINDOW_SIZE, build_windows


# raw Dataset(Phase 2 산출물)으로부터 Current Prediction Feature Dataset 생성
def build_current_prediction_dataset(
    raw_df: pd.DataFrame,
    seed: int,
    window_size: int = WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> pd.DataFrame:
    split_by_run = assign_splits(raw_df, seed)
    windows = build_windows(raw_df, window_size=window_size, stride=stride)
    current_lookup = build_current_lookup(raw_df)

    rows: list[dict] = []
    for window in windows:
        target = next_sample_current(current_lookup, window.run_id, window.window_end)
        if target is None:
            continue  # run의 마지막 window - 다음 sample이 없어 target을 만들 수 없음

        meta = {
            "run_id": window.run_id,
            "scenario": window.scenario,
            "window_index": window.window_index,
            "window_start": window.window_start,
            "window_end": window.window_end,
            "split": split_by_run[window.run_id],
        }
        rows.append({**meta, **compute_current_features(window.rows), "target_current": target})

    return pd.DataFrame(rows, columns=ALL_COLUMNS)
