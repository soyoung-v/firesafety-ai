"""Feature Engineering 파이프라인. window.py + legacy_arc.py/risk.py + labels.py + split.py를 잇는다.

모델별 Feature Dataset을 분리해서 만든다 (ADR-002):
- Legacy ARC Feature Dataset: NORMAL/ARC scenario window만, 기존 레거시 7개 feature, pred(0/1) label
- General Risk Feature Dataset: 6개 scenario 전체 window, 신규 risk feature, risk_level(3-class) label
"""

from __future__ import annotations

import pandas as pd

from . import legacy_arc, risk
from .columns import ARC_COLUMNS, RISK_COLUMNS
from .labels import resolve_arc_label, resolve_window_risk_level
from .split import assign_splits
from .window import DEFAULT_STRIDE, WINDOW_SIZE, build_windows

# Legacy ARC Feature Dataset에 포함할 시나리오 (ADR-002: LEAKAGE/OVERHEATING 등을 ARC로 변환하지 않는다)
ARC_SCENARIOS = ("NORMAL", "ARC")


# 원본 raw Dataset(Phase 2 산출물)으로부터 Legacy ARC / Risk 두 Feature Dataset을 생성
def build_feature_datasets(
    raw_df: pd.DataFrame,
    seed: int,
    window_size: int = WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_by_run = assign_splits(raw_df, seed)
    windows = build_windows(raw_df, window_size=window_size, stride=stride)

    arc_rows: list[dict] = []
    risk_rows: list[dict] = []

    for window in windows:
        window_risk_level = resolve_window_risk_level(window.rows)
        meta = {
            "run_id": window.run_id,
            "scenario": window.scenario,
            "window_index": window.window_index,
            "window_start": window.window_start,
            "window_end": window.window_end,
            "split": split_by_run[window.run_id],
        }

        risk_rows.append(
            {**meta, **risk.compute_risk_features(window.rows), "risk_level": window_risk_level}
        )

        if window.scenario in ARC_SCENARIOS:
            arc_label = resolve_arc_label(window.scenario, window_risk_level)
            arc_rows.append(
                {**meta, **legacy_arc.compute_legacy_arc_features(window.rows), "pred": arc_label}
            )

    arc_df = pd.DataFrame(arc_rows, columns=ARC_COLUMNS)
    risk_df = pd.DataFrame(risk_rows, columns=RISK_COLUMNS)
    return arc_df, risk_df
