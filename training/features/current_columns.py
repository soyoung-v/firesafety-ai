"""Current Prediction Feature Dataset 컬럼 정의 - 단일 소스.

`scenario`/`risk_level`/`run_id`/`split`/`aerror`/`device_arc_flag`는 metadata/grouping 전용이며
feature로 쓰지 않는다(Phase 6 명세 4절). `target_current`는 label이며 feature에 포함하지 않는다.
"""

from __future__ import annotations

from .current_prediction import CURRENT_PREDICTION_FEATURE_NAMES as FEATURE_COLUMNS

META_COLUMNS = ["run_id", "scenario", "window_index", "window_start", "window_end", "split"]
TARGET_COLUMN = "target_current"

FORBIDDEN_FEATURE_COLUMNS = [
    "scenario",
    "risk_level",
    "aerror",
    "device_arc_flag",
    "run_id",
    "split",
    TARGET_COLUMN,
]

ALL_COLUMNS = META_COLUMNS + list(FEATURE_COLUMNS) + [TARGET_COLUMN]
