"""Feature Dataset 컬럼 정의 - 단일 소스. Feature Matrix와 metadata/label을 명확히 분리한다.

scenario/risk_level/aerror/device_arc_flag/run_id는 절대 FEATURE 목록에 들어가지 않는다
(Label Leakage 방지 - Phase 3 명세 5절).
"""

from __future__ import annotations

from .legacy_arc import FEATURE_NAMES as ARC_FEATURE_COLUMNS
from .risk import RISK_FEATURE_NAMES as RISK_FEATURE_COLUMNS

# 식별/컨텍스트 컬럼 - feature도 label도 아님
META_COLUMNS = ["run_id", "scenario", "window_index", "window_start", "window_end", "split"]

# ML 입력으로 절대 포함하면 안 되는 컬럼 (label leakage 방지 검증용)
FORBIDDEN_FEATURE_COLUMNS = ["scenario", "risk_level", "aerror", "device_arc_flag", "run_id", "pred"]

ARC_LABEL_COLUMNS = ["pred"]
RISK_LABEL_COLUMNS = ["risk_level"]

ARC_COLUMNS = META_COLUMNS + list(ARC_FEATURE_COLUMNS) + ARC_LABEL_COLUMNS
RISK_COLUMNS = META_COLUMNS + list(RISK_FEATURE_COLUMNS) + RISK_LABEL_COLUMNS
