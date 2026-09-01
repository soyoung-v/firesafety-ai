"""Dataset 컬럼 정의 - dataset-spec.md 기준 단일 소스. 코드 여러 곳에 컬럼 목록을 중복 정의하지 않는다.

ADR-001에 따라 aerror/device_arc_flag는 ML feature가 아니다. 이번 Phase 2 구현에는 aerror 파생 컬럼
자체를 생성하지 않았다(선택 사항으로 명시됐던 부분, 다음 단계에서 필요 시 추가) - .agent-docs/dataset-spec.md
및 완료 보고의 TBD 참고.
"""

from __future__ import annotations

# 식별/컨텍스트 컬럼 - feature도 label도 아님
META_COLUMNS = ["run_id", "timestamp", "sample_index", "m_no", "mode", "circuit"]

# ML 입력 feature 후보 (원천 물리값). 이 중 current/arc_count만 현재 /predict 계약에 실제로 쓰인다.
FEATURE_COLUMNS = [
    "current",
    "arc_count",
    "voltage",
    "leakage_current",
    "temperature",
    "humidity",
    "fire_raw",
    "gas_raw",
    "door_open",
    "total_current",
    "total_power",
]

# 정답 라벨 - ML 입력 feature로 사용하지 않는다 (data leakage 방지)
LABEL_COLUMNS = ["scenario", "risk_level"]

ALL_COLUMNS = META_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS

SCENARIOS = ["NORMAL", "OVER_CURRENT", "ARC", "LEAKAGE", "OVERHEATING", "COMPLEX_RISK"]
RISK_LEVELS = ["NORMAL", "WARNING", "DANGER"]
