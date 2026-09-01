"""Window 단위 label 결정 규칙.

한 window 안에서 risk_level이 섞일 수 있다(NORMAL->WARNING 전이가 window 경계와 어긋나는 경우).
max-severity(window 내 최고 위험도)를 채택했다 - 근거는 feature-spec.md "Window Label 결정 방식" 절.
"""

from __future__ import annotations

import pandas as pd

_SEVERITY = {"NORMAL": 0, "WARNING": 1, "DANGER": 2}
_SEVERITY_REVERSE = {v: k for k, v in _SEVERITY.items()}


# window 내 최고 위험도를 최종 risk_level label로 채택 (max-severity 정책)
def resolve_window_risk_level(window_rows: pd.DataFrame) -> str:
    max_severity = int(window_rows["risk_level"].map(_SEVERITY).max())
    return _SEVERITY_REVERSE[max_severity]


# Legacy ARC 이진 label: 이 window가 실제로 ARC 상태(risk_level != NORMAL)였는지.
# scenario가 ARC라도 아직 NORMAL 구간이면 0 - LEAKAGE/OVERHEATING 등은 애초에 이 subset에 포함되지 않는다(ADR-002)
def resolve_arc_label(scenario: str, window_risk_level: str) -> int:
    if scenario == "ARC" and window_risk_level != "NORMAL":
        return 1
    return 0
