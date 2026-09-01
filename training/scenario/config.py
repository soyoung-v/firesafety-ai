"""시나리오 파라미터 단일 소스. 숫자를 여러 파일에 중복 하드코딩하지 않는다 (AGENTS.md 절대 규칙).

임계값 출처 구분 (dataset-spec.md 3절 표기 규칙과 동일):
- [HW]/[BE]: 하드웨어 프로토콜 문서 + firesafety-be 코드 기본값(DeviceAlertService/PanelStatusAggregationService)과
  일치가 확인된 값. LEAKAGE 20mA, OVERHEATING 80도, HUMIDITY 80%, OVER_CURRENT 30A(total_current 기준).
- [SIM]: 이 문서/코드에서 새로 정한 시뮬레이션용 가정값. 실제 전기안전 기준이 아니다.
- GAS/FIRE는 임계값 자체가 [TBD]이므로 여기서 위험 판정용 수치를 만들지 않는다(baseline 노이즈만 부여).
"""

from __future__ import annotations

from .base import FieldTarget, PhaseSpec, RiskLevel

# --- 실행 기본값 ---
DEFAULT_SEED = 42
DEFAULT_RUNS_PER_SCENARIO = 40  # dataset-spec.md 8절 제안값(30~50)의 중간값 [SIM]
DEFAULT_SAMPLES_PER_RUN = 400  # dataset-spec.md 8절 제안 범위(300~600) 내 [SIM]

# NORMAL : WARNING : DANGER 비율. dataset-spec.md 8절의 클래스 비율 제안과 동일하게 맞춘다 [SIM]
NORMAL_RATIO = 0.4
WARNING_RATIO = 0.3
DANGER_RATIO = 0.3

# --- 확인된 하드웨어/백엔드 임계값 (물리 단위) ---
LEAK_MA_THRESHOLD = 20.0  # [HW]/[BE] DEFAULT_LEAK_MA_THRESHOLD
TEMP_THRESHOLD_C = 80.0  # [HW]/[BE] DEFAULT_TEMP_THRESHOLD_TENTHS=800
HUMIDITY_THRESHOLD_PCT = 80.0  # [HW]/[BE] DEFAULT_HUMIDITY_THRESHOLD_TENTHS=800 (현재 6개 시나리오에는 미사용)
OVERCURRENT_THRESHOLD_A = 30.0  # [HW]/[BE] DEFAULT_OVERCURRENT_THRESHOLD, total_current 기준

# --- NORMAL 공통 baseline (모든 시나리오의 NORMAL 구간 + 비핵심 필드가 공유) [SIM] ---
COMMON_NORMAL_TARGETS: dict[str, FieldTarget] = {
    "current": FieldTarget(mean=5.0, std=0.3),
    "arc_count": FieldTarget(mean=0.0, std=0.0, cumulative=True),
    "voltage": FieldTarget(mean=224.0, std=3.0),
    "leakage_current": FieldTarget(mean=2.0, std=0.5),
    "temperature": FieldTarget(mean=27.0, std=1.5),
    "humidity": FieldTarget(mean=45.0, std=4.0),
    "fire_raw": FieldTarget(mean=500.0, std=80.0),
    "gas_raw": FieldTarget(mean=500.0, std=80.0),
    "total_current": FieldTarget(mean=10.0, std=1.0),
    "total_power": FieldTarget(mean=2000.0, std=200.0),
}


# 공통 baseline에 시나리오별 override를 덮어씌운 필드 dict 반환
def merged_targets(*overrides: dict[str, FieldTarget]) -> dict[str, FieldTarget]:
    merged = dict(COMMON_NORMAL_TARGETS)
    for override in overrides:
        merged.update(override)
    return merged


# --- 시나리오별 WARNING/DANGER override (핵심 변화 센서만 덮어씀, 나머지는 baseline 유지) ---

OVER_CURRENT_WARNING = {
    "current": FieldTarget(mean=14.0, std=2.0),
    "total_current": FieldTarget(mean=24.0, std=2.0),  # 임계치(30A) 근접 [SIM 상승폭, 임계치는 HW/BE]
}
OVER_CURRENT_DANGER = {
    "current": FieldTarget(mean=28.0, std=4.0),
    "total_current": FieldTarget(mean=33.0, std=3.0),  # >= 30A 지속 [HW]/[BE]
}

ARC_WARNING = {
    "current": FieldTarget(mean=5.0, std=6.0),  # 변동성 증가 시작 [SIM]
    "arc_count": FieldTarget(mean=0.3, std=0.5, cumulative=True),  # 간헐적 증가 [SIM]
}
ARC_DANGER = {
    "current": FieldTarget(mean=5.0, std=15.0),  # 불규칙한 급변 [SIM, CSTech 패턴 근거]
    "arc_count": FieldTarget(mean=2.0, std=1.0, cumulative=True),  # 급증 [SIM]
}

LEAKAGE_WARNING = {
    "leakage_current": FieldTarget(mean=15.0, std=2.0),  # 임계치(20mA) 근접 [SIM 상승폭, 임계치는 HW/BE]
}
LEAKAGE_DANGER = {
    "leakage_current": FieldTarget(mean=24.0, std=3.0),  # >= 20mA 지속 [HW]/[BE]
}

OVERHEATING_WARNING = {
    "temperature": FieldTarget(mean=75.0, std=2.0),  # 임계치(80도) 근접 [SIM 상승폭, 임계치는 HW/BE]
}
OVERHEATING_DANGER = {
    "temperature": FieldTarget(mean=85.0, std=3.0),  # >= 80도 지속 [HW]/[BE]
}

# COMPLEX_RISK가 조합할 시나리오 쌍 후보 [SIM] - firesafety-be에 대응 로직 없는 순수 신규 시나리오
COMPLEX_RISK_PAIRS: list[tuple[str, str]] = [
    ("OVERHEATING", "OVER_CURRENT"),
    ("ARC", "LEAKAGE"),
    ("OVERHEATING", "LEAKAGE"),
]

SCENARIO_OVERRIDES: dict[str, tuple[dict, dict]] = {
    "OVER_CURRENT": (OVER_CURRENT_WARNING, OVER_CURRENT_DANGER),
    "ARC": (ARC_WARNING, ARC_DANGER),
    "LEAKAGE": (LEAKAGE_WARNING, LEAKAGE_DANGER),
    "OVERHEATING": (OVERHEATING_WARNING, OVERHEATING_DANGER),
}


# 3단계(NORMAL/WARNING/DANGER) PhaseSpec 리스트를 override로부터 생성하는 공통 헬퍼.
# WARNING은 NORMAL에서 상승(ramp), DANGER는 진입 즉시 target 수준을 유지(hold)한다 -
# 그래야 "DANGER = 임계치 초과 지속"이 실제로 유지된다(중간에 걸쳐 있는 상태로 라벨링되지 않는다).
def build_three_phase_spec(name: str, warning_override: dict, danger_override: dict) -> list[PhaseSpec]:
    return [
        PhaseSpec(RiskLevel.NORMAL, NORMAL_RATIO, dict(COMMON_NORMAL_TARGETS)),
        PhaseSpec(RiskLevel.WARNING, WARNING_RATIO, merged_targets(warning_override), ramp=True),
        PhaseSpec(RiskLevel.DANGER, DANGER_RATIO, merged_targets(danger_override), ramp=False),
    ]
