"""Continuous Demo Simulator 전용 이벤트 프레임 시퀀스 생성.

training.scenario의 PhaseSpec/ScenarioEngine/override 상수를 그대로 재사용한다 - 이 파일에는
새 threshold나 mean/std 숫자를 하나도 정의하지 않는다. 표준 registry(get_scenario)는 3-phase
비율(NORMAL 40%/WARNING 30%/DANGER 30%)이 고정돼 있어 "정확히 N개 WARNING + M개 DANGER"를
직접 통제할 수 없기 때문에, 여기서는 같은 override 값으로 길이를 직접 지정한 ScenarioSpec을
그 자리에서 조립한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.scenario import config as scenario_config
from training.scenario.base import PhaseSpec, RiskLevel, ScenarioEngine, ScenarioSpec

from .frame_builder import MAX_CHANNEL_COUNT, PanelFrameSeries

# 위험 이벤트 후보 - config.SCENARIO_OVERRIDES에 이미 있는 4개 + COMPLEX_RISK(합성)
EVENT_SCENARIOS: tuple[str, ...] = ("ARC", "LEAKAGE", "OVERHEATING", "OVER_CURRENT", "COMPLEX_RISK")


# scenario_name의 (warning_override, danger_override)를 training.scenario.config에서 그대로 가져온다.
# COMPLEX_RISK는 매번 rng로 쌍을 골라 두 시나리오의 override를 합성한다(complex_risk.py와 동일 규칙) -
# 같은 seed로 다시 호출하면 rng 소비 순서가 같아 항상 같은 쌍이 나온다(결정론적 재현의 핵심).
def _resolve_overrides(scenario_name: str, rng: np.random.Generator) -> tuple[dict, dict]:
    if scenario_name == "COMPLEX_RISK":
        pair_index = int(rng.integers(0, len(scenario_config.COMPLEX_RISK_PAIRS)))
        name_a, name_b = scenario_config.COMPLEX_RISK_PAIRS[pair_index]
        warning_a, danger_a = scenario_config.SCENARIO_OVERRIDES[name_a]
        warning_b, danger_b = scenario_config.SCENARIO_OVERRIDES[name_b]
        return {**warning_a, **warning_b}, {**danger_a, **danger_b}
    return scenario_config.SCENARIO_OVERRIDES[scenario_name]


# NORMAL 앵커 1프레임(ramp 시작점 계산용, 전송 안 함) + WARNING warning_frames개(ramp) +
# DANGER danger_frames개(hold) 길이의 ScenarioSpec을 조립한다. ratio = count/total이라
# PhaseSpec._phase_sample_counts의 round() 오차 없이 정확히 그 개수가 나온다.
def _build_event_spec(scenario_name: str, warning_override: dict, danger_override: dict, warning_frames: int, danger_frames: int) -> ScenarioSpec:
    total = 1 + warning_frames + danger_frames
    return ScenarioSpec(
        name=scenario_name,
        phases=[
            PhaseSpec(RiskLevel.NORMAL, 1 / total, dict(scenario_config.COMMON_NORMAL_TARGETS)),
            PhaseSpec(RiskLevel.WARNING, warning_frames / total, scenario_config.merged_targets(warning_override), ramp=True),
            PhaseSpec(RiskLevel.DANGER, danger_frames / total, scenario_config.merged_targets(danger_override), ramp=False),
        ],
    )


# 이벤트 전체(target_circuit + 나머지 9개 baseline)를 매번 seed로부터 처음부터 재생성한다.
# 같은 (scenario, seed, warning_frames, danger_frames, target_circuit)이면 항상 같은 DataFrame이 나오므로,
# state.json에는 seed만 저장해두고 매 실행마다 여기서 다시 만들어 필요한 인덱스만 꺼내 쓰면 된다.
def build_event_series(
    scenario_name: str, m_no: str, warning_frames: int, danger_frames: int, target_circuit: int, seed: int
) -> PanelFrameSeries:
    if not 1 <= target_circuit <= MAX_CHANNEL_COUNT:
        raise ValueError(f"target_circuit은 1~{MAX_CHANNEL_COUNT} 범위여야 한다: {target_circuit}")

    total_frames = 1 + warning_frames + danger_frames

    target_rng = np.random.default_rng(seed)
    warning_override, danger_override = _resolve_overrides(scenario_name, target_rng)
    spec = _build_event_spec(scenario_name, warning_override, danger_override, warning_frames, danger_frames)
    target_df: pd.DataFrame = ScenarioEngine(spec).generate_run(run_id=f"demo-{m_no}", samples_per_run=total_frames, rng=target_rng)

    circuit_dfs: dict[int, pd.DataFrame] = {target_circuit: target_df}
    for channel_no in range(1, MAX_CHANNEL_COUNT + 1):
        if channel_no == target_circuit:
            continue
        baseline_rng = np.random.default_rng(seed + 1000 + channel_no)
        circuit_dfs[channel_no] = _normal_baseline(baseline_rng, total_frames, m_no, channel_no)

    return PanelFrameSeries(m_no, target_circuit, total_frames, target_df, circuit_dfs)


def _normal_baseline(rng: np.random.Generator, n: int, m_no: str, channel_no: int) -> pd.DataFrame:
    normal_spec = ScenarioSpec(
        name="NORMAL",
        phases=[PhaseSpec(RiskLevel.NORMAL, 1.0, dict(scenario_config.COMMON_NORMAL_TARGETS))],
    )
    return ScenarioEngine(normal_spec).generate_run(run_id=f"demo-{m_no}-c{channel_no}", samples_per_run=n, rng=rng)
