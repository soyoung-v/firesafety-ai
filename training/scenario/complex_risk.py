"""COMPLEX_RISK(복합 위험) 시나리오 - dataset-spec.md 3-6절.

firesafety-be에 대응하는 실제 로직이 없는 순수 신규 시나리오다 [SIM]. run마다 config.COMPLEX_RISK_PAIRS
중 하나를 무작위로 골라 두 시나리오의 WARNING/DANGER override를 합쳐서 사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .base import ScenarioEngine, ScenarioSpec


# 두 시나리오 이름으로 합성 ScenarioSpec 생성
def build_spec(pair: tuple[str, str]) -> ScenarioSpec:
    name_a, name_b = pair
    warning_a, danger_a = config.SCENARIO_OVERRIDES[name_a]
    warning_b, danger_b = config.SCENARIO_OVERRIDES[name_b]
    return ScenarioSpec(
        name="COMPLEX_RISK",
        phases=config.build_three_phase_spec(
            "COMPLEX_RISK",
            {**warning_a, **warning_b},
            {**danger_a, **danger_b},
        ),
    )


class ComplexRiskScenario(ScenarioEngine):
    def __init__(self) -> None:
        # 기본 spec은 첫 번째 pair로 두되, run마다 generate_run에서 다시 뽑아 덮어쓴다
        super().__init__(build_spec(config.COMPLEX_RISK_PAIRS[0]))

    # run마다 시나리오 쌍을 무작위로 골라 합성 spec으로 생성
    def generate_run(self, run_id: str, samples_per_run: int, rng: np.random.Generator) -> pd.DataFrame:
        pair_index = int(rng.integers(0, len(config.COMPLEX_RISK_PAIRS)))
        pair = config.COMPLEX_RISK_PAIRS[pair_index]
        spec = build_spec(pair)
        return self._generate_from_spec(spec, run_id, samples_per_run, rng)
