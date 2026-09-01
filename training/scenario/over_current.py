"""OVER_CURRENT 시나리오 - dataset-spec.md 3-2절.

과전류 CAUTION/DANGER 판정은 firesafety-be에서 회로별 current가 아니라 분전반 total_current 기준임을
PanelStatusAggregationMapper.xml SQL로 확인했다(dataset-spec.md 1절 정정 사항) - total_current를 핵심으로 삼는다.
"""

from __future__ import annotations

from . import config
from .base import ScenarioEngine, ScenarioSpec


def build_spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="OVER_CURRENT",
        phases=config.build_three_phase_spec(
            "OVER_CURRENT", config.OVER_CURRENT_WARNING, config.OVER_CURRENT_DANGER
        ),
    )


class OverCurrentScenario(ScenarioEngine):
    def __init__(self) -> None:
        super().__init__(build_spec())
