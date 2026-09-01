"""ARC 시나리오 - dataset-spec.md 3-3절.

기존 레거시 AI는 명시적 수치 임계치 없이 패턴(전류 급변 + arc_count 누적)으로 판정했다 - 이 시나리오도
절대 임계값이 아니라 변동성/누적 패턴으로 WARNING/DANGER를 구성한다 [SIM].
"""

from __future__ import annotations

from . import config
from .base import ScenarioEngine, ScenarioSpec


def build_spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="ARC",
        phases=config.build_three_phase_spec("ARC", config.ARC_WARNING, config.ARC_DANGER),
    )


class ArcScenario(ScenarioEngine):
    def __init__(self) -> None:
        super().__init__(build_spec())
