"""NORMAL 시나리오 - dataset-spec.md 3-1절."""

from __future__ import annotations

from . import config
from .base import PhaseSpec, RiskLevel, ScenarioEngine, ScenarioSpec


# 정상 시나리오 스펙 생성 - 전체 구간이 NORMAL, 위험도 전이 없음
def build_spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="NORMAL",
        phases=[PhaseSpec(RiskLevel.NORMAL, 1.0, dict(config.COMMON_NORMAL_TARGETS))],
    )


class NormalScenario(ScenarioEngine):
    def __init__(self) -> None:
        super().__init__(build_spec())
