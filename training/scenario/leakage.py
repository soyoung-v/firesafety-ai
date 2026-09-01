"""LEAKAGE(누전) 시나리오 - dataset-spec.md 3-4절. leakage_current는 AI /predict 계약 밖 확장 후보 필드다."""

from __future__ import annotations

from . import config
from .base import ScenarioEngine, ScenarioSpec


def build_spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="LEAKAGE",
        phases=config.build_three_phase_spec("LEAKAGE", config.LEAKAGE_WARNING, config.LEAKAGE_DANGER),
    )


class LeakageScenario(ScenarioEngine):
    def __init__(self) -> None:
        super().__init__(build_spec())
