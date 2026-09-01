"""OVERHEATING(과열) 시나리오 - dataset-spec.md 3-5절. temperature는 AI /predict 계약 밖 확장 후보 필드다."""

from __future__ import annotations

from . import config
from .base import ScenarioEngine, ScenarioSpec


def build_spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="OVERHEATING",
        phases=config.build_three_phase_spec(
            "OVERHEATING", config.OVERHEATING_WARNING, config.OVERHEATING_DANGER
        ),
    )


class OverheatingScenario(ScenarioEngine):
    def __init__(self) -> None:
        super().__init__(build_spec())
