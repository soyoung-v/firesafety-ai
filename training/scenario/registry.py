"""시나리오 이름 -> 엔진 클래스 registry. Dataset Generator와 향후 Sensor Simulator가 공유하는 진입점."""

from __future__ import annotations

from .arc import ArcScenario
from .base import ScenarioEngine
from .complex_risk import ComplexRiskScenario
from .leakage import LeakageScenario
from .normal import NormalScenario
from .over_current import OverCurrentScenario
from .overheating import OverheatingScenario

SCENARIOS: dict[str, type[ScenarioEngine]] = {
    "NORMAL": NormalScenario,
    "OVER_CURRENT": OverCurrentScenario,
    "ARC": ArcScenario,
    "LEAKAGE": LeakageScenario,
    "OVERHEATING": OverheatingScenario,
    "COMPLEX_RISK": ComplexRiskScenario,
}


# 이름으로 시나리오 엔진 인스턴스 생성
def get_scenario(name: str) -> ScenarioEngine:
    return SCENARIOS[name]()
