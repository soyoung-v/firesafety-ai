"""시나리오 공통 엔진. 시나리오 규칙(각 scenario/*.py)과 Dataset 저장 코드(training/dataset/)를 분리하기 위한 경계.

Sensor Simulator(향후 구현)도 이 모듈의 ScenarioSpec/PhaseSpec 구조를 그대로 재사용할 수 있도록,
난수 생성과 물리값 시계열 생성 로직만 담당하고 CSV/HTTP 등 출력 방식은 알지 못한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# 물리적으로 음수가 불가능한 feature (dataset-spec.md 13절 검증 규칙과 동일 기준)
NON_NEGATIVE_FIELDS = (
    "current",
    "leakage_current",
    "fire_raw",
    "gas_raw",
    "total_current",
    "total_power",
    "voltage",
)


class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DANGER = "DANGER"


@dataclass
class FieldTarget:
    # 레벨 필드: 이 phase가 끝날 때의 목표값(mean)과 노이즈(std)
    # 누적 필드(cumulative=True, 예: arc_count): 이 phase 동안 매 샘플에 더해지는 증가량의 mean/std
    mean: float
    std: float
    cumulative: bool = False


@dataclass
class PhaseSpec:
    risk_level: RiskLevel
    ratio: float  # 전체 run 샘플 수 중 이 phase가 차지하는 비율
    targets: dict[str, FieldTarget] = field(default_factory=dict)
    # True면 이전 phase의 끝값에서 이 phase의 target까지 선형 상승(예: WARNING).
    # False면 이 phase 시작부터 target에서 노이즈만 두고 유지한다(예: DANGER - "임계치 초과 지속").
    ramp: bool = True


@dataclass
class ScenarioSpec:
    name: str
    phases: list[PhaseSpec]


class ScenarioEngine:
    """시나리오 1개의 run 생성기. 생성자에서 받은 spec을 그대로 쓰거나(대부분의 시나리오),
    COMPLEX_RISK처럼 run마다 다른 spec을 동적으로 만들어 _generate_from_spec에 넘길 수 있다."""

    def __init__(self, spec: ScenarioSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    # 단일 run 생성 (기본 spec 사용)
    def generate_run(self, run_id: str, samples_per_run: int, rng: np.random.Generator) -> pd.DataFrame:
        return self._generate_from_spec(self.spec, run_id, samples_per_run, rng)

    # 주어진 spec으로 물리값 시계열 row를 생성
    def _generate_from_spec(
        self, spec: ScenarioSpec, run_id: str, samples_per_run: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        phase_counts = self._phase_sample_counts(spec, samples_per_run)
        field_names = list(spec.phases[0].targets.keys())

        data = {name: self._generate_field(spec, rng, name, phase_counts) for name in field_names}

        risk_levels: list[str] = []
        for phase, n in zip(spec.phases, phase_counts):
            risk_levels.extend([phase.risk_level.value] * n)

        df = pd.DataFrame(data)
        df["sample_index"] = np.arange(samples_per_run)
        df["risk_level"] = risk_levels
        df["scenario"] = spec.name
        df["run_id"] = run_id

        self._clip_physical_ranges(df)
        return df

    # 위험도 단계별 샘플 개수 계산 (마지막 phase가 나머지를 전부 흡수)
    def _phase_sample_counts(self, spec: ScenarioSpec, samples_per_run: int) -> list[int]:
        counts: list[int] = []
        assigned = 0
        for i, phase in enumerate(spec.phases):
            if i == len(spec.phases) - 1:
                counts.append(samples_per_run - assigned)
            else:
                n = int(round(samples_per_run * phase.ratio))
                counts.append(n)
                assigned += n
        return counts

    # 필드 1개의 phase별 시계열 생성 (레벨 ramp 또는 누적 카운터)
    def _generate_field(
        self, spec: ScenarioSpec, rng: np.random.Generator, field_name: str, phase_counts: list[int]
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        prev_level: float | None = None
        cumulative_value = 0.0

        for phase, n in zip(spec.phases, phase_counts):
            if n <= 0:
                continue
            target = phase.targets[field_name]
            if target.cumulative:
                increments = np.clip(rng.normal(target.mean, target.std, n), 0, None)
                cumvals = cumulative_value + np.cumsum(increments)
                chunks.append(cumvals)
                cumulative_value = cumvals[-1]
            elif phase.ramp:
                # 이전 phase 끝값 -> 이 phase의 target까지 선형 상승 (예: WARNING 진입 과정)
                start = target.mean if prev_level is None else prev_level
                curve = np.linspace(start, target.mean, n)
                noise = rng.normal(0, target.std, n)
                chunks.append(curve + noise)
                prev_level = target.mean
            else:
                # target에서 즉시 시작해 노이즈만 두고 유지 (예: DANGER - 임계치 초과 지속)
                curve = np.full(n, target.mean)
                noise = rng.normal(0, target.std, n)
                chunks.append(curve + noise)
                prev_level = target.mean

        return np.concatenate(chunks) if chunks else np.zeros(0)

    # 물리적으로 불가능한 값(음수 등) 방지 - GAS/FIRE 임계값 판정은 하지 않는다(TBD, dataset-spec.md 3절)
    def _clip_physical_ranges(self, df: pd.DataFrame) -> None:
        for field_name in NON_NEGATIVE_FIELDS:
            if field_name in df.columns:
                df[field_name] = df[field_name].clip(lower=0)
        if "humidity" in df.columns:
            df["humidity"] = df["humidity"].clip(lower=0, upper=100)
        if "arc_count" in df.columns:
            df["arc_count"] = df["arc_count"].round().astype(int)
