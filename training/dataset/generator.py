"""Full Synthetic Dataset 생성 오케스트레이션.

시나리오 규칙(training/scenario/)과 저장 코드(training/dataset/io.py)를 잇는 역할만 한다 -
row를 실제로 만드는 로직은 여기 두지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..scenario.config import DEFAULT_RUNS_PER_SCENARIO, DEFAULT_SAMPLES_PER_RUN, DEFAULT_SEED
from ..scenario.registry import SCENARIOS, get_scenario
from .columns import FEATURE_COLUMNS, LABEL_COLUMNS, META_COLUMNS

# 실제 장비번호(m_no)와 겹치지 않도록 9로 시작하는 synthetic 분전반 pool [SIM]
PANEL_POOL = [f"9{n:04d}" for n in range(1, 6)]

# 실제 센서 전송 주기는 TBD(dataset-spec.md 6절) - 정렬/가독성을 위한 synthetic 값일 뿐, 물리적 근거 없음
TIMESTAMP_INTERVAL_SECONDS = 1
BASE_TIMESTAMP = pd.Timestamp("2026-01-01T00:00:00")
RUN_TIMESTAMP_GAP_SECONDS = 100_000  # run 간 timestamp가 겹치지 않도록 여유를 둔다


class DatasetGenerator:
    def __init__(
        self,
        runs_per_scenario: int = DEFAULT_RUNS_PER_SCENARIO,
        samples_per_run: int = DEFAULT_SAMPLES_PER_RUN,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.runs_per_scenario = runs_per_scenario
        self.samples_per_run = samples_per_run
        self.seed = seed

    # 6개 시나리오 전체에 대해 run을 생성해 하나의 DataFrame으로 합친다
    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        run_frames: list[pd.DataFrame] = []
        run_counter = 0

        for scenario_name in SCENARIOS:
            engine = get_scenario(scenario_name)
            for i in range(self.runs_per_scenario):
                run_counter += 1
                run_id = f"{scenario_name}-{i + 1:04d}"
                df = engine.generate_run(run_id, self.samples_per_run, rng)
                self._attach_context(df, rng, run_counter)
                run_frames.append(df)

        full = pd.concat(run_frames, ignore_index=True)
        return full[META_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS]

    # scenario engine이 만들지 않는 컨텍스트 컬럼(m_no/mode/circuit/timestamp/door_open) 부여
    def _attach_context(self, df: pd.DataFrame, rng: np.random.Generator, run_counter: int) -> None:
        m_no = PANEL_POOL[int(rng.integers(0, len(PANEL_POOL)))]
        circuit = int(rng.integers(1, 11))  # 1~10 (facility 도메인 채널 범위와 동일)

        df["m_no"] = m_no
        df["mode"] = 0
        df["circuit"] = circuit
        # door_open은 이번 6개 시나리오의 핵심 변화 센서가 아니므로 상시 False [SIM] (dataset-spec.md 3절 참고)
        df["door_open"] = False

        run_start = BASE_TIMESTAMP + pd.Timedelta(seconds=run_counter * RUN_TIMESTAMP_GAP_SECONDS)
        df["timestamp"] = [
            run_start + pd.Timedelta(seconds=TIMESTAMP_INTERVAL_SECONDS * j) for j in range(len(df))
        ]
