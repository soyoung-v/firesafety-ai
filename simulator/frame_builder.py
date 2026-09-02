"""시나리오 물리값 시계열(training/scenario 재사용) -> /m_noUpload.php 쿼리 파라미터 프레임 시퀀스.

target_circuit 1개만 선택된 시나리오를 따르고, 나머지 9개 회로는 독립된 NORMAL 시나리오로
baseline을 유지한다(동일 seed에서 회로마다 다른 난수 스트림을 쓰도록 seed+channel_no로 분리).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from training.scenario import config as scenario_config
from training.scenario.registry import get_scenario

from . import protocol

MAX_CHANNEL_COUNT = 10


@dataclass
class PanelFrameSeries:
    m_no: str
    target_circuit: int
    samples: int
    target_df: pd.DataFrame
    circuit_dfs: dict[int, pd.DataFrame]  # channel_no -> DataFrame (target 포함, 전체 10개)


# target_circuit은 scenario를 그대로 따르고, 나머지 회로는 독립 NORMAL 시나리오로 생성한다.
def build_panel_frame_series(
    scenario_name: str, m_no: str, samples: int, seed: int, target_circuit: int
) -> PanelFrameSeries:
    if not 1 <= target_circuit <= MAX_CHANNEL_COUNT:
        raise ValueError(f"target_circuit은 1~{MAX_CHANNEL_COUNT} 범위여야 한다: {target_circuit}")

    target_rng = np.random.default_rng(seed)
    target_df = get_scenario(scenario_name).generate_run(
        run_id=f"sim-{m_no}", samples_per_run=samples, rng=target_rng
    )

    circuit_dfs: dict[int, pd.DataFrame] = {target_circuit: target_df}
    for channel_no in range(1, MAX_CHANNEL_COUNT + 1):
        if channel_no == target_circuit:
            continue
        baseline_rng = np.random.default_rng(seed + channel_no)
        circuit_dfs[channel_no] = get_scenario("NORMAL").generate_run(
            run_id=f"sim-{m_no}-c{channel_no}", samples_per_run=samples, rng=baseline_rng
        )

    return PanelFrameSeries(m_no, target_circuit, samples, target_df, circuit_dfs)


# 회로별 이번 프레임에서 ARC bit를 켤지 판정 - 누적 arc_count가 직전 샘플 대비 증가했으면 아크 발생으로 본다
def _arc_channels_at(series: PanelFrameSeries, sample_index: int) -> set[int]:
    arc_channels: set[int] = set()
    for channel_no, df in series.circuit_dfs.items():
        current_total = int(df["arc_count"].iloc[sample_index])
        previous_total = int(df["arc_count"].iloc[sample_index - 1]) if sample_index > 0 else 0
        if current_total > previous_total:
            arc_channels.add(channel_no)
    return arc_channels


# 분전반 공통 센서값(target_df 기준)이 확인된 [HW]/[BE] 임계값을 넘으면 해당 ALARM bit를 켠다.
# GAS/FIRE/HUMIDITY/DOOR는 임계값이 TBD이거나 시나리오가 다루지 않아 항상 끈다.
def _alarm_bits_at(row: pd.Series) -> set[int]:
    alarm_bits: set[int] = set()
    if row["leakage_current"] > scenario_config.LEAK_MA_THRESHOLD:
        alarm_bits.add(protocol.ALARM_BIT_LEAKAGE)
    if row["temperature"] > scenario_config.TEMP_THRESHOLD_C:
        alarm_bits.add(protocol.ALARM_BIT_OVERHEAT)
    if row["total_current"] > scenario_config.OVERCURRENT_THRESHOLD_A:
        alarm_bits.add(protocol.ALARM_BIT_OVERCURRENT)
    return alarm_bits


# sample_index 하나에 대한 /m_noUpload.php 쿼리 파라미터 dict 생성
def build_frame_params(series: PanelFrameSeries, sample_index: int) -> dict[str, str]:
    target_row = series.target_df.iloc[sample_index]

    params: dict[str, str] = {
        "m_no": series.m_no,
        "mode": "0",
        "volt": protocol.encode_voltage(target_row["voltage"]),
        "hct_count": protocol.pad(sample_index % 10000, 4),
        "s_circuit": protocol.encode_leak_ma(target_row["leakage_current"]),
        "tem": protocol.encode_temperature(target_row["temperature"]),
        "humi": protocol.encode_humidity(target_row["humidity"]),
        "fire": protocol.encode_fire_raw(target_row["fire_raw"]),
        "gas": protocol.encode_gas_raw(target_row["gas_raw"]),
        "door": protocol.encode_door(False),
        "total_circuit": protocol.encode_total_current(target_row["total_current"]),
        "e_energy": protocol.encode_total_power(target_row["total_power"]),
    }

    for channel_no, df in series.circuit_dfs.items():
        circuit_row = df.iloc[sample_index]
        params[f"am{channel_no}"] = protocol.encode_current(circuit_row["current"])
        params[f"count{channel_no}"] = protocol.encode_arc_counter(circuit_row["arc_count"])

    arc_channels = _arc_channels_at(series, sample_index)
    alarm_bits = _alarm_bits_at(target_row)
    params["aerror"] = protocol.build_aerror(arc_channels, alarm_bits)

    return params
