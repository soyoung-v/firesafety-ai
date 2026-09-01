"""다음 sample의 전류값(target_current)을 window에 붙이는 로직.

target 정의: window 마지막 sample_index 바로 다음 sample의 current 값 (= "next sample current").
실제 센서 전송 주기가 TBD이므로 "1초 뒤"/"1분 뒤"처럼 시간 단위로 해석하지 않는다(dataset-spec.md 6절과
동일 원칙). 다음 sample이 같은 run_id 안에 존재하지 않으면(run의 마지막 window) target을 만들지
않는다 - 서로 다른 run을 이어 붙여 target을 만들지 않는다.
"""

from __future__ import annotations

import pandas as pd


# (run_id, sample_index) -> current 값 lookup 생성 (다음 sample 조회용)
def build_current_lookup(raw_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    return {
        (run_id, sample_index): current
        for run_id, sample_index, current in zip(
            raw_df["run_id"], raw_df["sample_index"], raw_df["current"]
        )
    }


# window 바로 다음 sample의 current 값을 조회. 같은 run에 다음 sample이 없으면 None
def next_sample_current(current_lookup: dict[tuple[str, int], float], run_id: str, window_end: int) -> float | None:
    return current_lookup.get((run_id, window_end + 1))
