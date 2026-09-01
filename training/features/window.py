"""Dataset을 run_id 경계를 넘지 않는 고정 크기 window로 분할한다.

기존 레거시 AI 호환을 위해 기본은 non-overlapping(WINDOW_SIZE=stride=60)이다. 레거시 학습 파이프라인이
쓰던 비중첩 슬라이딩과 동일하게 꼬리에 남는 부족분(window_size 미만)은 버린다(dataset-spec.md 6절).

"60"은 샘플 개수 기준이며 60초라고 가정하지 않는다 - 실제 전송 주기는 TBD(dataset-spec.md 6절).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WINDOW_SIZE = 60
DEFAULT_STRIDE = 60  # non-overlapping 기본값


@dataclass
class Window:
    run_id: str
    scenario: str
    window_index: int
    window_start: int  # run 내 시작 sample_index
    window_end: int  # run 내 끝 sample_index (포함)
    rows: pd.DataFrame  # 원본 row 슬라이스 (길이 = window_size)


# run_id 경계를 넘지 않는 window 목록 생성 (꼬리에 남는 부족분은 버림)
def build_windows(
    df: pd.DataFrame, window_size: int = WINDOW_SIZE, stride: int = DEFAULT_STRIDE
) -> list[Window]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size와 stride는 1 이상이어야 한다")

    windows: list[Window] = []
    for run_id, group in df.groupby("run_id", sort=False):
        ordered = group.sort_values("sample_index").reset_index(drop=True)
        scenario = ordered["scenario"].iloc[0]
        n = len(ordered)

        window_index = 0
        start = 0
        while start + window_size <= n:
            chunk = ordered.iloc[start : start + window_size]
            windows.append(
                Window(
                    run_id=run_id,
                    scenario=scenario,
                    window_index=window_index,
                    window_start=int(chunk["sample_index"].iloc[0]),
                    window_end=int(chunk["sample_index"].iloc[-1]),
                    rows=chunk,
                )
            )
            window_index += 1
            start += stride

    return windows
