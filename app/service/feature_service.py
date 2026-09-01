"""요청 raw sample -> Feature 계산.

`training/features/*`의 Feature 계산 함수를 그대로 재사용한다 - Training Feature Engineering과
Inference Feature Engineering이 다른 코드가 되지 않도록 한다(Phase 8 명세 8절).
"""

from __future__ import annotations

import pandas as pd

MIN_SAMPLES = 30
RECOMMENDED_SAMPLES = 60


# 회로 샘플(am/count)을 legacy_arc.py/risk.py/current_prediction.py가 기대하는 컬럼명으로 변환
def circuit_samples_to_frame(samples) -> pd.DataFrame:
    return pd.DataFrame({"current": [s.am for s in samples], "arc_count": [s.count for s in samples]})


# context 샘플이 risk feature 계산에 쓸 수 있을 만큼 충분한지 확인
# (필수 3개 필드가 전부 채워져 있고, circuit 샘플과 같은 개수여야 같은 시간창으로 취급할 수 있다)
def context_is_sufficient(context_samples, circuit_sample_count: int) -> bool:
    if context_samples is None:
        return False
    if len(context_samples) != circuit_sample_count:
        return False
    if circuit_sample_count < MIN_SAMPLES:
        return False
    return all(
        s.leakage_current is not None and s.temperature is not None and s.total_current is not None
        for s in context_samples
    )


# circuit frame + context를 합쳐 risk.py::compute_risk_features가 기대하는 5개 컬럼 DataFrame 생성
def build_risk_input_frame(circuit_frame: pd.DataFrame, context_samples) -> pd.DataFrame:
    context_frame = pd.DataFrame(
        {
            "leakage_current": [s.leakage_current for s in context_samples],
            "temperature": [s.temperature for s in context_samples],
            "total_current": [s.total_current for s in context_samples],
        }
    )
    return pd.concat(
        [circuit_frame.reset_index(drop=True), context_frame.reset_index(drop=True)], axis=1
    )
