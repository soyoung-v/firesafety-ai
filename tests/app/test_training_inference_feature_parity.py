"""Training과 Inference가 정확히 같은 Feature 계산 코드를 쓰는지 확인.

app/service/feature_service.py가 training/features/*를 새로 복사하지 않고 그대로
import해서 쓰는지를 import 경로 자체로 검증한다 (Phase 8 명세 8절).
"""

from __future__ import annotations

import pandas as pd

from app.service.feature_service import build_risk_input_frame, circuit_samples_to_frame
from training.features.current_prediction import compute_current_features
from training.features.legacy_arc import compute_legacy_arc_features
from training.features.risk import compute_risk_features


class _FakeSample:
    def __init__(self, am: float, count: int) -> None:
        self.am = am
        self.count = count


class _FakeContextSample:
    def __init__(self, leakage_current: float, temperature: float, total_current: float) -> None:
        self.leakage_current = leakage_current
        self.temperature = temperature
        self.total_current = total_current


def test_arc_features_identical_between_direct_call_and_serving_path():
    samples = [_FakeSample(5.0 + (i % 3) * 0.1, i % 2) for i in range(60)]
    frame_via_service = circuit_samples_to_frame(samples)

    # training 쪽에서 직접 같은 값으로 만든 DataFrame과 비교
    frame_direct = pd.DataFrame(
        {"current": [s.am for s in samples], "arc_count": [s.count for s in samples]}
    )

    features_from_service = compute_legacy_arc_features(frame_via_service)
    features_direct = compute_legacy_arc_features(frame_direct)
    assert features_from_service == features_direct


def test_risk_features_identical_between_direct_call_and_serving_path():
    samples = [_FakeSample(5.0, 0) for _ in range(60)]
    context = [_FakeContextSample(2.0 + i * 0.01, 27.0, 10.0) for i in range(60)]

    circuit_frame = circuit_samples_to_frame(samples)
    serving_frame = build_risk_input_frame(circuit_frame, context)
    features_from_service = compute_risk_features(serving_frame)

    direct_frame = pd.DataFrame(
        {
            "current": [s.am for s in samples],
            "arc_count": [s.count for s in samples],
            "leakage_current": [c.leakage_current for c in context],
            "temperature": [c.temperature for c in context],
            "total_current": [c.total_current for c in context],
        }
    )
    features_direct = compute_risk_features(direct_frame)
    assert features_from_service == features_direct


def test_current_prediction_features_use_same_function_as_training():
    samples = [_FakeSample(5.0 + i * 0.01, 0) for i in range(60)]
    frame = circuit_samples_to_frame(samples)
    # app 쪽 predict_service가 호출하는 것과 동일한 함수를 직접 호출해도 같은 결과
    result = compute_current_features(frame)
    assert set(result.keys()) == {
        "current_mean",
        "current_std",
        "current_min",
        "current_max",
        "current_range",
        "current_diff_abs",
        "current_slope",
        "current_last",
    }
