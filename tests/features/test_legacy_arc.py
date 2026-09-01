"""training/features/legacy_arc.py 테스트 - 레거시 7개 feature 재현 검증."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.features.legacy_arc import FEATURE_NAMES, compute_legacy_arc_features

# 레거시 참고 구현과의 compatibility test는 선택 사항이다. 로컬 개발 중에만 환경변수로
# 참고 구현 파일 경로를 지정해 실행한다 - 이 저장소에는 그 경로/코드를 포함하지 않는다.
REFERENCE_FEATURES_PATH = os.environ.get("LEGACY_REFERENCE_FEATURES_PATH")


def _small_window(current, arc_count) -> pd.DataFrame:
    return pd.DataFrame({"current": current, "arc_count": arc_count})


def test_feature_order_matches_spec():
    assert FEATURE_NAMES == [
        "cnt_std",
        "cnt_mean",
        "cnt_range",
        "cnt_diff_abs",
        "cur_std",
        "cur_mean",
        "cur_range",
    ]


def test_manual_calculation_matches_pandas_defaults():
    # 손으로 계산 가능한 작은 예시로 ddof/diff/range 동작을 명시적으로 검증한다
    current = [1.0, 2.0, 3.0, 4.0]
    arc_count = [0, 1, 3, 3]
    features = compute_legacy_arc_features(_small_window(current, arc_count))

    cnt = pd.Series(arc_count, dtype=float)
    cur = pd.Series(current)

    assert features["cnt_std"] == pytest.approx(cnt.std())  # ddof=1 (pandas 기본)
    assert features["cnt_mean"] == pytest.approx(1.75)
    assert features["cnt_range"] == pytest.approx(3.0)  # max(3)-min(0)
    assert features["cnt_diff_abs"] == pytest.approx(1.0)  # |1-0|,|2-1|,|0-2| 평균 = (1+2+0)/3

    assert features["cur_std"] == pytest.approx(cur.std())
    assert features["cur_mean"] == pytest.approx(2.5)
    assert features["cur_range"] == pytest.approx(3.0)


def test_no_nan_or_inf_for_full_window():
    rng = np.random.default_rng(1)
    window = _small_window(rng.normal(5, 1, 60), rng.integers(0, 5, 60))
    features = compute_legacy_arc_features(window)
    for value in features.values():
        assert np.isfinite(value)


def test_deterministic_for_same_input():
    window = _small_window([1.0, 2.0, 3.0] * 20, [0, 1, 2] * 20)
    a = compute_legacy_arc_features(window)
    b = compute_legacy_arc_features(window)
    assert a == b


@pytest.mark.skipif(
    not REFERENCE_FEATURES_PATH,
    reason="LEGACY_REFERENCE_FEATURES_PATH 미설정 - 레거시 compatibility test를 건너뜀",
)
def test_matches_legacy_reference_implementation():
    # 환경변수로 지정된 참고 구현 파일을 경로로 직접 로드해 동일 입력에 대한 출력을 비교한다
    # (코드를 이 저장소에 복사하지 않고 참조만 함 - AGENTS.md 절대 규칙)
    reference_path = Path(REFERENCE_FEATURES_PATH)
    reference_src = reference_path.parent
    added = str(reference_src) not in sys.path
    if added:
        sys.path.insert(0, str(reference_src))
    try:
        spec = importlib.util.spec_from_file_location("legacy_features_reference", reference_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference_window_features = module.window_features
    finally:
        if added:
            sys.path.remove(str(reference_src))

    rng = np.random.default_rng(123)
    current = rng.normal(5.0, 1.5, 60)
    arc_count = rng.integers(0, 10, 60).cumsum()

    reference_input = pd.DataFrame({"전류값": current, "카운터": arc_count})
    expected = reference_window_features(reference_input)

    ours_input = _small_window(current, arc_count)
    ours = compute_legacy_arc_features(ours_input)

    for key in FEATURE_NAMES:
        assert ours[key] == pytest.approx(expected[key]), f"{key} 불일치"
