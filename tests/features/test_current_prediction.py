"""training/features/current_*.py 테스트 - target 생성 정확성/leakage 방지가 핵심."""

import pandas as pd
import pytest

from training.dataset.generator import DatasetGenerator
from training.features.current_columns import ALL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from training.features.current_pipeline import build_current_prediction_dataset
from training.features.current_prediction import compute_current_features
from training.features.current_target import build_current_lookup, next_sample_current


def _make_run(run_id: str, n_samples: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": run_id,
            "scenario": "NORMAL",
            "sample_index": range(n_samples),
            "current": [float(i) for i in range(n_samples)],  # 값 자체가 sample_index와 같아 검증이 쉬움
            "arc_count": 0,
        }
    )


def test_compute_current_features_basic():
    window = pd.DataFrame({"current": [1.0, 2.0, 3.0, 4.0]})
    features = compute_current_features(window)
    assert features["current_mean"] == 2.5
    assert features["current_min"] == 1.0
    assert features["current_max"] == 4.0
    assert features["current_range"] == 3.0
    assert features["current_last"] == 4.0


def test_next_sample_current_exists():
    df = _make_run("R-1", 65)
    lookup = build_current_lookup(df)
    # window 0..59 -> 다음 sample_index=60의 current 값은 60.0
    assert next_sample_current(lookup, "R-1", window_end=59) == 60.0


def test_next_sample_current_missing_at_run_end():
    df = _make_run("R-1", 60)  # sample_index 0..59만 존재
    lookup = build_current_lookup(df)
    assert next_sample_current(lookup, "R-1", window_end=59) is None


def test_target_generation_excludes_last_window_of_each_run():
    # window=60, stride=60, 샘플 360개(정확히 6 window) -> 6번째 window(299..359 아님, 0-index 5:
    # start=300..359)는 다음 sample(360)이 없어 제외되어야 함 -> 5개 window만 남는다
    df = _make_run("R-1", 360)
    result = build_current_prediction_dataset(df, seed=1, window_size=60, stride=60)
    assert len(result) == 5
    assert result["window_index"].max() == 4


def test_target_never_crosses_run_boundary():
    df = pd.concat([_make_run("A", 120), _make_run("B", 120)], ignore_index=True)
    result = build_current_prediction_dataset(df, seed=1, window_size=60, stride=60)
    # 각 run은 120 샘플 -> window 2개(0..59, 60..119), 마지막 window(60..119)는 다음 sample(120)이
    # 그 run에 없으므로 제외 -> run당 1개(첫 window)만 남아야 한다
    assert len(result) == 2
    assert set(result["run_id"]) == {"A", "B"}
    for _, row in result.iterrows():
        assert row["window_index"] == 0


def test_target_value_matches_next_sample_not_window_content():
    df = _make_run("R-1", 130)
    result = build_current_prediction_dataset(df, seed=1, window_size=60, stride=60)
    first_window_target = result.iloc[0]["target_current"]
    # window 0: sample_index 0..59, current==sample_index이므로 window 안의 값은 최대 59
    # target은 window 밖의 sample_index=60 값(60.0)이어야 한다 - window 안 값이 아님을 확인
    assert first_window_target == 60.0


def test_target_column_not_in_feature_columns():
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_forbidden_metadata_not_in_feature_columns():
    for forbidden in ("scenario", "run_id", "split", "risk_level", "aerror", "device_arc_flag"):
        assert forbidden not in FEATURE_COLUMNS


def test_full_pipeline_with_real_scenarios_has_no_target_leakage():
    raw = DatasetGenerator(runs_per_scenario=2, samples_per_run=180, seed=3).generate()
    result = build_current_prediction_dataset(raw, seed=3, window_size=60, stride=60)

    assert list(result.columns) == ALL_COLUMNS
    assert result[TARGET_COLUMN].notna().all()

    # run당 180 샘플, window=60 -> 3 window, 마지막 window는 target이 없어 제외 -> run당 2개
    windows_per_run = result.groupby("run_id").size()
    assert (windows_per_run == 2).all()


def test_split_preserved_no_run_id_overlap():
    raw = DatasetGenerator(runs_per_scenario=2, samples_per_run=180, seed=4).generate()
    result = build_current_prediction_dataset(raw, seed=4, window_size=60, stride=60)
    per_run_splits = result.groupby("run_id")["split"].nunique()
    assert (per_run_splits == 1).all()
