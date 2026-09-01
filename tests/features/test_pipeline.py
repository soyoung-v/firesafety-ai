"""training/features/pipeline.py 통합 테스트. 작은 규모 raw Dataset으로 검증한다."""

import numpy as np
import pandas as pd
import pytest

from training.dataset.generator import DatasetGenerator
from training.features.columns import (
    ARC_FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    RISK_FEATURE_COLUMNS,
)
from training.features.pipeline import build_feature_datasets


def _small_raw_df(seed: int = 1) -> pd.DataFrame:
    # 시나리오당 3 run, run당 180 sample(=window 3개) - 대용량 생성 없이 파이프라인 전체 검증
    return DatasetGenerator(runs_per_scenario=3, samples_per_run=180, seed=seed).generate()


def test_arc_dataset_only_contains_normal_and_arc_scenarios():
    arc_df, _ = build_feature_datasets(_small_raw_df(), seed=1)
    assert set(arc_df["scenario"].unique()).issubset({"NORMAL", "ARC"})


def test_risk_dataset_contains_all_six_scenarios():
    _, risk_df = build_feature_datasets(_small_raw_df(), seed=1)
    assert set(risk_df["scenario"].unique()) == {
        "NORMAL",
        "OVER_CURRENT",
        "ARC",
        "LEAKAGE",
        "OVERHEATING",
        "COMPLEX_RISK",
    }


def test_risk_dataset_has_all_three_risk_levels():
    _, risk_df = build_feature_datasets(_small_raw_df(), seed=1)
    assert set(risk_df["risk_level"].unique()) == {"NORMAL", "WARNING", "DANGER"}


def test_forbidden_columns_never_appear_in_feature_columns():
    for col in FORBIDDEN_FEATURE_COLUMNS:
        assert col not in ARC_FEATURE_COLUMNS
        assert col not in RISK_FEATURE_COLUMNS


def test_metadata_and_label_are_separate_from_feature_matrix():
    arc_df, risk_df = build_feature_datasets(_small_raw_df(), seed=1)
    # Feature Matrix로 쓸 컬럼 목록만 뽑아도 leakage 컬럼이 섞이지 않아야 한다
    arc_matrix = arc_df[ARC_FEATURE_COLUMNS]
    risk_matrix = risk_df[RISK_FEATURE_COLUMNS]
    for col in FORBIDDEN_FEATURE_COLUMNS:
        assert col not in arc_matrix.columns
        assert col not in risk_matrix.columns


def test_run_id_never_split_across_train_val_test():
    _, risk_df = build_feature_datasets(_small_raw_df(), seed=1)
    per_run_splits = risk_df.groupby("run_id")["split"].nunique()
    assert (per_run_splits == 1).all()


def test_split_values_are_valid():
    _, risk_df = build_feature_datasets(_small_raw_df(), seed=1)
    assert set(risk_df["split"].unique()).issubset({"train", "val", "test"})


def test_no_nan_or_inf_in_feature_columns():
    arc_df, risk_df = build_feature_datasets(_small_raw_df(), seed=1)
    assert not arc_df[ARC_FEATURE_COLUMNS].isna().any().any()
    assert not risk_df[RISK_FEATURE_COLUMNS].isna().any().any()
    assert np.isfinite(arc_df[ARC_FEATURE_COLUMNS].to_numpy(dtype=float)).all()
    assert np.isfinite(risk_df[RISK_FEATURE_COLUMNS].to_numpy(dtype=float)).all()


def test_deterministic_for_same_raw_dataset_and_seed():
    raw = _small_raw_df(seed=5)
    arc_a, risk_a = build_feature_datasets(raw, seed=42)
    arc_b, risk_b = build_feature_datasets(raw, seed=42)
    assert arc_a.equals(arc_b)
    assert risk_a.equals(risk_b)


def test_window_count_matches_expected():
    # run당 180 sample, window=60, stride=60 -> run당 정확히 3 window
    raw = _small_raw_df(seed=1)
    _, risk_df = build_feature_datasets(raw, seed=1, window_size=60, stride=60)
    windows_per_run = risk_df.groupby("run_id").size()
    assert (windows_per_run == 3).all()


def test_arc_windows_use_max_severity_label_consistently():
    arc_df, _ = build_feature_datasets(_small_raw_df(seed=2), seed=2)
    arc_only = arc_df[arc_df["scenario"] == "ARC"]
    assert set(arc_only["pred"].unique()).issubset({0, 1})
    assert arc_only["pred"].sum() > 0  # ARC run에는 반드시 WARNING/DANGER 구간이 있어야 함


@pytest.mark.parametrize("window_size,stride", [(30, 30), (60, 60)])
def test_custom_window_size_and_stride(window_size, stride):
    raw = _small_raw_df(seed=3)
    arc_df, risk_df = build_feature_datasets(raw, seed=3, window_size=window_size, stride=stride)
    assert len(risk_df) > 0
    assert len(arc_df) > 0
