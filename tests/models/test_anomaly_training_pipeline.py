"""Phase 5 통합 테스트 - 작은 규모 Feature Dataset으로 학습/평가/artifact 저장까지 전체 흐름 검증."""

import numpy as np
import pandas as pd
import pytest

from training.dataset.generator import DatasetGenerator
from training.features.columns import FORBIDDEN_FEATURE_COLUMNS, RISK_FEATURE_COLUMNS
from training.features.pipeline import build_feature_datasets
from training.models.anomaly_candidates import build_anomaly_candidate_models
from training.models.anomaly_io import (
    build_anomaly_metadata,
    load_anomaly_artifact,
    load_anomaly_metadata,
    save_anomaly_artifact,
)
from training.models.anomaly_metrics import (
    compute_anomaly_metrics,
    fit_score_normalizer,
    normalize_score,
    raw_to_anomaly_score,
    select_threshold,
)
from training.models.anomaly_inference import AnomalyDetector


@pytest.fixture(scope="module")
def small_risk_features() -> pd.DataFrame:
    raw = DatasetGenerator(runs_per_scenario=6, samples_per_run=180, seed=2).generate()
    _, risk_df = build_feature_datasets(raw, seed=2)
    return risk_df


def _splits(df):
    return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]


def test_normal_training_data_excludes_warning_and_danger(small_risk_features):
    train_df, _, _ = _splits(small_risk_features)
    normal_train = train_df[train_df["risk_level"] == "NORMAL"]
    assert len(normal_train) > 0
    assert set(normal_train["risk_level"].unique()) == {"NORMAL"}
    # ADR-004: scenario와 무관하게 risk_level==NORMAL이면 포함되어야 함
    assert set(normal_train["scenario"].unique()) == set(train_df["scenario"].unique())


def test_forbidden_columns_not_used_as_training_input():
    for col in FORBIDDEN_FEATURE_COLUMNS:
        assert col not in RISK_FEATURE_COLUMNS
    assert "scenario" not in RISK_FEATURE_COLUMNS
    assert "risk_level" not in RISK_FEATURE_COLUMNS


def test_no_run_id_overlap_between_splits(small_risk_features):
    train_df, val_df, test_df = _splits(small_risk_features)
    train_runs = set(train_df["run_id"])
    val_runs = set(val_df["run_id"])
    test_runs = set(test_df["run_id"])
    assert not (train_runs & val_runs)
    assert not (train_runs & test_runs)
    assert not (val_runs & test_runs)


def test_candidate_models_fit_on_normal_only_and_score(small_risk_features):
    train_df, val_df, _ = _splits(small_risk_features)
    X_train = train_df[train_df["risk_level"] == "NORMAL"][RISK_FEATURE_COLUMNS]
    X_val = val_df[RISK_FEATURE_COLUMNS]

    for name, model in build_anomaly_candidate_models().items():
        model.fit(X_train)
        scores = raw_to_anomaly_score(model, X_val)
        assert len(scores) == len(X_val)
        assert np.isfinite(scores).all()


def test_threshold_produces_boolean_predictions(small_risk_features):
    train_df, val_df, _ = _splits(small_risk_features)
    X_train = train_df[train_df["risk_level"] == "NORMAL"][RISK_FEATURE_COLUMNS]
    X_val, val_risk_levels = val_df[RISK_FEATURE_COLUMNS], val_df["risk_level"]

    model = build_anomaly_candidate_models()["IsolationForest"]
    model.fit(X_train)
    score_min, score_max = fit_score_normalizer(model, X_train)
    val_scores = normalize_score(raw_to_anomaly_score(model, X_val), score_min, score_max)
    threshold_info = select_threshold(val_scores, val_risk_levels)
    metrics = compute_anomaly_metrics(val_scores, val_risk_levels, threshold_info["threshold"])

    assert 0.0 <= threshold_info["threshold"] <= 1.0
    assert metrics["danger_recall"] is not None


def test_reproducible_with_same_seed(small_risk_features):
    train_df, val_df, _ = _splits(small_risk_features)
    X_train = train_df[train_df["risk_level"] == "NORMAL"][RISK_FEATURE_COLUMNS]
    X_val = val_df[RISK_FEATURE_COLUMNS]

    model_a = build_anomaly_candidate_models()["IsolationForest"]
    model_a.fit(X_train)
    scores_a = raw_to_anomaly_score(model_a, X_val)

    model_b = build_anomaly_candidate_models()["IsolationForest"]
    model_b.fit(X_train)
    scores_b = raw_to_anomaly_score(model_b, X_val)

    assert np.allclose(scores_a, scores_b)


def test_artifact_save_and_load_roundtrip_gives_same_scores(small_risk_features, tmp_path):
    train_df, val_df, test_df = _splits(small_risk_features)
    X_train = train_df[train_df["risk_level"] == "NORMAL"][RISK_FEATURE_COLUMNS]
    X_val, val_risk_levels = val_df[RISK_FEATURE_COLUMNS], val_df["risk_level"]
    X_test = test_df[RISK_FEATURE_COLUMNS]

    model = build_anomaly_candidate_models()["IsolationForest"]
    model.fit(X_train)
    score_min, score_max = fit_score_normalizer(model, X_train)
    val_scores = normalize_score(raw_to_anomaly_score(model, X_val), score_min, score_max)
    threshold_info = select_threshold(val_scores, val_risk_levels)
    val_metrics = compute_anomaly_metrics(val_scores, val_risk_levels, threshold_info["threshold"])

    metadata = build_anomaly_metadata(
        model_type="IsolationForest",
        feature_names=list(RISK_FEATURE_COLUMNS),
        normal_training_definition="risk_level == 'NORMAL' (test fixture)",
        window_size=60,
        stride=60,
        dataset_seed=2,
        score_min=score_min,
        score_max=score_max,
        threshold=threshold_info["threshold"],
        validation_metrics=val_metrics,
        test_metrics=val_metrics,
    )

    model_path = tmp_path / "anomaly_detector.joblib"
    save_anomaly_artifact(model, metadata, model_path)

    before = raw_to_anomaly_score(model, X_test)
    loaded_model = load_anomaly_artifact(model_path)
    after = raw_to_anomaly_score(loaded_model, X_test)
    assert np.allclose(before, after)

    loaded_metadata = load_anomaly_metadata(model_path)
    for key in (
        "model_type",
        "created_at",
        "feature_names",
        "normal_training_definition",
        "window_size",
        "stride",
        "dataset_seed",
        "score_transform",
        "anomaly_threshold",
        "validation_metrics",
        "test_metrics",
        "sklearn_version",
        "note",
    ):
        assert key in loaded_metadata


def test_anomaly_detector_inference_wrapper(small_risk_features, tmp_path):
    train_df, val_df, test_df = _splits(small_risk_features)
    X_train = train_df[train_df["risk_level"] == "NORMAL"][RISK_FEATURE_COLUMNS]
    X_val, val_risk_levels = val_df[RISK_FEATURE_COLUMNS], val_df["risk_level"]
    X_test = test_df[RISK_FEATURE_COLUMNS]

    model = build_anomaly_candidate_models()["IsolationForest"]
    model.fit(X_train)
    score_min, score_max = fit_score_normalizer(model, X_train)
    val_scores = normalize_score(raw_to_anomaly_score(model, X_val), score_min, score_max)
    threshold_info = select_threshold(val_scores, val_risk_levels)
    val_metrics = compute_anomaly_metrics(val_scores, val_risk_levels, threshold_info["threshold"])

    metadata = build_anomaly_metadata(
        model_type="IsolationForest",
        feature_names=list(RISK_FEATURE_COLUMNS),
        normal_training_definition="risk_level == 'NORMAL' (test fixture)",
        window_size=60,
        stride=60,
        dataset_seed=2,
        score_min=score_min,
        score_max=score_max,
        threshold=threshold_info["threshold"],
        validation_metrics=val_metrics,
        test_metrics=val_metrics,
    )
    model_path = tmp_path / "anomaly_detector.joblib"
    save_anomaly_artifact(model, metadata, model_path)

    detector = AnomalyDetector.load(model_path)
    result = detector.predict(X_test)

    assert set(result.columns) == {"raw_score", "anomalyScore", "anomaly"}
    assert result["anomalyScore"].between(0, 1).all()
    assert result["anomaly"].dtype == bool
    assert len(result) == len(X_test)
