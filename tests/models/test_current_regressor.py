"""Phase 6 통합 테스트 - Persistence Baseline/ML 회귀 모델 학습/평가/artifact/추론."""

import numpy as np
import pandas as pd
import pytest

from training.dataset.generator import DatasetGenerator
from training.features.current_columns import FEATURE_COLUMNS
from training.features.current_pipeline import build_current_prediction_dataset
from training.models.current_candidates import PersistenceBaseline, build_current_candidate_models
from training.models.current_inference import CurrentRegressor
from training.models.current_io import build_current_metadata, load_current_artifact, load_current_metadata, save_current_artifact
from training.models.current_metrics import (
    compute_mae_improvement_pct,
    compute_regression_metrics,
    regression_metrics_by_change_magnitude,
    regression_metrics_by_group,
)


@pytest.fixture(scope="module")
def small_current_features() -> pd.DataFrame:
    raw = DatasetGenerator(runs_per_scenario=6, samples_per_run=180, seed=5).generate()
    return build_current_prediction_dataset(raw, seed=5, window_size=60, stride=60)


def _splits(df):
    return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]


def test_persistence_baseline_predicts_current_last():
    model = PersistenceBaseline()
    X = pd.DataFrame({"current_last": [1.0, 2.0, 3.0], **{c: [0.0] * 3 for c in FEATURE_COLUMNS if c != "current_last"}})
    model.fit(X)
    preds = model.predict(X)
    assert list(preds) == [1.0, 2.0, 3.0]


def test_compute_regression_metrics_perfect_prediction():
    metrics = compute_regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_mae_improvement_positive_when_model_better():
    improvement = compute_mae_improvement_pct(baseline_mae=1.0, model_mae=0.5)
    assert improvement == 50.0


def test_mae_improvement_negative_when_model_worse():
    improvement = compute_mae_improvement_pct(baseline_mae=0.5, model_mae=1.0)
    assert improvement == -100.0


def test_regression_metrics_by_group():
    y_true = [1.0, 1.0, 5.0, 5.0]
    y_pred = [1.0, 1.0, 4.0, 4.0]
    groups = ["A", "A", "B", "B"]
    result = regression_metrics_by_group(y_true, y_pred, groups)
    assert result["A"]["mae"] == 0.0
    assert result["B"]["mae"] == 1.0


def test_regression_metrics_by_change_magnitude_splits_correctly():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    slopes = [0.0, 0.0, 5.0, 5.0]
    result = regression_metrics_by_change_magnitude(y_true, y_pred, slopes, threshold=1.0)
    assert result["stable"]["n"] == 2
    assert result["changing"]["n"] == 2


def test_all_ml_candidates_fit_and_predict(small_current_features):
    train_df, val_df, _ = _splits(small_current_features)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_current"]
    X_val = val_df[FEATURE_COLUMNS]

    for name, model in build_current_candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert np.isfinite(preds).all()


def test_reproducible_with_same_seed(small_current_features):
    train_df, val_df, _ = _splits(small_current_features)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_current"]
    X_val = val_df[FEATURE_COLUMNS]

    models_a = build_current_candidate_models()
    models_a["HistGradientBoostingRegressor"].fit(X_train, y_train)
    preds_a = models_a["HistGradientBoostingRegressor"].predict(X_val)

    models_b = build_current_candidate_models()
    models_b["HistGradientBoostingRegressor"].fit(X_train, y_train)
    preds_b = models_b["HistGradientBoostingRegressor"].predict(X_val)

    assert np.allclose(preds_a, preds_b)


def test_artifact_save_load_roundtrip_and_inference_wrapper(small_current_features, tmp_path):
    train_df, val_df, test_df = _splits(small_current_features)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_current"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["target_current"]
    X_test = test_df[FEATURE_COLUMNS]

    model = build_current_candidate_models()["HistGradientBoostingRegressor"]
    model.fit(X_train, y_train)
    val_metrics = compute_regression_metrics(y_val, model.predict(X_val))

    metadata = build_current_metadata(
        model_type="HistGradientBoostingRegressor",
        feature_names=list(FEATURE_COLUMNS),
        window_size=60,
        stride=60,
        dataset_seed=5,
        validation_metrics=val_metrics,
        test_metrics=val_metrics,
        baseline_metrics={"validation": val_metrics, "test": val_metrics},
        scenario_metrics={},
    )

    model_path = tmp_path / "current_regressor.joblib"
    save_current_artifact(model, metadata, model_path)

    before = model.predict(X_test)
    loaded_model = load_current_artifact(model_path)
    after = loaded_model.predict(X_test)
    assert np.allclose(before, after)

    loaded_metadata = load_current_metadata(model_path)
    for key in (
        "model_type",
        "created_at",
        "feature_names",
        "target_definition",
        "window_size",
        "stride",
        "dataset_seed",
        "validation_metrics",
        "test_metrics",
        "baseline_metrics",
        "scenario_metrics",
        "sklearn_version",
        "note",
    ):
        assert key in loaded_metadata

    regressor = CurrentRegressor.load(model_path)
    predicted = regressor.predict_next_current(X_test)
    assert predicted.name == "predictedCurrent"
    assert np.allclose(predicted.to_numpy(), before)
