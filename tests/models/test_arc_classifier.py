"""Phase 7 통합 테스트 - Legacy ARC Classifier 학습/평가/artifact/추론."""

import numpy as np
import pandas as pd
import pytest

from training.dataset.generator import DatasetGenerator
from training.features.columns import ARC_FEATURE_COLUMNS, FORBIDDEN_FEATURE_COLUMNS
from training.features.pipeline import build_feature_datasets
from training.models.arc_inference import ArcClassifier
from training.models.arc_io import build_arc_metadata, load_arc_artifact, load_arc_metadata, save_arc_artifact
from training.models.arc_metrics import THRESHOLD, compute_arc_metrics
from training.models.candidates import build_candidate_models


@pytest.fixture(scope="module")
def small_arc_features() -> pd.DataFrame:
    raw = DatasetGenerator(runs_per_scenario=6, samples_per_run=180, seed=6).generate()
    arc_df, _ = build_feature_datasets(raw, seed=6)
    return arc_df


def _splits(df):
    return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]


def test_feature_order_matches_legacy_seven():
    assert list(ARC_FEATURE_COLUMNS) == [
        "cnt_std",
        "cnt_mean",
        "cnt_range",
        "cnt_diff_abs",
        "cur_std",
        "cur_mean",
        "cur_range",
    ]


def test_arc_dataset_only_contains_normal_and_arc_scenarios(small_arc_features):
    assert set(small_arc_features["scenario"].unique()).issubset({"NORMAL", "ARC"})


def test_label_is_binary_0_or_1(small_arc_features):
    assert set(small_arc_features["pred"].unique()).issubset({0, 1})


def test_forbidden_columns_not_used_as_training_input():
    for col in FORBIDDEN_FEATURE_COLUMNS:
        assert col not in ARC_FEATURE_COLUMNS


def test_no_run_id_overlap_between_splits(small_arc_features):
    train_df, val_df, test_df = _splits(small_arc_features)
    train_runs, val_runs, test_runs = set(train_df["run_id"]), set(val_df["run_id"]), set(test_df["run_id"])
    assert not (train_runs & val_runs)
    assert not (train_runs & test_runs)
    assert not (val_runs & test_runs)


def test_candidate_models_fit_predict_and_predict_proba(small_arc_features):
    train_df, val_df, _ = _splits(small_arc_features)
    X_train, y_train = train_df[ARC_FEATURE_COLUMNS], train_df["pred"]
    X_val = val_df[ARC_FEATURE_COLUMNS]

    for name, model in build_candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]
        assert set(preds).issubset({0, 1})
        assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_pred_matches_threshold_relationship(small_arc_features):
    train_df, val_df, _ = _splits(small_arc_features)
    X_train, y_train = train_df[ARC_FEATURE_COLUMNS], train_df["pred"]
    X_val = val_df[ARC_FEATURE_COLUMNS]

    model = build_candidate_models()["LogisticRegression"]
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    metrics = compute_arc_metrics(val_df["pred"], proba, THRESHOLD)
    assert metrics["threshold"] == 0.5


def test_reproducible_with_same_seed(small_arc_features):
    train_df, val_df, _ = _splits(small_arc_features)
    X_train, y_train = train_df[ARC_FEATURE_COLUMNS], train_df["pred"]
    X_val = val_df[ARC_FEATURE_COLUMNS]

    model_a = build_candidate_models()["RandomForestClassifier"]
    model_a.fit(X_train, y_train)
    proba_a = model_a.predict_proba(X_val)[:, 1]

    model_b = build_candidate_models()["RandomForestClassifier"]
    model_b.fit(X_train, y_train)
    proba_b = model_b.predict_proba(X_val)[:, 1]

    assert np.allclose(proba_a, proba_b)


def test_artifact_save_load_roundtrip_and_inference_wrapper(small_arc_features, tmp_path):
    train_df, val_df, test_df = _splits(small_arc_features)
    X_train, y_train = train_df[ARC_FEATURE_COLUMNS], train_df["pred"]
    X_val, y_val = val_df[ARC_FEATURE_COLUMNS], val_df["pred"]
    X_test = test_df[ARC_FEATURE_COLUMNS]

    model = build_candidate_models()["LogisticRegression"]
    model.fit(X_train, y_train)
    val_metrics = compute_arc_metrics(y_val, model.predict_proba(X_val)[:, 1], THRESHOLD)

    metadata = build_arc_metadata(
        model_type="LogisticRegression",
        feature_names=list(ARC_FEATURE_COLUMNS),
        labels={"0": "NORMAL", "1": "ARC"},
        threshold=THRESHOLD,
        window_size=60,
        stride=60,
        dataset_seed=6,
        validation_metrics=val_metrics,
        test_metrics=val_metrics,
    )

    model_path = tmp_path / "arc_classifier.joblib"
    save_arc_artifact(model, metadata, model_path)

    before = model.predict_proba(X_test)[:, 1]
    loaded_model = load_arc_artifact(model_path)
    after = loaded_model.predict_proba(X_test)[:, 1]
    assert np.allclose(before, after)

    loaded_metadata = load_arc_metadata(model_path)
    for key in (
        "model_type",
        "created_at",
        "feature_names",
        "labels",
        "threshold",
        "window_size",
        "stride",
        "dataset_seed",
        "validation_metrics",
        "test_metrics",
        "sklearn_version",
        "note",
    ):
        assert key in loaded_metadata
    assert loaded_metadata["threshold"] == 0.5

    classifier = ArcClassifier.load(model_path)
    result = classifier.predict_arc(X_test)
    assert set(result.columns) == {"pred", "proba"}
    assert set(result["pred"].unique()).issubset({0, 1})
    assert result["proba"].between(0, 1).all()
