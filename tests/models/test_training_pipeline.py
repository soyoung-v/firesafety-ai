"""Phase 4 통합 테스트 - 작은 규모 Feature Dataset으로 학습/평가/artifact 저장까지 전체 흐름 검증."""

import numpy as np
import pandas as pd
import pytest

from training.dataset.columns import RISK_LEVELS
from training.dataset.generator import DatasetGenerator
from training.features.columns import FORBIDDEN_FEATURE_COLUMNS, RISK_FEATURE_COLUMNS
from training.features.pipeline import build_feature_datasets
from training.models.candidates import build_candidate_models
from training.models.importance import compute_feature_importance
from training.models.io import build_metadata, load_artifact, load_metadata, save_artifact
from training.models.metrics import compute_classification_metrics


@pytest.fixture(scope="module")
def small_risk_features() -> pd.DataFrame:
    # 시나리오당 6 run, run당 180 sample(window 3개) - 대용량 학습 없이 학습 가능성만 검증
    raw = DatasetGenerator(runs_per_scenario=6, samples_per_run=180, seed=1).generate()
    _, risk_df = build_feature_datasets(raw, seed=1)
    return risk_df


def _splits(df):
    return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]


def test_feature_order_matches_columns_module(small_risk_features):
    assert list(RISK_FEATURE_COLUMNS) == list(small_risk_features[RISK_FEATURE_COLUMNS].columns)


def test_label_values_match_risk_levels(small_risk_features):
    assert set(small_risk_features["risk_level"].unique()).issubset(set(RISK_LEVELS))


def test_train_val_test_split_preserved_not_resplit(small_risk_features):
    train_df, val_df, test_df = _splits(small_risk_features)
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    # run_id가 한 split에만 속하는지 (Phase 3에서 이미 보장되지만 학습 직전에도 재확인)
    combined = pd.concat([train_df, val_df, test_df])
    assert combined.groupby("run_id")["split"].nunique().eq(1).all()


def test_forbidden_columns_not_used_as_training_input(small_risk_features):
    for col in FORBIDDEN_FEATURE_COLUMNS:
        assert col not in RISK_FEATURE_COLUMNS


def test_all_candidate_models_fit_and_predict(small_risk_features):
    train_df, val_df, _ = _splits(small_risk_features)
    X_train, y_train = train_df[RISK_FEATURE_COLUMNS], train_df["risk_level"]
    X_val = val_df[RISK_FEATURE_COLUMNS]

    for name, model in build_candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert set(preds).issubset(set(RISK_LEVELS))


def test_all_candidate_models_support_predict_proba(small_risk_features):
    train_df, val_df, _ = _splits(small_risk_features)
    X_train, y_train = train_df[RISK_FEATURE_COLUMNS], train_df["risk_level"]
    X_val = val_df[RISK_FEATURE_COLUMNS]

    for name, model in build_candidate_models().items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), len(RISK_LEVELS))
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_feature_importance_returns_all_features(small_risk_features):
    train_df, val_df, _ = _splits(small_risk_features)
    X_train, y_train = train_df[RISK_FEATURE_COLUMNS], train_df["risk_level"]
    X_val, y_val = val_df[RISK_FEATURE_COLUMNS], val_df["risk_level"]

    model = build_candidate_models()["RandomForestClassifier"]
    model.fit(X_train, y_train)
    importance = compute_feature_importance(model, X_val, y_val, list(RISK_FEATURE_COLUMNS))
    assert {item["feature"] for item in importance} == set(RISK_FEATURE_COLUMNS)


def test_artifact_save_and_load_roundtrip_gives_same_predictions(small_risk_features, tmp_path):
    train_df, val_df, test_df = _splits(small_risk_features)
    X_train, y_train = train_df[RISK_FEATURE_COLUMNS], train_df["risk_level"]
    X_val, y_val = val_df[RISK_FEATURE_COLUMNS], val_df["risk_level"]
    X_test = test_df[RISK_FEATURE_COLUMNS]

    model = build_candidate_models()["RandomForestClassifier"]
    model.fit(X_train, y_train)
    val_metrics = compute_classification_metrics(y_val, model.predict(X_val))

    metadata = build_metadata(
        model_type="RandomForestClassifier",
        feature_names=list(RISK_FEATURE_COLUMNS),
        label_names=RISK_LEVELS,
        window_size=60,
        stride=60,
        dataset_seed=1,
        validation_metrics=val_metrics,
        test_metrics=val_metrics,
    )

    model_path = tmp_path / "risk_classifier.joblib"
    save_artifact(model, metadata, model_path)

    before = model.predict(X_test)
    loaded_model = load_artifact(model_path)
    after = loaded_model.predict(X_test)
    assert list(before) == list(after)

    loaded_metadata = load_metadata(model_path)
    for key in (
        "model_type",
        "created_at",
        "feature_names",
        "label_names",
        "window_size",
        "stride",
        "dataset_seed",
        "validation_metrics",
        "test_metrics",
        "sklearn_version",
        "note",
    ):
        assert key in loaded_metadata
    assert loaded_metadata["feature_names"] == list(RISK_FEATURE_COLUMNS)
