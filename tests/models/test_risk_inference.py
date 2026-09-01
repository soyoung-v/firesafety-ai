"""training/models/risk_inference.py 테스트."""

import numpy as np
import pandas as pd
import pytest

from training.dataset.generator import DatasetGenerator
from training.dataset.columns import RISK_LEVELS
from training.features.columns import RISK_FEATURE_COLUMNS
from training.features.pipeline import build_feature_datasets
from training.models.candidates import build_candidate_models
from training.models.io import build_metadata, save_artifact
from training.models.metrics import compute_classification_metrics
from training.models.risk_inference import RiskClassifier


@pytest.fixture(scope="module")
def small_risk_features() -> pd.DataFrame:
    raw = DatasetGenerator(runs_per_scenario=6, samples_per_run=180, seed=7).generate()
    _, risk_df = build_feature_datasets(raw, seed=7)
    return risk_df


def test_risk_classifier_load_and_predict(small_risk_features, tmp_path):
    train_df = small_risk_features[small_risk_features["split"] == "train"]
    val_df = small_risk_features[small_risk_features["split"] == "val"]
    test_df = small_risk_features[small_risk_features["split"] == "test"]

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
        dataset_seed=7,
        validation_metrics=val_metrics,
        test_metrics=val_metrics,
    )
    model_path = tmp_path / "risk_classifier.joblib"
    save_artifact(model, metadata, model_path)

    classifier = RiskClassifier.load(model_path)
    result = classifier.predict_risk(X_test)

    assert set(result.columns) == {"riskLevel", "riskScore"}
    assert set(result["riskLevel"].unique()).issubset(set(RISK_LEVELS))
    assert result["riskScore"].between(0, 1).all()
    assert len(result) == len(X_test)


def test_risk_score_higher_for_danger_than_normal():
    # DANGER 확률이 높을수록 riskScore가 높아야 한다 (severity 가중치 방향 검증)
    from training.models.risk_inference import SEVERITY_WEIGHTS

    assert SEVERITY_WEIGHTS["NORMAL"] < SEVERITY_WEIGHTS["WARNING"] < SEVERITY_WEIGHTS["DANGER"]
    assert SEVERITY_WEIGHTS["NORMAL"] == 0.0
    assert SEVERITY_WEIGHTS["DANGER"] == 1.0
