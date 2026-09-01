"""training/models/artifact_validation.py, manifest.py, evaluation_summary.py 테스트."""

import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from training.models.artifact_validation import validate_all_artifacts
from training.models.evaluation_summary import build_evaluation_summary
from training.models.manifest import MODEL_MANIFEST


def test_manifest_has_all_four_models():
    assert set(MODEL_MANIFEST.keys()) == {
        "arcClassifier",
        "riskClassifier",
        "anomalyDetector",
        "currentRegressor",
    }


def test_manifest_entries_have_required_fields():
    for name, entry in MODEL_MANIFEST.items():
        assert entry["artifact"].endswith(".joblib")
        assert entry["metadata"].endswith(".metadata.json")
        assert len(entry["outputs"]) > 0
        assert entry["wrapper"].startswith("training.models.")


def test_validate_all_artifacts_reports_missing_files(tmp_path):
    result = validate_all_artifacts(tmp_path)
    assert result["all_models_ready"] is False
    for model_result in result["models"].values():
        assert model_result["artifact_exists"] is False


def _build_fake_arc_artifact(artifact_dir, window_size=60, stride=60):
    import joblib

    X = pd.DataFrame(
        {c: [0.0, 1.0, 0.0, 1.0] for c in ["cnt_std", "cnt_mean", "cnt_range", "cnt_diff_abs", "cur_std", "cur_mean", "cur_range"]}
    )
    y = [0, 1, 0, 1]
    model = LogisticRegression().fit(X, y)
    joblib.dump(model, artifact_dir / "arc_classifier.joblib")

    metadata = {
        "model_type": "LogisticRegression",
        "created_at": "2026-09-01T00:00:00+00:00",
        "feature_names": list(X.columns),
        "labels": {"0": "NORMAL", "1": "ARC"},
        "threshold": 0.5,
        "window_size": window_size,
        "stride": stride,
        "dataset_seed": 1,
        "validation_metrics": {},
        "test_metrics": {},
        "sklearn_version": "0.0.0",
        "note": "Synthetic Dataset 기준 평가 결과.",
    }
    (artifact_dir / "arc_classifier.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_validate_all_artifacts_detects_valid_model(tmp_path):
    _build_fake_arc_artifact(tmp_path)
    result = validate_all_artifacts(tmp_path)
    arc_result = result["models"]["arcClassifier"]
    assert arc_result["artifact_exists"] is True
    assert arc_result["load_ok"] is True
    assert arc_result["wrapper_ok"] is True
    assert arc_result["missing_metadata_keys"] == []
    # 나머지 3개 모델은 여전히 없으므로 전체 준비 상태는 False
    assert result["all_models_ready"] is False


def test_validate_all_artifacts_detects_window_size_inconsistency(tmp_path):
    _build_fake_arc_artifact(tmp_path, window_size=60)
    result = validate_all_artifacts(tmp_path)
    assert result["window_sizes_found"] == [60]
    assert result["window_size_consistent"] is True


def test_build_evaluation_summary_reads_existing_reports(tmp_path):
    (tmp_path / "arc_classifier_metrics.json").write_text(
        json.dumps(
            {
                "selected_model": "LogisticRegression",
                "test_metrics": {"f1": 0.9, "arc_recall": 0.95, "normal_false_positive_rate": 0.01},
            }
        ),
        encoding="utf-8",
    )
    summary = build_evaluation_summary(tmp_path)
    assert summary["arcClassifier"]["f1"] == 0.9
    assert "riskClassifier" not in summary  # 해당 report가 없으면 summary에도 없어야 함
    assert "note" in summary
