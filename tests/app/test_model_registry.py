"""app/service/model_registry.py 테스트 - 로딩 실패/manifest 오류 처리."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.service.model_registry import ModelLoadError, ModelRegistry


def test_load_all_succeeds_with_real_artifacts():
    registry = ModelRegistry()
    registry.load_all(Path("artifacts"))
    assert registry.get("arcClassifier") is not None
    assert registry.status()["arcClassifier"] is True


def test_missing_manifest_raises(tmp_path):
    registry = ModelRegistry()
    with pytest.raises(FileNotFoundError):
        registry.load_all(tmp_path)


def test_missing_required_model_raises_model_load_error(tmp_path):
    # manifest는 있지만 arcClassifier artifact 파일이 없는 경우
    manifest = {
        "arcClassifier": {"artifact": "arc_classifier.joblib", "metadata": "arc_classifier.metadata.json"},
        "riskClassifier": {"artifact": "risk_classifier.joblib", "metadata": "risk_classifier.metadata.json"},
        "anomalyDetector": {"artifact": "anomaly_detector.joblib", "metadata": "anomaly_detector.metadata.json"},
        "currentRegressor": {"artifact": "current_regressor.joblib", "metadata": "current_regressor.metadata.json"},
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    registry = ModelRegistry()
    with pytest.raises(ModelLoadError):
        registry.load_all(tmp_path)


def test_public_info_hides_local_paths():
    registry = ModelRegistry()
    registry.load_all(Path("artifacts"))
    info = registry.public_info()
    info_text = json.dumps(info)
    assert "/Users/" not in info_text
    assert ".joblib" not in info_text
