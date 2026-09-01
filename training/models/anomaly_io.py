"""Anomaly Detector artifact(joblib) + metadata(json) 저장/로드."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

METADATA_NOTE = (
    "Synthetic Dataset 기준 평가 결과. 실제 전기화재 이상 탐지 성능이나 안전 인증을 의미하지 않는다."
)


# 모델(joblib)과 metadata(json)를 같은 이름으로 저장
def save_anomaly_artifact(model, metadata: dict, model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


# 저장된 모델 로드
def load_anomaly_artifact(model_path: Path):
    return joblib.load(model_path)


# 저장된 metadata 로드
def load_anomaly_metadata(model_path: Path) -> dict:
    metadata_path = model_path.with_suffix(".metadata.json")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


# artifact와 함께 저장할 metadata 구성
def build_anomaly_metadata(
    model_type: str,
    feature_names: list[str],
    normal_training_definition: str,
    window_size: int,
    stride: int,
    dataset_seed: int,
    score_min: float,
    score_max: float,
    threshold: float,
    validation_metrics: dict,
    test_metrics: dict,
) -> dict:
    return {
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "normal_training_definition": normal_training_definition,
        "window_size": window_size,
        "stride": stride,
        "dataset_seed": dataset_seed,
        "score_transform": {
            "raw_score": "-model.score_samples(X) (높을수록 비정상이 되도록 부호 반전)",
            "normalization": "train(정상) 데이터 raw score의 min/max로 0~1 스케일링 후 clip",
            "train_score_min": score_min,
            "train_score_max": score_max,
        },
        "anomaly_threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "sklearn_version": sklearn.__version__,
        "note": METADATA_NOTE,
    }
