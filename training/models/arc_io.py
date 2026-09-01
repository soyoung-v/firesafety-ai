"""Legacy ARC Classifier artifact(joblib) + metadata(json) 저장/로드."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

METADATA_NOTE = (
    "Synthetic Dataset 기준 평가 결과. 실제 전기화재 아크 탐지 성능이나 안전 인증을 의미하지 않는다."
)


# 모델(joblib)과 metadata(json)를 같은 이름으로 저장
def save_arc_artifact(model, metadata: dict, model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


# 저장된 모델 로드
def load_arc_artifact(model_path: Path):
    return joblib.load(model_path)


# 저장된 metadata 로드
def load_arc_metadata(model_path: Path) -> dict:
    metadata_path = model_path.with_suffix(".metadata.json")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


# artifact와 함께 저장할 metadata 구성
def build_arc_metadata(
    model_type: str,
    feature_names: list[str],
    labels: dict,
    threshold: float,
    window_size: int,
    stride: int,
    dataset_seed: int,
    validation_metrics: dict,
    test_metrics: dict,
) -> dict:
    return {
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "labels": labels,
        "threshold": threshold,
        "window_size": window_size,
        "stride": stride,
        "dataset_seed": dataset_seed,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "sklearn_version": sklearn.__version__,
        "note": METADATA_NOTE,
    }
