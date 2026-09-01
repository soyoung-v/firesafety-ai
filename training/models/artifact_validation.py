"""전체 모델 Artifact 종합 검증.

Phase 2~7에서 만든 4개 모델(arcClassifier/riskClassifier/anomalyDetector/currentRegressor)의
artifact/metadata/inference wrapper가 FastAPI(Phase 8)에서 함께 사용할 준비가 됐는지 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .anomaly_inference import AnomalyDetector
from .arc_inference import ArcClassifier
from .current_inference import CurrentRegressor
from .manifest import MODEL_MANIFEST
from .risk_inference import RiskClassifier

# 모든 metadata가 공통으로 가져야 하는 키 (4개 모델 metadata 실제 비교로 확정)
REQUIRED_METADATA_KEYS = (
    "model_type",
    "created_at",
    "feature_names",
    "window_size",
    "stride",
    "dataset_seed",
    "validation_metrics",
    "test_metrics",
    "sklearn_version",
    "note",
)

_WRAPPER_CLASSES = {
    "arcClassifier": (ArcClassifier, "predict_arc"),
    "riskClassifier": (RiskClassifier, "predict_risk"),
    "anomalyDetector": (AnomalyDetector, "predict"),
    "currentRegressor": (CurrentRegressor, "predict_next_current"),
}


# 모델 하나의 artifact 존재/load/metadata/wrapper 호출을 검증
def _validate_one(model_name: str, entry: dict, artifact_dir: Path) -> dict:
    model_path = artifact_dir / entry["artifact"]
    metadata_path = artifact_dir / entry["metadata"]

    result: dict = {
        "artifact_exists": model_path.exists(),
        "metadata_exists": metadata_path.exists(),
    }
    if not (result["artifact_exists"] and result["metadata_exists"]):
        result["load_ok"] = False
        result["wrapper_ok"] = False
        return result

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    result["missing_metadata_keys"] = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
    result["window_size"] = metadata.get("window_size")
    result["stride"] = metadata.get("stride")
    result["dataset_seed"] = metadata.get("dataset_seed")
    result["sklearn_version"] = metadata.get("sklearn_version")
    result["has_synthetic_note"] = "Synthetic Dataset" in str(metadata.get("note", ""))
    result["feature_count"] = len(metadata.get("feature_names", []))

    wrapper_cls, predict_method_name = _WRAPPER_CLASSES[model_name]
    try:
        wrapper = wrapper_cls.load(model_path)
        result["load_ok"] = True
    except Exception as exc:  # noqa: BLE001 - 검증 스크립트는 실패 사유를 그대로 보고해야 한다
        result["load_ok"] = False
        result["load_error"] = str(exc)
        result["wrapper_ok"] = False
        return result

    try:
        dummy = pd.DataFrame([[0.0] * result["feature_count"]], columns=metadata["feature_names"])
        getattr(wrapper, predict_method_name)(dummy)
        result["wrapper_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["wrapper_ok"] = False
        result["wrapper_error"] = str(exc)

    return result


# 4개 모델 전체를 검증하고, window_size/stride 일관성까지 확인해 결과 dict를 반환
def validate_all_artifacts(artifact_dir: Path) -> dict:
    per_model = {name: _validate_one(name, entry, artifact_dir) for name, entry in MODEL_MANIFEST.items()}

    window_sizes = {r["window_size"] for r in per_model.values() if r.get("window_size") is not None}
    strides = {r["stride"] for r in per_model.values() if r.get("stride") is not None}

    all_ok = all(
        r.get("artifact_exists") and r.get("metadata_exists") and r.get("load_ok") and r.get("wrapper_ok")
        and not r.get("missing_metadata_keys")
        for r in per_model.values()
    )

    return {
        "models": per_model,
        "window_size_consistent": len(window_sizes) <= 1,
        "stride_consistent": len(strides) <= 1,
        "window_sizes_found": sorted(window_sizes),
        "strides_found": sorted(strides),
        "all_models_ready": all_ok,
    }
