"""FastAPI startup 시 model_manifest.json 기준으로 4개 모델을 1회 로드해 재사용한다.

요청마다 joblib 파일을 다시 읽지 않는다. arcClassifier는 기존 Spring Boot 계약(pred/proba)의
필수 모델이라 로드 실패 시 서비스 시작 자체를 막는다. 나머지 3개(확장 모델)는 로드 실패해도
서비스는 계속 뜨되 해당 필드를 null로 반환한다(/health가 DEGRADED로 표시).
"""

from __future__ import annotations

import json
from pathlib import Path

from training.models.anomaly_inference import AnomalyDetector
from training.models.arc_inference import ArcClassifier
from training.models.current_inference import CurrentRegressor
from training.models.risk_inference import RiskClassifier

DEFAULT_ARTIFACT_DIR = Path("artifacts")
REQUIRED_MODEL = "arcClassifier"

_WRAPPER_CLASSES = {
    "arcClassifier": ArcClassifier,
    "riskClassifier": RiskClassifier,
    "anomalyDetector": AnomalyDetector,
    "currentRegressor": CurrentRegressor,
}


class ModelLoadError(RuntimeError):
    """필수 모델(arcClassifier) 로드 실패 - 서비스 시작을 막는다."""


class ModelRegistry:
    def __init__(self) -> None:
        self._wrappers: dict[str, object] = {}
        self._metadata: dict[str, dict] = {}
        self._load_errors: dict[str, str] = {}

    # manifest를 읽어 4개 모델을 전부 로드 시도 (필수 모델 실패 시 예외 발생)
    def load_all(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
        manifest_path = artifact_dir / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for name, entry in manifest.items():
            model_path = artifact_dir / entry["artifact"]
            metadata_path = artifact_dir / entry["metadata"]
            try:
                self._wrappers[name] = _WRAPPER_CLASSES[name].load(model_path)
                self._metadata[name] = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - 모델별 로딩 실패를 기록하고 계속 진행
                self._load_errors[name] = str(exc)

        if REQUIRED_MODEL not in self._wrappers:
            raise ModelLoadError(
                f"{REQUIRED_MODEL} 로드 실패 - 기존 Spring Boot 계약(pred/proba)의 필수 모델이라 서비스를 시작할 수 없습니다"
            )

    def get(self, name: str):
        return self._wrappers.get(name)

    def metadata(self, name: str) -> dict | None:
        return self._metadata.get(name)

    # /health용 - 모델별 로드 여부만 반환 (경로/에러 상세는 노출하지 않음)
    def status(self) -> dict:
        return {name: (name in self._wrappers) for name in _WRAPPER_CLASSES}

    # /model-info용 - 공개 가능한 정보만 뽑아서 반환 (로컬 경로/내부 구조 노출 금지)
    def public_info(self) -> dict:
        info = {}
        for name in _WRAPPER_CLASSES:
            metadata = self._metadata.get(name)
            if metadata is None:
                info[name] = {"loaded": False}
                continue
            info[name] = {
                "loaded": True,
                "modelType": metadata.get("model_type"),
                "featureCount": len(metadata.get("feature_names", [])),
                "windowSize": metadata.get("window_size"),
                "stride": metadata.get("stride"),
                "createdAt": metadata.get("created_at"),
                "labels": metadata.get("labels") or metadata.get("label_names"),
                "threshold": metadata.get("threshold") or metadata.get("anomaly_threshold"),
            }
        return info
