"""FastAPI(Phase 8)가 모델을 일관되게 로드할 수 있도록 artifact/metadata 경로와 출력 필드를
한 곳에서 관리한다 - 경로 문자열을 여러 코드에 중복 하드코딩하지 않기 위한 단일 소스다.
"""

from __future__ import annotations

# 모델 이름 -> artifact 파일명 / 출력 필드 / inference wrapper 클래스 경로
MODEL_MANIFEST = {
    "arcClassifier": {
        "artifact": "arc_classifier.joblib",
        "metadata": "arc_classifier.metadata.json",
        "outputs": ["pred", "proba"],
        "wrapper": "training.models.arc_inference.ArcClassifier",
    },
    "riskClassifier": {
        "artifact": "risk_classifier.joblib",
        "metadata": "risk_classifier.metadata.json",
        "outputs": ["riskLevel", "riskScore"],
        "wrapper": "training.models.risk_inference.RiskClassifier",
    },
    "anomalyDetector": {
        "artifact": "anomaly_detector.joblib",
        "metadata": "anomaly_detector.metadata.json",
        "outputs": ["anomaly", "anomalyScore"],
        "wrapper": "training.models.anomaly_inference.AnomalyDetector",
    },
    "currentRegressor": {
        "artifact": "current_regressor.joblib",
        "metadata": "current_regressor.metadata.json",
        "outputs": ["predictedCurrent"],
        "wrapper": "training.models.current_inference.CurrentRegressor",
    },
}
