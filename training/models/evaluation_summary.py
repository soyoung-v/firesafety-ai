"""여러 Phase에 흩어진 평가 결과를 한눈에 볼 수 있는 종합 Report를 만든다.

새로운 성능을 계산하지 않는다 - 각 Phase가 이미 만든 artifacts/reports/*.json을 그대로 읽어
필요한 지표만 뽑아 모은다.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTE = "Synthetic Dataset 기준 평가 결과를 각 Phase report에서 그대로 인용했다. 여기서 새로 계산한 값은 없다."


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# 4개 모델 report를 읽어 핵심 지표만 뽑은 종합 요약 dict 생성 (기존 report를 인용만 함)
def build_evaluation_summary(report_dir: Path) -> dict:
    summary: dict = {}

    arc = _read_json(report_dir / "arc_classifier_metrics.json")
    if arc:
        tm = arc["test_metrics"]
        summary["arcClassifier"] = {
            "selected_model": arc["selected_model"],
            "f1": tm["f1"],
            "arc_recall": tm["arc_recall"],
            "normal_false_positive_rate": tm["normal_false_positive_rate"],
        }

    risk = _read_json(report_dir / "risk_classifier_metrics.json")
    if risk:
        tm = risk["test_metrics"]
        summary["riskClassifier"] = {
            "selected_model": risk["selected_model"],
            "macro_f1": tm["macro_f1"],
            "danger_recall": tm["danger_recall"],
        }

    anomaly = _read_json(report_dir / "anomaly_detector_metrics.json")
    if anomaly:
        tm = anomaly["test_metrics"]
        summary["anomalyDetector"] = {
            "selected_model": anomaly["selected_model"],
            "f1": tm["f1"],
            "danger_recall": tm["danger_recall"],
            "normal_false_positive_rate": tm["normal_false_positive_rate"],
        }

    current = _read_json(report_dir / "current_regressor_metrics.json")
    if current:
        tm = current["test_metrics"]
        summary["currentRegressor"] = {
            "selected_model": current["selected_model"],
            "mae": tm["mae"],
            "rmse": tm["rmse"],
            "r2": tm["r2"],
            "mae_improvement_pct_vs_baseline": tm["mae_improvement_pct_vs_baseline"],
        }

    summary["note"] = NOTE
    return summary
