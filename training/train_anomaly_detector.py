"""Anomaly Detector 학습/평가 CLI (Phase 5).

실행: python -m training.train_anomaly_detector
      [--input dataset/generated/risk_features.csv]
      [--artifact-dir artifacts] [--report-dir artifacts/reports] [--max-normal-fpr 0.1]

Phase 3(`python -m training.generate_features`)가 먼저 risk_features.csv를 만들어 둔 상태여야 한다.
학습에는 risk_level=='NORMAL' window만 사용한다(ADR-004). Phase 3에서 만든 train/validation/test
split은 다시 나누지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .features.risk import RISK_FEATURE_NAMES
from .features.window import DEFAULT_STRIDE, WINDOW_SIZE
from .models.anomaly_candidates import build_anomaly_candidate_models
from .models.anomaly_io import build_anomaly_metadata, save_anomaly_artifact
from .models.anomaly_metrics import (
    compute_anomaly_metrics,
    fit_score_normalizer,
    normalize_score,
    raw_to_anomaly_score,
    score_distribution_by_group,
    select_threshold,
)
from .scenario.config import DEFAULT_SEED

DEFAULT_INPUT = Path("dataset/generated/risk_features.csv")
DEFAULT_FEATURE_SUMMARY = Path("dataset/generated/feature_summary.json")
DEFAULT_DATASET_SUMMARY = Path("dataset/generated/dataset_summary.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_REPORT_DIR = Path("artifacts/reports")

# ADR-004: 정상 학습 데이터는 scenario와 무관하게 risk_level=='NORMAL'인 모든 window를 사용
NORMAL_TRAINING_DEFINITION = "risk_level == 'NORMAL' (전체 6개 시나리오의 정상 구간, ADR-004)"
DEFAULT_MAX_NORMAL_FPR = 0.1


# CLI 인자 파싱
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Anomaly Detector 학습/평가 (Phase 5)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-summary", type=Path, default=DEFAULT_FEATURE_SUMMARY)
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--max-normal-fpr", type=float, default=DEFAULT_MAX_NORMAL_FPR)
    return parser.parse_args(argv)


def _split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split]


def _read_window_config(path: Path) -> tuple[int, int]:
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        return summary["window_size"], summary["stride"]
    return WINDOW_SIZE, DEFAULT_STRIDE


def _read_dataset_seed(path: Path) -> int:
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        return summary["seed"]
    return DEFAULT_SEED


# WARNING/DANGER 점수가 전부 동일 값(std=0)으로 포화되면 anomalyScore가 "얼마나 벗어났는지"를
# 표현하지 못하고 사실상 이진값이 된다 - 이런 모델은 F1이 근소하게 높아도 우선순위를 낮춘다.
def _is_score_saturated(score_distribution: dict) -> bool:
    warning_std = score_distribution.get("WARNING", {}).get("std")
    danger_std = score_distribution.get("DANGER", {}).get("std")
    return (warning_std or 0.0) == 0.0 and (danger_std or 0.0) == 0.0


# validation DANGER Recall을 우선(동률이면 F1)으로 후보 모델을 비교해 최종 모델을 고른다 (Phase 5 명세 9절).
# 단, WARNING/DANGER 점수가 포화(std=0)되지 않은 모델을 우선한다 - 전부 포화됐으면 그대로 비교한다.
def _select_best_model(validation_results: dict) -> str:
    def score(name: str) -> tuple[int, float, float]:
        result = validation_results[name]
        selection = result["threshold_selection"]
        saturated = _is_score_saturated(result["score_distribution_by_risk_level"])
        return (0 if saturated else 1, selection["danger_recall"], selection["f1"])

    return max(validation_results, key=score)


# risk_features.csv를 읽어 정상 데이터로 후보 모델 학습 -> validation으로 threshold/모델 선택 -> test 1회 평가
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"입력 Feature Dataset이 없습니다: {args.input} (먼저 `python -m training.generate_features` 실행)")
        return 1

    df = pd.read_csv(args.input)
    train_df, val_df, test_df = _split(df, "train"), _split(df, "val"), _split(df, "test")

    normal_train_df = train_df[train_df["risk_level"] == "NORMAL"]
    X_train = normal_train_df[RISK_FEATURE_NAMES]

    X_val, val_risk_levels = val_df[RISK_FEATURE_NAMES], val_df["risk_level"]
    X_test, test_risk_levels = test_df[RISK_FEATURE_NAMES], test_df["risk_level"]

    candidates = build_anomaly_candidate_models()
    validation_results: dict = {}
    fitted_models: dict = {}
    normalizers: dict = {}

    for name, model in candidates.items():
        model.fit(X_train)
        fitted_models[name] = model

        score_min, score_max = fit_score_normalizer(model, X_train)
        normalizers[name] = (score_min, score_max)

        val_scores = normalize_score(raw_to_anomaly_score(model, X_val), score_min, score_max)
        threshold_info = select_threshold(val_scores, val_risk_levels, max_normal_fpr=args.max_normal_fpr)
        val_metrics = compute_anomaly_metrics(val_scores, val_risk_levels, threshold_info["threshold"])

        validation_results[name] = {
            "threshold_selection": threshold_info,
            "metrics": val_metrics,
            "score_distribution_by_risk_level": score_distribution_by_group(val_scores, val_risk_levels),
        }

    best_name = _select_best_model(validation_results)
    best_model = fitted_models[best_name]
    score_min, score_max = normalizers[best_name]
    threshold = validation_results[best_name]["threshold_selection"]["threshold"]

    test_scores = normalize_score(raw_to_anomaly_score(best_model, X_test), score_min, score_max)
    test_metrics = compute_anomaly_metrics(test_scores, test_risk_levels, threshold)
    test_score_by_risk_level = score_distribution_by_group(test_scores, test_risk_levels)
    test_score_by_scenario = score_distribution_by_group(test_scores, test_df["scenario"])

    window_size, stride = _read_window_config(args.feature_summary)
    dataset_seed = _read_dataset_seed(args.dataset_summary)

    metadata = build_anomaly_metadata(
        model_type=best_name,
        feature_names=list(RISK_FEATURE_NAMES),
        normal_training_definition=NORMAL_TRAINING_DEFINITION,
        window_size=window_size,
        stride=stride,
        dataset_seed=dataset_seed,
        score_min=score_min,
        score_max=score_max,
        threshold=threshold,
        validation_metrics=validation_results[best_name]["metrics"],
        test_metrics=test_metrics,
    )

    model_path = args.artifact_dir / "anomaly_detector.joblib"
    save_anomaly_artifact(best_model, metadata, model_path)

    report = {
        "normal_training_definition": NORMAL_TRAINING_DEFINITION,
        "validation_results": validation_results,
        "selected_model": best_name,
        "threshold": threshold,
        "test_metrics": test_metrics,
        "test_score_distribution_by_risk_level": test_score_by_risk_level,
        "test_score_distribution_by_scenario": test_score_by_scenario,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "anomaly_detector_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"선택된 모델: {best_name}, threshold={threshold:.4f}")
    print(
        f"Test DANGER recall={test_metrics['danger_recall']:.4f}, "
        f"NORMAL FPR={test_metrics['normal_false_positive_rate']:.4f}, F1={test_metrics['f1']:.4f}"
    )
    print(f"저장 완료: {model_path}, {args.report_dir / 'anomaly_detector_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
