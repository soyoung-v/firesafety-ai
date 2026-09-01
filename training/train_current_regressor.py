"""Current Regressor 학습/평가 CLI (Phase 6).

실행: python -m training.train_current_regressor
      [--input dataset/generated/current_prediction_features.csv]
      [--artifact-dir artifacts] [--report-dir artifacts/reports]

Phase 6(`python -m training.generate_current_features`)가 먼저 Feature Dataset을 만들어 둔
상태여야 한다. Phase 3에서 만든 train/validation/test split을 다시 나누지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features.current_columns import FEATURE_COLUMNS
from .features.window import DEFAULT_STRIDE, WINDOW_SIZE
from .models.current_candidates import build_current_candidate_models
from .models.current_io import build_current_metadata, save_current_artifact
from .models.current_metrics import (
    compute_mae_improvement_pct,
    compute_regression_metrics,
    regression_metrics_by_change_magnitude,
    regression_metrics_by_group,
)
from .scenario.config import DEFAULT_SEED

DEFAULT_INPUT = Path("dataset/generated/current_prediction_features.csv")
DEFAULT_FEATURE_SUMMARY = Path("dataset/generated/current_prediction_feature_summary.json")
DEFAULT_DATASET_SUMMARY = Path("dataset/generated/dataset_summary.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_REPORT_DIR = Path("artifacts/reports")

BASELINE_NAME = "PersistenceBaseline"


# CLI 인자 파싱
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Current Regressor 학습/평가 (Phase 6)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-summary", type=Path, default=DEFAULT_FEATURE_SUMMARY)
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
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


# validation MAE가 가장 낮은 ML 모델을 고른다 (baseline은 비교 대상일 뿐 후보에서 제외 - Phase 6 명세 13절)
def _select_best_ml_model(validation_results: dict) -> str:
    ml_candidates = {name: m for name, m in validation_results.items() if name != BASELINE_NAME}
    return min(ml_candidates, key=lambda name: ml_candidates[name]["mae"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"입력 Feature Dataset이 없습니다: {args.input} (먼저 `python -m training.generate_current_features` 실행)")
        return 1

    df = pd.read_csv(args.input)
    train_df, val_df, test_df = _split(df, "train"), _split(df, "val"), _split(df, "test")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_current"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["target_current"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_current"]

    candidates = build_current_candidate_models()
    validation_results: dict = {}
    fitted_models: dict = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        validation_results[name] = compute_regression_metrics(y_val, model.predict(X_val))

    baseline_val_mae = validation_results[BASELINE_NAME]["mae"]
    for name, metrics in validation_results.items():
        metrics["mae_improvement_pct_vs_baseline"] = compute_mae_improvement_pct(baseline_val_mae, metrics["mae"])

    best_name = _select_best_ml_model(validation_results)
    best_model = fitted_models[best_name]
    baseline_model = fitted_models[BASELINE_NAME]

    test_metrics = compute_regression_metrics(y_test, best_model.predict(X_test))
    baseline_test_metrics = compute_regression_metrics(y_test, baseline_model.predict(X_test))
    test_metrics["mae_improvement_pct_vs_baseline"] = compute_mae_improvement_pct(
        baseline_test_metrics["mae"], test_metrics["mae"]
    )

    scenario_metrics = regression_metrics_by_group(y_test, best_model.predict(X_test), test_df["scenario"])
    baseline_scenario_metrics = regression_metrics_by_group(
        y_test, baseline_model.predict(X_test), test_df["scenario"]
    )

    # |current_slope| 중앙값을 안정/변화 구간 경계로 사용 (feature 기반 grouping, target 미사용)
    change_threshold = float(np.median(np.abs(test_df["current_slope"])))
    change_metrics = regression_metrics_by_change_magnitude(
        y_test, best_model.predict(X_test), test_df["current_slope"], change_threshold
    )
    baseline_change_metrics = regression_metrics_by_change_magnitude(
        y_test, baseline_model.predict(X_test), test_df["current_slope"], change_threshold
    )

    window_size, stride = _read_window_config(args.feature_summary)
    dataset_seed = _read_dataset_seed(args.dataset_summary)

    metadata = build_current_metadata(
        model_type=best_name,
        feature_names=list(FEATURE_COLUMNS),
        window_size=window_size,
        stride=stride,
        dataset_seed=dataset_seed,
        validation_metrics=validation_results[best_name],
        test_metrics=test_metrics,
        baseline_metrics={"validation": validation_results[BASELINE_NAME], "test": baseline_test_metrics},
        scenario_metrics=scenario_metrics,
    )

    model_path = args.artifact_dir / "current_regressor.joblib"
    save_current_artifact(best_model, metadata, model_path)

    report = {
        "validation_results": validation_results,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "scenario_metrics": {"selected_model": scenario_metrics, "baseline": baseline_scenario_metrics},
        "stable_vs_changing_metrics": {"selected_model": change_metrics, "baseline": baseline_change_metrics},
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "current_regressor_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"선택된 모델: {best_name}")
    print(
        f"Test MAE={test_metrics['mae']:.4f}A, RMSE={test_metrics['rmse']:.4f}A, R2={test_metrics['r2']:.4f}, "
        f"baseline 대비 개선={test_metrics['mae_improvement_pct_vs_baseline']:.2f}%"
    )
    print(f"저장 완료: {model_path}, {args.report_dir / 'current_regressor_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
