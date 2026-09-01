"""Risk Classifier 학습/평가 CLI (Phase 4).

실행: python -m training.train_risk_classifier
      [--input dataset/generated/risk_features.csv]
      [--artifact-dir artifacts] [--report-dir artifacts/reports]

Phase 3(`python -m training.generate_features`)가 먼저 risk_features.csv를 만들어 둔 상태여야 한다.
Phase 3에서 만든 train/validation/test split을 그대로 쓰고 다시 나누지 않는다(Phase 4 명세 3절).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .dataset.columns import RISK_LEVELS
from .features.risk import RISK_FEATURE_NAMES
from .features.window import DEFAULT_STRIDE, WINDOW_SIZE
from .models.candidates import build_candidate_models
from .models.importance import compute_feature_importance
from .models.io import build_metadata, save_artifact
from .models.metrics import compute_classification_metrics
from .scenario.config import DEFAULT_SEED

DEFAULT_INPUT = Path("dataset/generated/risk_features.csv")
DEFAULT_FEATURE_SUMMARY = Path("dataset/generated/feature_summary.json")
DEFAULT_DATASET_SUMMARY = Path("dataset/generated/dataset_summary.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_REPORT_DIR = Path("artifacts/reports")


# CLI 인자 파싱
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Risk Classifier 학습/평가 (Phase 4)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-summary", type=Path, default=DEFAULT_FEATURE_SUMMARY)
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


def _split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split]


# Macro F1을 우선 기준으로, 동률이면 DANGER Recall로 최종 모델을 선택한다 (Phase 4 명세 5절 - accuracy 단독 사용 금지)
def _select_best_model(validation_results: dict) -> str:
    return max(validation_results, key=lambda name: (validation_results[name]["macro_f1"], validation_results[name]["danger_recall"]))


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


# risk_features.csv를 읽어 후보 모델 학습 -> validation으로 모델 선택 -> test 1회 평가 -> artifact/report 저장
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"입력 Feature Dataset이 없습니다: {args.input} (먼저 `python -m training.generate_features` 실행)")
        return 1

    df = pd.read_csv(args.input)
    train_df, val_df, test_df = _split(df, "train"), _split(df, "val"), _split(df, "test")

    X_train, y_train = train_df[RISK_FEATURE_NAMES], train_df["risk_level"]
    X_val, y_val = val_df[RISK_FEATURE_NAMES], val_df["risk_level"]
    X_test, y_test = test_df[RISK_FEATURE_NAMES], test_df["risk_level"]

    candidates = build_candidate_models()
    validation_results: dict = {}
    fitted_models: dict = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        validation_results[name] = compute_classification_metrics(y_val, model.predict(X_val))

    best_name = _select_best_model(validation_results)
    best_model = fitted_models[best_name]

    test_metrics = compute_classification_metrics(y_test, best_model.predict(X_test))
    importance = compute_feature_importance(best_model, X_val, y_val, RISK_FEATURE_NAMES)

    window_size, stride = _read_window_config(args.feature_summary)
    dataset_seed = _read_dataset_seed(args.dataset_summary)

    metadata = build_metadata(
        model_type=best_name,
        feature_names=RISK_FEATURE_NAMES,
        label_names=RISK_LEVELS,
        window_size=window_size,
        stride=stride,
        dataset_seed=dataset_seed,
        validation_metrics=validation_results[best_name],
        test_metrics=test_metrics,
    )

    model_path = args.artifact_dir / "risk_classifier.joblib"
    save_artifact(best_model, metadata, model_path)

    report = {
        "validation_results": validation_results,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "feature_importance": importance,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "risk_classifier_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"선택된 모델: {best_name}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}, DANGER recall: {test_metrics['danger_recall']:.4f}")
    print(f"저장 완료: {model_path}, {args.report_dir / 'risk_classifier_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
