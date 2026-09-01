"""Legacy ARC Classifier 학습/평가 CLI (Phase 7).

실행: python -m training.train_arc_classifier
      [--input dataset/generated/arc_features.csv]
      [--artifact-dir artifacts] [--report-dir artifacts/reports]

Phase 3(`python -m training.generate_features`)가 먼저 arc_features.csv를 만들어 둔 상태여야 한다.
NORMAL/ARC scenario window만 사용하며(Phase 3에서 이미 필터링됨), 기존 Spring Boot 계약을 유지하기
위해 threshold=0.5를 그대로 사용한다(ADR-002). Phase 3에서 만든 split은 다시 나누지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .features.columns import ARC_FEATURE_COLUMNS
from .features.window import DEFAULT_STRIDE, WINDOW_SIZE
from .models.arc_io import build_arc_metadata, save_arc_artifact
from .models.arc_metrics import THRESHOLD, compute_arc_metrics
from .models.candidates import build_candidate_models
from .scenario.config import DEFAULT_SEED

DEFAULT_INPUT = Path("dataset/generated/arc_features.csv")
DEFAULT_FEATURE_SUMMARY = Path("dataset/generated/feature_summary.json")
DEFAULT_DATASET_SUMMARY = Path("dataset/generated/dataset_summary.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_REPORT_DIR = Path("artifacts/reports")

LABELS = {"0": "NORMAL", "1": "ARC"}


# CLI 인자 파싱
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Legacy ARC Classifier 학습/평가 (Phase 7)")
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


# validation F1을 우선(동률이면 ARC Recall)으로 후보 모델을 비교해 최종 모델을 고른다 (Phase 4와 동일 원칙)
def _select_best_model(validation_results: dict) -> str:
    return max(validation_results, key=lambda name: (validation_results[name]["f1"], validation_results[name]["arc_recall"]))


# arc_features.csv를 읽어 NORMAL/ARC 분류기 학습 -> validation으로 모델 선택 -> test 1회 평가
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"입력 Feature Dataset이 없습니다: {args.input} (먼저 `python -m training.generate_features` 실행)")
        return 1

    df = pd.read_csv(args.input)
    train_df, val_df, test_df = _split(df, "train"), _split(df, "val"), _split(df, "test")

    X_train, y_train = train_df[ARC_FEATURE_COLUMNS], train_df["pred"]
    X_val, y_val = val_df[ARC_FEATURE_COLUMNS], val_df["pred"]
    X_test, y_test = test_df[ARC_FEATURE_COLUMNS], test_df["pred"]

    candidates = build_candidate_models()
    validation_results: dict = {}
    fitted_models: dict = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        val_proba = model.predict_proba(X_val)[:, 1]
        validation_results[name] = compute_arc_metrics(y_val, val_proba, THRESHOLD)

    best_name = _select_best_model(validation_results)
    best_model = fitted_models[best_name]

    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_arc_metrics(y_test, test_proba, THRESHOLD)

    window_size, stride = _read_window_config(args.feature_summary)
    dataset_seed = _read_dataset_seed(args.dataset_summary)

    metadata = build_arc_metadata(
        model_type=best_name,
        feature_names=list(ARC_FEATURE_COLUMNS),
        labels=LABELS,
        threshold=THRESHOLD,
        window_size=window_size,
        stride=stride,
        dataset_seed=dataset_seed,
        validation_metrics=validation_results[best_name],
        test_metrics=test_metrics,
    )

    model_path = args.artifact_dir / "arc_classifier.joblib"
    save_arc_artifact(best_model, metadata, model_path)

    report = {
        "validation_results": validation_results,
        "selected_model": best_name,
        "test_metrics": test_metrics,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "arc_classifier_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"선택된 모델: {best_name}")
    print(
        f"Test F1={test_metrics['f1']:.4f}, ARC Recall={test_metrics['arc_recall']:.4f}, "
        f"NORMAL FPR={test_metrics['normal_false_positive_rate']:.4f}"
    )
    print(f"저장 완료: {model_path}, {args.report_dir / 'arc_classifier_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
