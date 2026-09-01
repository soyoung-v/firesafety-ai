"""전체 모델 Artifact 종합 검증 + Model Manifest/평가 종합 Report 생성 CLI (Phase 7).

실행: python -m training.validate_artifacts [--artifact-dir artifacts] [--report-dir artifacts/reports]

Phase 2~7의 학습 스크립트를 전부 실행해 4개 artifact(arc/risk/anomaly/current)가 이미 저장돼
있어야 한다. FastAPI(Phase 8) 통합 직전 상태 점검용이며 새 모델을 학습하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models.artifact_validation import validate_all_artifacts
from .models.evaluation_summary import build_evaluation_summary
from .models.manifest import MODEL_MANIFEST

DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_REPORT_DIR = Path("artifacts/reports")


# CLI 인자 파싱
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard 전체 모델 Artifact 검증 (Phase 7)")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


# Model Manifest 저장 -> 4개 Artifact 검증 -> 종합 평가 Report 생성
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.artifact_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(MODEL_MANIFEST, indent=2, ensure_ascii=False), encoding="utf-8")

    validation = validate_all_artifacts(args.artifact_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "artifact_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = build_evaluation_summary(args.report_dir)
    (args.report_dir / "model_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Manifest 저장: {manifest_path}")
    print(f"Artifact 검증 결과: {args.report_dir / 'artifact_validation.json'} (all_models_ready={validation['all_models_ready']})")
    print(f"종합 평가 Report: {args.report_dir / 'model_evaluation_summary.json'}")

    return 0 if validation["all_models_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
