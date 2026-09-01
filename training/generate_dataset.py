"""Synthetic Sensor Dataset 생성 CLI.

실행: python -m training.generate_dataset [--seed 42] [--runs-per-scenario 40] [--samples-per-run 400]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset.generator import DatasetGenerator
from .dataset.io import save_dataset
from .dataset.validation import validate_dataset
from .scenario.config import DEFAULT_RUNS_PER_SCENARIO, DEFAULT_SAMPLES_PER_RUN, DEFAULT_SEED

DEFAULT_OUTPUT_DIR = Path("dataset/generated")


# CLI 인자 파싱
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Synthetic Sensor Dataset Generator")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs-per-scenario", type=int, default=DEFAULT_RUNS_PER_SCENARIO)
    parser.add_argument("--samples-per-run", type=int, default=DEFAULT_SAMPLES_PER_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-validation", action="store_true", help="생성 후 자동 검증을 건너뛴다(디버깅용)"
    )
    return parser.parse_args(argv)


# Dataset 생성 + 검증 + 저장 실행
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    generator = DatasetGenerator(
        runs_per_scenario=args.runs_per_scenario,
        samples_per_run=args.samples_per_run,
        seed=args.seed,
    )
    df = generator.generate()

    if not args.skip_validation:
        issues = validate_dataset(df)
        if issues:
            for issue in issues:
                print(f"[VALIDATION FAIL] {issue}")
            return 1

    summary = save_dataset(df, args.output_dir, args.seed, args.runs_per_scenario, args.samples_per_run)
    print(f"저장 완료: {args.output_dir} (rows={summary['total_rows']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
