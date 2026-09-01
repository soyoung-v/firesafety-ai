"""Feature Dataset 생성 CLI (Phase 3).

실행: python -m training.generate_features [--window-size 60] [--stride 60] [--seed 42]
      [--input dataset/generated/full_sensor_dataset.csv] [--output-dir dataset/generated]

Phase 2 Dataset Generator(`python -m training.generate_dataset`)가 먼저 --input 경로에
raw Dataset을 만들어 둔 상태여야 한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features.io import save_feature_datasets
from .features.pipeline import build_feature_datasets
from .features.window import DEFAULT_STRIDE, WINDOW_SIZE
from .scenario.config import DEFAULT_SEED

DEFAULT_INPUT = Path("dataset/generated/full_sensor_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("dataset/generated")


# CLI 인자 파싱
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Feature Engineering (Phase 3)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


# raw Dataset을 읽어 Feature Dataset 2종을 생성 + 저장
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"입력 Dataset이 없습니다: {args.input} (먼저 `python -m training.generate_dataset` 실행)")
        return 1

    raw_df = pd.read_csv(args.input, parse_dates=["timestamp"])
    arc_df, risk_df = build_feature_datasets(
        raw_df, seed=args.seed, window_size=args.window_size, stride=args.stride
    )
    summary = save_feature_datasets(arc_df, risk_df, args.output_dir, args.window_size, args.stride)

    print(
        f"저장 완료: {args.output_dir} "
        f"(arc windows={summary['arc_feature_dataset']['total_windows']}, "
        f"risk windows={summary['risk_feature_dataset']['total_windows']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
