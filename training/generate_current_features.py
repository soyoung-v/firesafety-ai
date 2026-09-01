"""Current Prediction Feature Dataset 생성 CLI (Phase 6).

실행: python -m training.generate_current_features [--window-size 60] [--stride 60] [--seed 42]
      [--input dataset/generated/full_sensor_dataset.csv] [--output-dir dataset/generated]

Phase 2(`python -m training.generate_dataset`)가 먼저 raw Dataset을 만들어 둔 상태여야 한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features.current_io import save_current_prediction_dataset
from .features.current_pipeline import build_current_prediction_dataset
from .features.window import DEFAULT_STRIDE, WINDOW_SIZE
from .scenario.config import DEFAULT_SEED

DEFAULT_INPUT = Path("dataset/generated/full_sensor_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("dataset/generated")


# CLI 인자 파싱
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Current Prediction Feature Dataset 생성 (Phase 6)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


# raw Dataset을 읽어 Current Prediction Feature Dataset 생성 + 저장
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"입력 Dataset이 없습니다: {args.input} (먼저 `python -m training.generate_dataset` 실행)")
        return 1

    raw_df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = build_current_prediction_dataset(
        raw_df, seed=args.seed, window_size=args.window_size, stride=args.stride
    )
    summary = save_current_prediction_dataset(df, args.output_dir, args.window_size, args.stride)

    print(f"저장 완료: {args.output_dir} (windows={summary['total_windows']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
