"""Current Prediction Feature Dataset을 CSV/요약 JSON으로 저장한다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .current_columns import FEATURE_COLUMNS


# Feature Dataset(CSV)과 요약(JSON)을 저장하고 요약 dict를 반환
def save_current_prediction_dataset(
    df: pd.DataFrame, out_dir: Path, window_size: int, stride: int
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "current_prediction_features.csv", index=False)

    summary = _build_summary(df, window_size, stride)
    (out_dir / "current_prediction_feature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _build_summary(df: pd.DataFrame, window_size: int, stride: int) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_size": window_size,
        "stride": stride,
        "target_definition": "next sample current (window 마지막 sample 바로 다음 sample의 current)",
        "total_windows": int(len(df)),
        "feature_columns": list(FEATURE_COLUMNS),
        "rows_per_scenario": {k: int(v) for k, v in df["scenario"].value_counts().items()},
        "rows_per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "target_current_stats": {
            "mean": float(df["target_current"].mean()),
            "std": float(df["target_current"].std()),
            "min": float(df["target_current"].min()),
            "max": float(df["target_current"].max()),
        },
    }
