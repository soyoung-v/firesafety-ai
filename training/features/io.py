"""Feature Dataset을 CSV/요약 JSON으로 저장한다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import ARC_FEATURE_COLUMNS, RISK_FEATURE_COLUMNS


# 두 Feature Dataset(CSV)과 요약(JSON)을 저장하고 요약 dict를 반환
def save_feature_datasets(
    arc_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    out_dir: Path,
    window_size: int,
    stride: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    arc_df.to_csv(out_dir / "arc_features.csv", index=False)
    risk_df.to_csv(out_dir / "risk_features.csv", index=False)

    summary = _build_summary(arc_df, risk_df, window_size, stride)
    (out_dir / "feature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _dataset_summary(df: pd.DataFrame, feature_columns: list[str], label_column: str) -> dict:
    return {
        "total_windows": int(len(df)),
        "feature_columns": feature_columns,
        "label_distribution": {str(k): int(v) for k, v in df[label_column].value_counts().items()},
        "split_distribution": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "nan_count": int(df[feature_columns].isna().sum().sum()),
        "inf_count": int(np.isinf(df[feature_columns].to_numpy(dtype=float)).sum()),
    }


def _build_summary(arc_df: pd.DataFrame, risk_df: pd.DataFrame, window_size: int, stride: int) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_size": window_size,
        "stride": stride,
        "arc_feature_dataset": _dataset_summary(arc_df, ARC_FEATURE_COLUMNS, "pred"),
        "risk_feature_dataset": _dataset_summary(risk_df, RISK_FEATURE_COLUMNS, "risk_level"),
    }
