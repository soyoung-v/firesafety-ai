"""Dataset을 CSV/요약 JSON으로 저장한다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GENERATOR_VERSION = "0.1.0"


# Dataset(CSV)과 요약(JSON)을 지정 디렉터리에 저장하고 요약 dict를 반환
def save_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    seed: int,
    runs_per_scenario: int,
    samples_per_run: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "full_sensor_dataset.csv"
    df.to_csv(csv_path, index=False)

    summary = _build_summary(df, seed, runs_per_scenario, samples_per_run)
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


# dataset_summary.json에 기록할 요약 통계 구성
def _build_summary(df: pd.DataFrame, seed: int, runs_per_scenario: int, samples_per_run: int) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "runs_per_scenario": runs_per_scenario,
        "samples_per_run": samples_per_run,
        "total_rows": int(len(df)),
        "runs_per_scenario_actual": {
            k: int(v) for k, v in df.groupby("scenario")["run_id"].nunique().items()
        },
        "rows_per_scenario": {k: int(v) for k, v in df["scenario"].value_counts().items()},
        "rows_per_risk_level": {k: int(v) for k, v in df["risk_level"].value_counts().items()},
        "columns": list(df.columns),
    }
