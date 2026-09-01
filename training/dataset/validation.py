"""생성된 Dataset의 품질 검증.

GAS/FIRE처럼 임계값이 [TBD]인 항목은 위험 판정 검증을 만들지 않는다 (dataset-spec.md 3절/10절) -
여기서는 null/범위/스키마/전이 존재 여부만 확인한다.
"""

from __future__ import annotations

import pandas as pd

from .columns import FEATURE_COLUMNS, LABEL_COLUMNS, META_COLUMNS, RISK_LEVELS, SCENARIOS

NON_NEGATIVE_COLUMNS = (
    "current",
    "arc_count",
    "leakage_current",
    "fire_raw",
    "gas_raw",
    "total_current",
    "total_power",
    "voltage",
)


# 전체 검증 실행 - 문제 목록 반환(빈 리스트면 통과)
def validate_dataset(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    issues += _check_columns(df)
    if issues:
        # 컬럼 자체가 없으면 이후 검증이 KeyError로 죽으므로 여기서 먼저 중단
        return issues
    issues += _check_nulls(df)
    issues += _check_ranges(df)
    issues += _check_run_ordering(df)
    issues += _check_labels(df)
    issues += _check_run_ids(df)
    issues += _check_risk_transitions(df)
    return issues


def _check_columns(df: pd.DataFrame) -> list[str]:
    required = META_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    return [f"필수 컬럼 누락: {missing}"] if missing else []


def _check_nulls(df: pd.DataFrame) -> list[str]:
    null_counts = df.isnull().sum()
    bad = null_counts[null_counts > 0]
    return [f"null 존재: {dict(bad)}"] if len(bad) else []


# circuit 범위, 물리적으로 불가능한 음수/범위 초과 확인
def _check_ranges(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if not df["circuit"].between(1, 10).all():
        issues.append("circuit 범위(1~10) 위반 row 존재")
    for col in NON_NEGATIVE_COLUMNS:
        if (df[col] < 0).any():
            issues.append(f"{col} 음수 존재")
    if not df["humidity"].between(0, 100).all():
        issues.append("humidity 0~100 범위 위반 row 존재")
    return issues


# 동일 run에서 sample_index/timestamp 순서 유지 확인
def _check_run_ordering(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for run_id, group in df.groupby("run_id"):
        if not group["sample_index"].is_monotonic_increasing:
            issues.append(f"run_id={run_id} sample_index 순서 어긋남")
        if not group["timestamp"].is_monotonic_increasing:
            issues.append(f"run_id={run_id} timestamp 순서 어긋남")
    return issues


def _check_labels(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    bad_scenario = set(df["scenario"].unique()) - set(SCENARIOS)
    if bad_scenario:
        issues.append(f"알 수 없는 scenario 값: {bad_scenario}")
    bad_risk = set(df["risk_level"].unique()) - set(RISK_LEVELS)
    if bad_risk:
        issues.append(f"알 수 없는 risk_level 값: {bad_risk}")
    return issues


# run_id 하나는 정확히 하나의 scenario에만 속해야 한다
def _check_run_ids(df: pd.DataFrame) -> list[str]:
    per_run_scenarios = df.groupby("run_id")["scenario"].nunique()
    bad = per_run_scenarios[per_run_scenarios > 1]
    return [f"run_id별 scenario 불일치: {list(bad.index)}"] if len(bad) else []


# 시나리오별로 문서에 정의된 위험도 전이가 존재하는지 확인 (NORMAL 시나리오는 예외)
def _check_risk_transitions(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for scenario, group in df.groupby("scenario"):
        levels_present = set(group["risk_level"].unique())
        if scenario == "NORMAL":
            if levels_present != {"NORMAL"}:
                issues.append(f"NORMAL 시나리오에 NORMAL 외 risk_level 존재: {levels_present}")
        else:
            expected = {"NORMAL", "WARNING", "DANGER"}
            if not expected.issubset(levels_present):
                issues.append(f"{scenario} 시나리오에 필요한 risk_level 전이 없음: 존재={levels_present}")
    return issues
