"""Current Regressor 평가 지표. MAE(단위 A)를 대표 지표로 삼는다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 회귀 지표 계산 (MAE/RMSE/R², 단위는 물리값 A)
def compute_regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(y_true))}


# baseline 대비 MAE 개선율(%) 계산 - 양수면 개선, 음수면 baseline보다 나쁨
def compute_mae_improvement_pct(baseline_mae: float, model_mae: float) -> float:
    if baseline_mae == 0:
        return 0.0
    return float((baseline_mae - model_mae) / baseline_mae * 100)


# group(scenario 등)별 회귀 지표 계산
def regression_metrics_by_group(y_true, y_pred, groups) -> dict:
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "group": np.asarray(groups)})
    return {
        str(group_value): compute_regression_metrics(g["y_true"], g["y_pred"])
        for group_value, g in frame.groupby("group")
    }


# |current_slope| 기준으로 안정 구간/변화 구간을 나눠 회귀 지표를 비교한다.
# current_slope는 window feature(과거 시점)이지 target(미래값)이 아니므로 leakage가 아니다.
def regression_metrics_by_change_magnitude(y_true, y_pred, current_slope, threshold: float) -> dict:
    change_mask = np.abs(np.asarray(current_slope)) > threshold
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "stable": compute_regression_metrics(y_true[~change_mask], y_pred[~change_mask]),
        "changing": compute_regression_metrics(y_true[change_mask], y_pred[change_mask]),
        "threshold": float(threshold),
    }
