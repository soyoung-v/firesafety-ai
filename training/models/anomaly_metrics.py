"""Anomaly Detector 평가 지표.

anomalyScore는 "높을수록 비정상"이 되도록 방향을 통일한다 - sklearn의 score_samples()는
"낮을수록 비정상"이 규약이므로 부호를 뒤집는다. 0~1 정규화는 train(정상) 데이터의 min/max만
사용한다 - validation/test 통계를 미리 써서 정규화하면 leakage가 되므로 하지 않는다(ADR-004).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# raw score(낮을수록 비정상)를 anomalyScore 방향(높을수록 비정상)으로 뒤집는다
def raw_to_anomaly_score(model, X) -> np.ndarray:
    return -model.score_samples(X)


# train(정상) 데이터 raw score의 min/max로 0~1 정규화 파라미터 산출 (leakage 방지)
def fit_score_normalizer(model, X_train) -> tuple[float, float]:
    train_scores = raw_to_anomaly_score(model, X_train)
    return float(train_scores.min()), float(train_scores.max())


# 0~1 정규화 적용 (train 범위를 벗어나는 값은 0/1로 clip)
def normalize_score(raw_scores: np.ndarray, score_min: float, score_max: float) -> np.ndarray:
    scaled = (raw_scores - score_min) / (score_max - score_min)
    return np.clip(scaled, 0.0, 1.0)


# risk_level(NORMAL/WARNING/DANGER)을 평가용 이진 anomaly 정답으로 변환 (WARNING/DANGER=1)
def risk_level_to_anomaly_label(risk_level) -> np.ndarray:
    return (np.asarray(risk_level) != "NORMAL").astype(int)


# anomalyScore와 threshold로 anomaly(boolean) 판정
def resolve_anomaly(anomaly_scores: np.ndarray, threshold: float) -> np.ndarray:
    return anomaly_scores >= threshold


# validation set으로 threshold 후보를 스윕해 최적 threshold를 고른다.
# 정책: NORMAL FPR <= max_normal_fpr을 만족하는 후보 중 DANGER Recall이 최대인 threshold,
# 동률이면 F1로 tie-break. 제약을 만족하는 후보가 없으면 전체 후보 중에서 고르고 그 사실을 기록한다.
def select_threshold(anomaly_scores: np.ndarray, risk_levels, max_normal_fpr: float = 0.1) -> dict:
    risk_levels = np.asarray(risk_levels)
    y_true = risk_level_to_anomaly_label(risk_levels)
    normal_mask = risk_levels == "NORMAL"
    danger_mask = risk_levels == "DANGER"
    warning_mask = risk_levels == "WARNING"

    rows = []
    for t in np.unique(anomaly_scores):
        pred = anomaly_scores >= t
        rows.append(
            {
                "threshold": float(t),
                "normal_fpr": float(pred[normal_mask].mean()) if normal_mask.any() else 0.0,
                "danger_recall": float(pred[danger_mask].mean()) if danger_mask.any() else 0.0,
                "warning_recall": float(pred[warning_mask].mean()) if warning_mask.any() else 0.0,
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )

    feasible = [r for r in rows if r["normal_fpr"] <= max_normal_fpr]
    pool = feasible if feasible else rows
    best = dict(max(pool, key=lambda r: (r["danger_recall"], r["f1"])))
    best["max_normal_fpr_constraint"] = max_normal_fpr
    best["constraint_satisfied"] = bool(feasible)
    return best


# 이진 anomaly 평가 지표 계산 (ROC-AUC/PR-AUC/Precision/Recall/F1/Confusion Matrix/NORMAL FPR/WARNING·DANGER Recall)
def compute_anomaly_metrics(anomaly_scores: np.ndarray, risk_levels, threshold: float) -> dict:
    risk_levels = np.asarray(risk_levels)
    y_true = risk_level_to_anomaly_label(risk_levels)
    y_pred = resolve_anomaly(anomaly_scores, threshold).astype(int)

    normal_mask = risk_levels == "NORMAL"
    warning_mask = risk_levels == "WARNING"
    danger_mask = risk_levels == "DANGER"

    has_both_classes = len(set(y_true.tolist())) > 1
    roc_auc = float(roc_auc_score(y_true, anomaly_scores)) if has_both_classes else None
    if has_both_classes:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, anomaly_scores)
        pr_auc = float(auc(recall_curve, precision_curve))
    else:
        pr_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"labels": ["NORMAL(0)", "ANOMALY(1)"], "matrix": cm.tolist()},
        "normal_false_positive_rate": float(y_pred[normal_mask].mean()) if normal_mask.any() else None,
        "warning_recall": float(y_pred[warning_mask].mean()) if warning_mask.any() else None,
        "danger_recall": float(y_pred[danger_mask].mean()) if danger_mask.any() else None,
    }


# group(risk_level 또는 scenario)별 anomalyScore 분포 요약
def score_distribution_by_group(anomaly_scores: np.ndarray, groups) -> dict:
    scores = pd.Series(anomaly_scores)
    group_series = pd.Series(np.asarray(groups))
    summary = scores.groupby(group_series).agg(["mean", "std", "min", "max", "count"])
    return {
        str(key): {stat: (float(value) if pd.notna(value) else None) for stat, value in row.items()}
        for key, row in summary.iterrows()
    }
