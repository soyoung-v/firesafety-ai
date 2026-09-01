"""Legacy ARC Classifier 평가 지표.

pred/proba는 기존 Spring Boot 계약과 동일한 의미를 유지한다 - pred=0(NORMAL)/1(ARC),
proba=ARC일 확률, threshold=0.5 고정(기존 계약 호환, Phase 7 명세 3절 - 임의로 바꾸지 않는다).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD = 0.5


# proba(ARC 확률)를 기존 계약 threshold(0.5)로 pred(0/1) 판정
def resolve_pred(proba: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    return (np.asarray(proba) >= threshold).astype(int)


# ARC Classifier 평가 지표 계산 (Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC/Confusion Matrix/ARC Recall/NORMAL FPR)
def compute_arc_metrics(y_true, proba, threshold: float = THRESHOLD) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = resolve_pred(proba, threshold)

    normal_mask = y_true == 0
    arc_mask = y_true == 1
    has_both_classes = len(set(y_true.tolist())) > 1

    roc_auc = float(roc_auc_score(y_true, proba)) if has_both_classes else None
    if has_both_classes:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba)
        pr_auc = float(auc(recall_curve, precision_curve))
    else:
        pr_auc = None

    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"labels": ["NORMAL(0)", "ARC(1)"], "matrix": cm.tolist()},
        "arc_recall": float(pred[arc_mask].mean()) if arc_mask.any() else None,
        "normal_false_positive_rate": float(pred[normal_mask].mean()) if normal_mask.any() else None,
    }
