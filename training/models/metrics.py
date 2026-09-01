"""분류 성능 지표 계산. accuracy 하나만으로 모델을 고르지 않는다 - Phase 4 명세 5절.

class 순서(NORMAL < WARNING < DANGER)는 training/dataset/columns.py::RISK_LEVELS를 그대로 쓴다
(단일 정의 소스, 여러 곳에 중복 하드코딩하지 않는다).
"""

from __future__ import annotations

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ..dataset.columns import RISK_LEVELS as LABEL_ORDER

DANGER_LABEL = "DANGER"


# 예측 결과에 대한 전체 분류 지표 계산 (accuracy/macro/class별/confusion matrix/DANGER recall)
def compute_classification_metrics(y_true, y_pred) -> dict:
    labels = LABEL_ORDER
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    danger_index = labels.index(DANGER_LABEL)

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i, label in enumerate(labels)
        },
        "danger_recall": float(recall[danger_index]),
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
