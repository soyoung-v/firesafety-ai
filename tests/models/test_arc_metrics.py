"""training/models/arc_metrics.py 테스트."""

import numpy as np

from training.models.arc_metrics import THRESHOLD, compute_arc_metrics, resolve_pred


def test_threshold_is_fixed_at_0_5():
    assert THRESHOLD == 0.5


def test_resolve_pred_uses_threshold():
    proba = np.array([0.1, 0.49, 0.5, 0.51, 0.9])
    pred = resolve_pred(proba, threshold=0.5)
    assert list(pred) == [0, 0, 1, 1, 1]


def test_compute_arc_metrics_perfect_case():
    y_true = [0, 0, 1, 1]
    proba = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_arc_metrics(y_true, proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["arc_recall"] == 1.0
    assert metrics["normal_false_positive_rate"] == 0.0
    assert metrics["threshold"] == 0.5


def test_compute_arc_metrics_missed_arc_lowers_recall():
    y_true = [0, 0, 1, 1]
    proba = [0.1, 0.2, 0.3, 0.9]  # 세번째(실제 ARC)를 NORMAL로 놓침
    metrics = compute_arc_metrics(y_true, proba)
    assert metrics["arc_recall"] == 0.5


def test_compute_arc_metrics_confusion_matrix_shape():
    y_true = [0, 1]
    proba = [0.1, 0.9]
    metrics = compute_arc_metrics(y_true, proba)
    assert metrics["confusion_matrix"]["labels"] == ["NORMAL(0)", "ARC(1)"]
    assert len(metrics["confusion_matrix"]["matrix"]) == 2
