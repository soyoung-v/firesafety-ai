"""training/models/metrics.py 테스트."""

import pandas as pd

from training.models.metrics import compute_classification_metrics


def test_perfect_predictions_score_1():
    y_true = pd.Series(["NORMAL", "WARNING", "DANGER", "NORMAL"])
    y_pred = ["NORMAL", "WARNING", "DANGER", "NORMAL"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["danger_recall"] == 1.0


def test_danger_recall_reflects_missed_danger():
    # DANGER 2건 중 1건을 WARNING으로 놓친 경우 -> DANGER recall = 0.5
    y_true = pd.Series(["DANGER", "DANGER", "NORMAL"])
    y_pred = ["DANGER", "WARNING", "NORMAL"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["danger_recall"] == 0.5


def test_confusion_matrix_shape_and_labels():
    y_true = pd.Series(["NORMAL", "WARNING", "DANGER"])
    y_pred = ["NORMAL", "WARNING", "DANGER"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["confusion_matrix"]["labels"] == ["NORMAL", "WARNING", "DANGER"]
    assert len(metrics["confusion_matrix"]["matrix"]) == 3
    assert all(len(row) == 3 for row in metrics["confusion_matrix"]["matrix"])


def test_per_class_metrics_present_for_all_labels():
    y_true = pd.Series(["NORMAL", "WARNING", "DANGER"])
    y_pred = ["NORMAL", "NORMAL", "NORMAL"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert set(metrics["per_class"].keys()) == {"NORMAL", "WARNING", "DANGER"}
    assert metrics["per_class"]["DANGER"]["recall"] == 0.0


def test_macro_f1_not_accuracy():
    # 클래스 불균형에서 accuracy와 macro_f1이 달라지는 것을 확인 (accuracy 단독 사용 금지 근거)
    y_true = pd.Series(["NORMAL"] * 9 + ["DANGER"])
    y_pred = ["NORMAL"] * 10  # DANGER를 전혀 못 맞춤
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.9
    assert metrics["macro_f1"] < 0.9
    assert metrics["danger_recall"] == 0.0
