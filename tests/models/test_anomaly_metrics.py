"""training/models/anomaly_metrics.py 테스트."""

import numpy as np
import pytest

from training.models.anomaly_metrics import (
    compute_anomaly_metrics,
    fit_score_normalizer,
    normalize_score,
    raw_to_anomaly_score,
    resolve_anomaly,
    risk_level_to_anomaly_label,
    score_distribution_by_group,
    select_threshold,
)


class _FakeModel:
    # score_samples를 흉내내는 가짜 모델 - 낮을수록 비정상(sklearn 규약)이 되게 만든다
    def __init__(self, values):
        self.values = np.asarray(values)

    def score_samples(self, X):
        return self.values


def test_raw_to_anomaly_score_flips_direction():
    # sklearn score_samples: 낮을수록 비정상 -> raw_to_anomaly_score: 높을수록 비정상이어야 함
    model = _FakeModel([-0.5, 0.1, 0.3])  # -0.5가 가장 비정상(가장 낮음)
    scores = raw_to_anomaly_score(model, X=None)
    assert scores[0] > scores[1] > scores[2]


def test_fit_score_normalizer_uses_only_given_data():
    model = _FakeModel([0.2, 0.5, 0.8])
    score_min, score_max = fit_score_normalizer(model, X_train=None)
    raw = raw_to_anomaly_score(model, X=None)
    assert score_min == pytest.approx(raw.min())
    assert score_max == pytest.approx(raw.max())


def test_risk_level_to_anomaly_label():
    labels = risk_level_to_anomaly_label(["NORMAL", "WARNING", "DANGER", "NORMAL"])
    assert list(labels) == [0, 1, 1, 0]


def test_normalize_score_maps_train_min_max_to_0_1():
    raw = np.array([1.0, 2.0, 3.0])
    scaled = normalize_score(raw, score_min=1.0, score_max=3.0)
    assert scaled[0] == 0.0
    assert scaled[-1] == 1.0
    assert scaled[1] == 0.5


def test_normalize_score_clips_out_of_range_values():
    raw = np.array([-5.0, 10.0])
    scaled = normalize_score(raw, score_min=0.0, score_max=1.0)
    assert scaled[0] == 0.0
    assert scaled[1] == 1.0


def test_resolve_anomaly_threshold():
    scores = np.array([0.1, 0.5, 0.9])
    result = resolve_anomaly(scores, threshold=0.5)
    assert list(result) == [False, True, True]


def test_select_threshold_prefers_max_danger_recall_within_fpr_constraint():
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.9, 0.95])
    risk_levels = ["NORMAL", "NORMAL", "NORMAL", "WARNING", "DANGER", "DANGER"]
    result = select_threshold(scores, risk_levels, max_normal_fpr=0.34)  # NORMAL 1/3까지 허용
    assert result["danger_recall"] == 1.0
    assert result["constraint_satisfied"] is True


def test_select_threshold_reports_infeasible_constraint():
    # 모든 threshold가 NORMAL FPR 제약을 못 지키는 극단적인 경우
    scores = np.array([0.9, 0.9, 0.9])
    risk_levels = ["NORMAL", "NORMAL", "DANGER"]
    result = select_threshold(scores, risk_levels, max_normal_fpr=0.0)
    assert result["constraint_satisfied"] is False


def test_compute_anomaly_metrics_perfect_case():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    risk_levels = ["NORMAL", "NORMAL", "WARNING", "DANGER"]
    metrics = compute_anomaly_metrics(scores, risk_levels, threshold=0.5)
    assert metrics["normal_false_positive_rate"] == 0.0
    assert metrics["danger_recall"] == 1.0
    assert metrics["warning_recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_anomaly_metrics_confusion_matrix_shape():
    scores = np.array([0.1, 0.9])
    risk_levels = ["NORMAL", "DANGER"]
    metrics = compute_anomaly_metrics(scores, risk_levels, threshold=0.5)
    assert len(metrics["confusion_matrix"]["matrix"]) == 2
    assert all(len(row) == 2 for row in metrics["confusion_matrix"]["matrix"])


def test_score_distribution_by_group():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    groups = ["NORMAL", "NORMAL", "DANGER", "DANGER"]
    dist = score_distribution_by_group(scores, groups)
    assert dist["NORMAL"]["mean"] == 0.15000000000000002 or abs(dist["NORMAL"]["mean"] - 0.15) < 1e-9
    assert dist["DANGER"]["count"] == 2.0
