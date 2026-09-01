"""training/features/labels.py 테스트."""

import pandas as pd

from training.features.labels import resolve_arc_label, resolve_window_risk_level


def _rows(risk_levels):
    return pd.DataFrame({"risk_level": risk_levels})


def test_pure_normal_window():
    assert resolve_window_risk_level(_rows(["NORMAL"] * 60)) == "NORMAL"


def test_pure_danger_window():
    assert resolve_window_risk_level(_rows(["DANGER"] * 60)) == "DANGER"


def test_mixed_window_takes_max_severity():
    # NORMAL이 대부분이어도 DANGER가 한 샘플만 있어도 DANGER로 라벨링 (max-severity 정책)
    levels = ["NORMAL"] * 55 + ["DANGER"] * 5
    assert resolve_window_risk_level(_rows(levels)) == "DANGER"


def test_normal_warning_mix_takes_warning():
    levels = ["NORMAL"] * 40 + ["WARNING"] * 20
    assert resolve_window_risk_level(_rows(levels)) == "WARNING"


def test_arc_label_zero_when_scenario_normal():
    assert resolve_arc_label("NORMAL", "NORMAL") == 0


def test_arc_label_zero_when_arc_scenario_but_window_normal():
    # scenario=ARC라도 아직 NORMAL 구간이면 0 (아직 아크가 발생하지 않음)
    assert resolve_arc_label("ARC", "NORMAL") == 0


def test_arc_label_one_when_arc_scenario_and_danger():
    assert resolve_arc_label("ARC", "DANGER") == 1


def test_arc_label_one_when_arc_scenario_and_warning():
    assert resolve_arc_label("ARC", "WARNING") == 1
