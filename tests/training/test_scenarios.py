"""시나리오 엔진 단위 테스트 - training/scenario/*.py."""

import numpy as np
import pytest

from training.scenario.registry import SCENARIOS, get_scenario

SMALL_SAMPLES = 60
RISKY_SAMPLES = 300  # NORMAL/WARNING/DANGER 세 구간이 모두 확보되도록 기본 window(60) 이상으로 넉넉히


@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
def test_scenario_generates_expected_schema(scenario_name):
    engine = get_scenario(scenario_name)
    rng = np.random.default_rng(1)
    df = engine.generate_run("TEST-0001", SMALL_SAMPLES, rng)

    assert len(df) == SMALL_SAMPLES
    assert {"current", "arc_count", "risk_level", "scenario", "run_id", "sample_index"}.issubset(df.columns)
    assert (df["scenario"] == scenario_name).all()
    assert (df["run_id"] == "TEST-0001").all()
    assert list(df["sample_index"]) == list(range(SMALL_SAMPLES))


def test_normal_scenario_has_no_risk_transition():
    engine = get_scenario("NORMAL")
    rng = np.random.default_rng(1)
    df = engine.generate_run("TEST-NORMAL", SMALL_SAMPLES, rng)
    assert set(df["risk_level"].unique()) == {"NORMAL"}


@pytest.mark.parametrize("scenario_name", ["OVER_CURRENT", "ARC", "LEAKAGE", "OVERHEATING", "COMPLEX_RISK"])
def test_risky_scenarios_transition_through_all_levels(scenario_name):
    engine = get_scenario(scenario_name)
    rng = np.random.default_rng(1)
    df = engine.generate_run(f"TEST-{scenario_name}", RISKY_SAMPLES, rng)
    assert set(df["risk_level"].unique()) == {"NORMAL", "WARNING", "DANGER"}


def test_arc_scenario_arc_count_increases_in_danger():
    engine = get_scenario("ARC")
    rng = np.random.default_rng(3)
    df = engine.generate_run("TEST-ARC", RISKY_SAMPLES, rng)
    danger_max = df[df["risk_level"] == "DANGER"]["arc_count"].max()
    normal_max = df[df["risk_level"] == "NORMAL"]["arc_count"].max()
    assert danger_max > normal_max


def test_leakage_scenario_leakage_current_increases():
    engine = get_scenario("LEAKAGE")
    rng = np.random.default_rng(4)
    df = engine.generate_run("TEST-LEAKAGE", RISKY_SAMPLES, rng)
    normal_mean = df[df["risk_level"] == "NORMAL"]["leakage_current"].mean()
    danger_mean = df[df["risk_level"] == "DANGER"]["leakage_current"].mean()
    assert danger_mean > normal_mean


def test_overheating_scenario_temperature_increases():
    engine = get_scenario("OVERHEATING")
    rng = np.random.default_rng(5)
    df = engine.generate_run("TEST-OVERHEATING", RISKY_SAMPLES, rng)
    normal_mean = df[df["risk_level"] == "NORMAL"]["temperature"].mean()
    danger_mean = df[df["risk_level"] == "DANGER"]["temperature"].mean()
    assert danger_mean > normal_mean


def test_over_current_scenario_total_current_increases():
    engine = get_scenario("OVER_CURRENT")
    rng = np.random.default_rng(6)
    df = engine.generate_run("TEST-OVER_CURRENT", RISKY_SAMPLES, rng)
    normal_mean = df[df["risk_level"] == "NORMAL"]["total_current"].mean()
    danger_mean = df[df["risk_level"] == "DANGER"]["total_current"].mean()
    assert danger_mean > normal_mean


def test_complex_risk_combines_two_signals():
    # COMPLEX_RISK는 pair 중 하나를 무작위로 고르므로 여러 seed로 반복해 다양성을 최소 확인
    seen_pairs_signal = set()
    for seed in range(10):
        engine = get_scenario("COMPLEX_RISK")
        rng = np.random.default_rng(seed)
        df = engine.generate_run(f"TEST-COMPLEX-{seed}", RISKY_SAMPLES, rng)
        danger = df[df["risk_level"] == "DANGER"]
        normal = df[df["risk_level"] == "NORMAL"]
        changed_fields = []
        for field in ("temperature", "total_current", "leakage_current"):
            if danger[field].mean() > normal[field].mean() * 1.05:
                changed_fields.append(field)
        if danger["arc_count"].max() > normal["arc_count"].max():
            changed_fields.append("arc_count")
        assert len(changed_fields) >= 2  # "2개 이상 동시 발생" (dataset-spec.md 3-6절)
        seen_pairs_signal.add(tuple(sorted(changed_fields)))
    assert len(seen_pairs_signal) > 1  # 여러 조합이 실제로 다양하게 나오는지 확인
