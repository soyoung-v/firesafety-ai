from simulator.frame_builder import build_frame_params, build_panel_frame_series


def test_build_panel_frame_series_has_all_ten_circuits():
    series = build_panel_frame_series("NORMAL", "00001", samples=10, seed=1, target_circuit=3)
    assert set(series.circuit_dfs.keys()) == set(range(1, 11))
    assert series.target_circuit == 3


def test_build_frame_params_has_ten_circuit_fields_and_panel_fields():
    series = build_panel_frame_series("NORMAL", "00001", samples=5, seed=1, target_circuit=1)
    params = build_frame_params(series, 0)

    for channel_no in range(1, 11):
        assert f"am{channel_no}" in params
        assert f"count{channel_no}" in params
    for key in ("m_no", "mode", "volt", "hct_count", "s_circuit", "tem", "humi",
                "fire", "gas", "door", "total_circuit", "e_energy", "aerror"):
        assert key in params


def test_only_target_circuit_follows_arc_scenario_others_stay_baseline():
    series = build_panel_frame_series("ARC", "00001", samples=60, seed=7, target_circuit=3)

    # DANGER phase(마지막 30%)에서 target 회로만 arc_count가 유의미하게 증가해야 한다
    target_arc_increase = series.target_df["arc_count"].iloc[-1] - series.target_df["arc_count"].iloc[0]
    assert target_arc_increase > 0

    for channel_no, df in series.circuit_dfs.items():
        if channel_no == series.target_circuit:
            continue
        assert (df["arc_count"] == 0).all()


def test_same_seed_reproduces_identical_frames():
    series_a = build_panel_frame_series("OVERHEATING", "00001", samples=20, seed=99, target_circuit=2)
    series_b = build_panel_frame_series("OVERHEATING", "00001", samples=20, seed=99, target_circuit=2)

    params_a = build_frame_params(series_a, 10)
    params_b = build_frame_params(series_b, 10)
    assert params_a == params_b


def test_scenario_progresses_from_normal_to_warning_to_danger():
    series = build_panel_frame_series("OVERHEATING", "00001", samples=100, seed=1, target_circuit=1)
    risk_levels = series.target_df["risk_level"].tolist()

    assert risk_levels[0] == "NORMAL"
    assert risk_levels[-1] == "DANGER"
    assert "WARNING" in risk_levels


def test_overheating_scenario_sets_overheat_alarm_bit_in_danger_phase():
    series = build_panel_frame_series("OVERHEATING", "00001", samples=100, seed=1, target_circuit=1)
    last_params = build_frame_params(series, 99)

    byte3 = int(last_params["aerror"][6:8], 16)
    assert byte3 & (1 << 1) != 0  # ALARM_BIT_OVERHEAT


def test_arc_scenario_sets_arc_bit_only_on_frames_where_arc_count_increments():
    series = build_panel_frame_series("ARC", "00001", samples=60, seed=3, target_circuit=4)

    saw_arc_bit_on = False
    for i in range(1, 60):
        params = build_frame_params(series, i)
        byte0 = int(params["aerror"][0:2], 16)
        arc_bit_on = byte0 & (1 << 3) != 0  # 회로 4 -> byte0 bit3
        current_total = int(series.target_df["arc_count"].iloc[i])
        previous_total = int(series.target_df["arc_count"].iloc[i - 1])
        assert arc_bit_on == (current_total > previous_total)
        saw_arc_bit_on = saw_arc_bit_on or arc_bit_on

    assert saw_arc_bit_on


def test_invalid_target_circuit_raises():
    import pytest

    with pytest.raises(ValueError):
        build_panel_frame_series("NORMAL", "00001", samples=5, seed=1, target_circuit=11)
