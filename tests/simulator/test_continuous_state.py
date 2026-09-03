"""simulator/continuous_state.py 단위 테스트 - state.json 스키마, 전이, 파일 I/O."""

from __future__ import annotations

import json
import os

import pytest

from simulator.continuous_state import (
    DEFAULT_COOLDOWN_FRAMES,
    PHASE_DANGER,
    PHASE_WARNING,
    STATE_IDLE,
    STATE_IN_EVENT,
    SimulatorState,
    load_state,
    write_state,
)


def test_default_state_is_idle():
    state = SimulatorState()
    assert state.is_idle()
    assert state.state == STATE_IDLE


def test_start_event_creates_warning_phase_zero():
    state = SimulatorState.start_event("ARC", target_circuit=3, warning_frames=5, danger_frames=6, seed=42)
    assert state.state == STATE_IN_EVENT
    assert state.scenario == "ARC"
    assert state.targetCircuit == 3
    assert state.phase == PHASE_WARNING
    assert state.phaseIndex == 0
    assert state.warningFrames == 5
    assert state.dangerFrames == 6
    assert state.seed == 42


def test_advance_within_warning_increments_index():
    state = SimulatorState.start_event("ARC", 3, warning_frames=5, danger_frames=6, seed=42)
    advanced = state.advanced()
    assert advanced.phase == PHASE_WARNING
    assert advanced.phaseIndex == 1
    assert advanced.state == STATE_IN_EVENT


def test_advance_transitions_warning_to_danger():
    state = SimulatorState(
        state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3,
        phase=PHASE_WARNING, phaseIndex=4, warningFrames=5, dangerFrames=6, seed=42,
    )
    advanced = state.advanced()
    assert advanced.phase == PHASE_DANGER
    assert advanced.phaseIndex == 0
    # scenario/targetCircuit/frames/seed는 이벤트가 끝나지 않았으니 그대로 유지된다
    assert advanced.scenario == "ARC"
    assert advanced.warningFrames == 5
    assert advanced.dangerFrames == 6
    assert advanced.seed == 42


def test_advance_holds_within_danger():
    state = SimulatorState(
        state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3,
        phase=PHASE_DANGER, phaseIndex=2, warningFrames=5, dangerFrames=6, seed=42,
    )
    advanced = state.advanced()
    assert advanced.phase == PHASE_DANGER
    assert advanced.phaseIndex == 3


def test_advance_ends_event_after_last_danger_frame():
    state = SimulatorState(
        state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3,
        phase=PHASE_DANGER, phaseIndex=5, warningFrames=5, dangerFrames=6, seed=42,
    )
    advanced = state.advanced()
    assert advanced.is_idle()
    assert advanced.scenario is None
    assert advanced.targetCircuit is None
    assert advanced.phase is None
    assert advanced.warningFrames is None
    assert advanced.dangerFrames is None
    assert advanced.seed is None


# --- Cooldown ---

def test_event_end_starts_cooldown_with_default_frames():
    state = SimulatorState(
        state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3,
        phase=PHASE_DANGER, phaseIndex=5, warningFrames=5, dangerFrames=6, seed=42,
    )
    advanced = state.advanced()
    assert advanced.is_idle()
    assert advanced.cooldownRemaining == DEFAULT_COOLDOWN_FRAMES
    assert not advanced.can_start_new_event()


def test_event_end_starts_cooldown_with_custom_frames():
    state = SimulatorState(
        state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3,
        phase=PHASE_DANGER, phaseIndex=5, warningFrames=5, dangerFrames=6, seed=42,
    )
    advanced = state.advanced(cooldown_frames=10)
    assert advanced.cooldownRemaining == 10


def test_cooldown_decrements_by_one_each_advance():
    state = SimulatorState(cooldownRemaining=3)
    state = state.advanced()
    assert state.cooldownRemaining == 2
    state = state.advanced()
    assert state.cooldownRemaining == 1
    state = state.advanced()
    assert state.cooldownRemaining == 0


def test_cooldown_never_goes_negative():
    state = SimulatorState(cooldownRemaining=0)
    state = state.advanced()
    assert state.cooldownRemaining == 0


def test_can_start_new_event_false_while_cooling_down():
    state = SimulatorState(cooldownRemaining=1)
    assert not state.can_start_new_event()


def test_can_start_new_event_true_once_cooldown_reaches_zero():
    state = SimulatorState(cooldownRemaining=0)
    assert state.can_start_new_event()


def test_can_start_new_event_false_during_in_event_regardless_of_cooldown():
    state = SimulatorState(state=STATE_IN_EVENT, scenario="ARC", targetCircuit=1, phase=PHASE_WARNING, phaseIndex=0, warningFrames=5, dangerFrames=5, seed=1, cooldownRemaining=0)
    assert not state.can_start_new_event()


def test_full_event_cycle_ends_in_cooldown_not_immediately_startable(tmp_path):
    state = SimulatorState.start_event("LEAKAGE", target_circuit=7, warning_frames=1, danger_frames=1, seed=99)
    state = state.advanced()  # WARNING(1) 다 썼으니 DANGER 0으로 전환
    state = state.advanced(cooldown_frames=5)  # DANGER(1) 다 썼으니 이벤트 종료 + cooldown 5
    assert state.is_idle()
    assert state.cooldownRemaining == 5
    assert not state.can_start_new_event()


def test_advance_idle_stays_idle():
    state = SimulatorState()
    advanced = state.advanced()
    assert advanced.is_idle()


def test_sequence_index_warning_offset_by_normal_anchor():
    state = SimulatorState(state=STATE_IN_EVENT, scenario="ARC", targetCircuit=1, phase=PHASE_WARNING, phaseIndex=2, warningFrames=5, dangerFrames=6, seed=1)
    assert state.sequence_index() == 3  # 1(anchor) + phaseIndex(2)


def test_sequence_index_danger_offset_by_anchor_and_warning():
    state = SimulatorState(state=STATE_IN_EVENT, scenario="ARC", targetCircuit=1, phase=PHASE_DANGER, phaseIndex=1, warningFrames=5, dangerFrames=6, seed=1)
    assert state.sequence_index() == 1 + 5 + 1


def test_full_event_cycle_returns_to_idle():
    state = SimulatorState.start_event("LEAKAGE", target_circuit=7, warning_frames=2, danger_frames=2, seed=99)
    # WARNING 0 -> 1 -> DANGER 0 -> 1 -> IDLE
    state = state.advanced()  # WARNING idx1
    assert state.phase == PHASE_WARNING and state.phaseIndex == 1
    state = state.advanced()  # WARNING(2) 다 썼으니 DANGER 0으로 전환
    assert state.phase == PHASE_DANGER and state.phaseIndex == 0
    state = state.advanced()  # DANGER idx1
    assert state.phase == PHASE_DANGER and state.phaseIndex == 1
    state = state.advanced()  # DANGER(2) 다 썼으니 종료
    assert state.is_idle()


# --- 파일 I/O ---

def test_load_state_missing_file_returns_idle(tmp_path):
    path = str(tmp_path / "state.json")
    state = load_state(path)
    assert state.is_idle()


def test_load_state_corrupted_json_returns_idle(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")
    state = load_state(str(path))
    assert state.is_idle()


def test_load_state_invalid_schema_returns_idle(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"state": "SOMETHING_WEIRD"}), encoding="utf-8")
    state = load_state(str(path))
    assert state.is_idle()


def test_load_state_in_event_with_invalid_phase_returns_idle(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"state": STATE_IN_EVENT, "phase": "NOT_A_PHASE"}), encoding="utf-8")
    state = load_state(str(path))
    assert state.is_idle()


def test_write_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "nested" / "state.json")  # 디렉토리도 자동 생성돼야 한다
    state = SimulatorState.start_event("OVERHEATING", 4, warning_frames=8, danger_frames=9, seed=7)
    write_state(path, state)
    loaded = load_state(path)
    assert loaded.scenario == "OVERHEATING"
    assert loaded.targetCircuit == 4
    assert loaded.warningFrames == 8
    assert loaded.dangerFrames == 9
    assert loaded.seed == 7


def test_write_state_is_atomic_no_tmp_file_left_behind(tmp_path):
    path = str(tmp_path / "state.json")
    write_state(path, SimulatorState())
    leftover = [f for f in os.listdir(tmp_path) if f != "state.json"]
    assert leftover == []


def test_write_state_overwrites_existing_file_completely(tmp_path):
    path = str(tmp_path / "state.json")
    write_state(path, SimulatorState.start_event("ARC", 1, warning_frames=20, danger_frames=20, seed=1))
    write_state(path, SimulatorState())  # 다시 IDLE로
    loaded = load_state(path)
    assert loaded.is_idle()
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["scenario"] is None
