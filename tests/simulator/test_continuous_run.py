"""simulator/continuous_run.py 단위 테스트 - IDLE/이벤트 전이, 성공/실패 시맨틱, deterministic seed.

실제 HTTP 호출은 절대 하지 않는다 - client.send_frame을 stub으로 교체한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator import client
from simulator.continuous_run import DEFAULT_DEMO_SCENARIO_POOL, decide_next_frame, main, run_once
from simulator.continuous_scenario import EVENT_SCENARIOS
from simulator.continuous_state import PHASE_DANGER, PHASE_WARNING, STATE_IN_EVENT, SimulatorState, load_state, write_state


class _StubClient:
    """client.send_frame을 대신하는 stub - 호출 인자를 기록하고 성공/실패를 강제한다."""

    def __init__(self, succeed: bool = True):
        self.succeed = succeed
        self.calls: list[dict] = []

    def __call__(self, base_url, params, timeout_s=5.0):
        self.calls.append(params)
        if not self.succeed:
            raise client.FrameSendError(500, "stub failure")
        return {"message": "OK"}


@pytest.fixture(autouse=True)
def _patch_send_frame(monkeypatch):
    stub = _StubClient(succeed=True)
    monkeypatch.setattr(client, "send_frame", stub)
    return stub


# --- decide_next_frame: 순수 로직 ---

def test_idle_low_roll_produces_normal_frame_and_stays_idle():
    rng = np.random.default_rng(1)
    # event_probability=0으로 두면 절대 이벤트가 시작되지 않는다
    params, next_state = decide_next_frame(SimulatorState(), rng, "90001", event_probability=0.0, warning_range=(5, 15), danger_range=(5, 15))
    assert next_state.is_idle()
    assert "am1" in params  # NORMAL 프레임도 정상적인 파라미터 구조를 갖는다


def test_idle_high_roll_but_cooling_down_stays_normal():
    # event_probability=1.0이라 원래는 무조건 이벤트를 시작하지만, cooldownRemaining>0이면 시작하지 않는다
    rng = np.random.default_rng(1)
    state = SimulatorState(cooldownRemaining=5)
    params, next_state = decide_next_frame(state, rng, "90001", event_probability=1.0, warning_range=(5, 6), danger_range=(5, 6))
    assert next_state.is_idle()
    assert next_state.cooldownRemaining == 4
    assert "am1" in params


def test_idle_high_roll_starts_event():
    rng = np.random.default_rng(1)
    # event_probability=1.0이면 항상 이벤트를 시작한다
    params, next_state = decide_next_frame(SimulatorState(), rng, "90001", event_probability=1.0, warning_range=(5, 6), danger_range=(5, 6))
    assert next_state.state == STATE_IN_EVENT
    assert next_state.phase == PHASE_WARNING
    assert next_state.phaseIndex == 1  # 이번 실행에서 phaseIndex=0 프레임을 보내고 advance된 상태
    assert next_state.scenario in DEFAULT_DEMO_SCENARIO_POOL  # 기본 호출은 데모 축소 풀만 쓴다
    assert 1 <= next_state.targetCircuit <= 10


# --- 데모 자동 선택 시나리오 풀 - OVER_CURRENT/COMPLEX_RISK는 legacy ARC classifier 오탐(실측 확인)
# 때문에 공개 데모 자동 선택에서만 제외한다. continuous_scenario.EVENT_SCENARIOS(전체 5종)는 그대로이며,
# scenario_pool을 명시적으로 넘기면 이 축소 없이 전체 후보를 그대로 쓸 수 있다(Simulator CLI/수동 E2E용). ---

def test_default_demo_scenario_pool_excludes_over_current_and_complex_risk():
    assert "OVER_CURRENT" not in DEFAULT_DEMO_SCENARIO_POOL
    assert "COMPLEX_RISK" not in DEFAULT_DEMO_SCENARIO_POOL
    assert set(DEFAULT_DEMO_SCENARIO_POOL) <= set(EVENT_SCENARIOS)  # 전체 후보의 부분집합이어야 한다


def test_repeated_auto_event_starts_never_pick_excluded_scenarios():
    rng = np.random.default_rng(7)
    state = SimulatorState()
    seen = set()
    for _ in range(200):
        _, state = decide_next_frame(state, rng, "90001", event_probability=1.0, warning_range=(1, 2), danger_range=(1, 2), cooldown_frames=0)
        if state.state == STATE_IN_EVENT and state.phaseIndex == 0:
            seen.add(state.scenario)
    assert seen  # 최소 한 번은 이벤트가 시작됐어야 검증 의미가 있다
    assert seen <= set(DEFAULT_DEMO_SCENARIO_POOL)


def test_explicit_scenario_pool_override_can_still_pick_over_current():
    # Simulator CLI(--scenario-pool)/수동 E2E는 OVER_CURRENT/COMPLEX_RISK를 여전히 선택할 수 있어야 한다
    class _FixedRng:
        def random(self):
            return 0.0

        def choice(self, seq):
            return seq[0]  # 첫 번째(OVER_CURRENT)를 그대로 선택

        def integers(self, low, high=None):
            return low

    _, next_state = decide_next_frame(
        SimulatorState(), _FixedRng(), "90001", event_probability=1.0, warning_range=(2, 3), danger_range=(2, 3),
        scenario_pool=("OVER_CURRENT", "COMPLEX_RISK"),
    )
    assert next_state.scenario == "OVER_CURRENT"


def test_event_probability_boundary_just_below_threshold_does_not_start():
    class _FixedRng:
        def random(self):
            return 0.003  # event_probability=0.003과 같으면 "< " 비교라 시작 안 함(경계 포함 아님)

        def integers(self, *a, **k):
            return 5

        def choice(self, seq):
            return seq[0]

    _, next_state = decide_next_frame(SimulatorState(), _FixedRng(), "90001", event_probability=0.003, warning_range=(5, 6), danger_range=(5, 6))
    assert next_state.is_idle()


def test_event_probability_boundary_just_at_threshold_starts():
    class _FixedRng:
        def random(self):
            return 0.0029999

        def integers(self, low, high=None):
            return low

        def choice(self, seq):
            return seq[0]

    _, next_state = decide_next_frame(SimulatorState(), _FixedRng(), "90001", event_probability=0.003, warning_range=(5, 6), danger_range=(5, 6))
    assert next_state.state == STATE_IN_EVENT


def test_in_event_warning_ramp_progresses():
    state = SimulatorState(state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3, phase=PHASE_WARNING, phaseIndex=2, warningFrames=5, dangerFrames=5, seed=42)
    params, next_state = decide_next_frame(state, np.random.default_rng(1), "90001", 0.0, (5, 6), (5, 6))
    assert next_state.phase == PHASE_WARNING
    assert next_state.phaseIndex == 3
    assert f"am3" in params


def test_in_event_warning_to_danger_transition():
    state = SimulatorState(state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3, phase=PHASE_WARNING, phaseIndex=4, warningFrames=5, dangerFrames=5, seed=42)
    _, next_state = decide_next_frame(state, np.random.default_rng(1), "90001", 0.0, (5, 6), (5, 6))
    assert next_state.phase == PHASE_DANGER
    assert next_state.phaseIndex == 0


def test_in_event_danger_hold_progresses():
    state = SimulatorState(state=STATE_IN_EVENT, scenario="LEAKAGE", targetCircuit=2, phase=PHASE_DANGER, phaseIndex=1, warningFrames=5, dangerFrames=5, seed=42)
    _, next_state = decide_next_frame(state, np.random.default_rng(1), "90001", 0.0, (5, 6), (5, 6))
    assert next_state.phase == PHASE_DANGER
    assert next_state.phaseIndex == 2


def test_in_event_ends_back_to_idle():
    state = SimulatorState(state=STATE_IN_EVENT, scenario="LEAKAGE", targetCircuit=2, phase=PHASE_DANGER, phaseIndex=4, warningFrames=5, dangerFrames=5, seed=42)
    _, next_state = decide_next_frame(state, np.random.default_rng(1), "90001", 0.0, (5, 6), (5, 6))
    assert next_state.is_idle()


def test_non_target_circuits_stay_normal_during_event(_patch_send_frame):
    stub = _patch_send_frame
    state = SimulatorState(state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3, phase=PHASE_DANGER, phaseIndex=0, warningFrames=5, dangerFrames=5, seed=42)
    params, _ = decide_next_frame(state, np.random.default_rng(1), "90001", 0.0, (5, 6), (5, 6))
    # ARC DANGER는 회로 3의 aerror만 세팅한다 - 나머지 회로 전류값은 NORMAL 범위(baseline mean=5.0)에서 크게 벗어나지 않아야 한다
    for ch in range(1, 11):
        if ch == 3:
            continue
        current_a = int(params[f"am{ch}"]) / 10
        assert 0 <= current_a < 15  # NORMAL baseline(mean=5.0, std=0.3) 대비 넉넉한 여유


# --- run_once: 성공/실패 시맨틱 + 파일 I/O ---

def test_run_once_success_persists_advanced_state(tmp_path, _patch_send_frame):
    state_path = str(tmp_path / "state.json")
    write_state(state_path, SimulatorState.start_event("ARC", 3, warning_frames=5, danger_frames=5, seed=42))

    ok = run_once("http://example.invalid", "90001", state_path, event_probability=0.0, rng=np.random.default_rng(1))

    assert ok is True
    saved = load_state(state_path)
    assert saved.phase == PHASE_WARNING
    assert saved.phaseIndex == 1  # advance됨


def test_run_once_failure_keeps_state_unchanged(tmp_path, monkeypatch):
    def _fail(base_url, params, timeout_s=5.0):
        raise client.FrameSendError(500, "boom")

    monkeypatch.setattr(client, "send_frame", _fail)

    state_path = str(tmp_path / "state.json")
    original = SimulatorState(state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3, phase=PHASE_WARNING, phaseIndex=2, warningFrames=5, dangerFrames=5, seed=42)
    write_state(state_path, original)

    ok = run_once("http://example.invalid", "90001", state_path, event_probability=0.0, rng=np.random.default_rng(1))

    assert ok is False
    saved = load_state(state_path)
    assert saved.phase == PHASE_WARNING
    assert saved.phaseIndex == 2  # advance되지 않고 그대로


def test_run_once_failure_does_not_start_new_event(tmp_path, monkeypatch):
    def _fail(base_url, params, timeout_s=5.0):
        raise client.FrameSendError(500, "boom")

    monkeypatch.setattr(client, "send_frame", _fail)

    state_path = str(tmp_path / "state.json")
    # 상태 파일이 아예 없는 상태(IDLE) - event_probability=1.0이라 이번 실행이 이벤트를 시작하려 했지만 전송 실패
    ok = run_once("http://example.invalid", "90001", state_path, event_probability=1.0, rng=np.random.default_rng(1))

    assert ok is False
    saved = load_state(state_path)
    assert saved.is_idle()  # 실패했으므로 이벤트가 실제로 시작된 적이 없어야 한다


def test_run_once_missing_state_file_defaults_to_idle(tmp_path, _patch_send_frame):
    state_path = str(tmp_path / "state.json")  # 파일이 아예 없음
    ok = run_once("http://example.invalid", "90001", state_path, event_probability=0.0, rng=np.random.default_rng(1))
    assert ok is True
    assert load_state(state_path).is_idle()


def test_run_once_corrupted_state_file_recovers_to_idle(tmp_path, _patch_send_frame):
    state_path = tmp_path / "state.json"
    state_path.write_text("{ this is not json", encoding="utf-8")
    ok = run_once("http://example.invalid", "90001", str(state_path), event_probability=0.0, rng=np.random.default_rng(1))
    assert ok is True
    assert load_state(str(state_path)).is_idle()


def test_run_once_event_end_persists_cooldown_and_blocks_next_event(tmp_path, _patch_send_frame):
    state_path = str(tmp_path / "state.json")
    # DANGER 마지막 프레임 직전 상태로 시작
    write_state(state_path, SimulatorState(state=STATE_IN_EVENT, scenario="ARC", targetCircuit=3, phase=PHASE_DANGER, phaseIndex=4, warningFrames=5, dangerFrames=5, seed=42))

    ok = run_once("http://example.invalid", "90001", state_path, event_probability=0.0, cooldown_frames=7, rng=np.random.default_rng(1))
    assert ok is True
    saved = load_state(state_path)
    assert saved.is_idle()
    assert saved.cooldownRemaining == 7

    # 쿨다운 중에는 event_probability=1.0이어도 새 이벤트가 시작되지 않는다
    ok2 = run_once("http://example.invalid", "90001", state_path, event_probability=1.0, cooldown_frames=7, rng=np.random.default_rng(2))
    assert ok2 is True
    saved2 = load_state(state_path)
    assert saved2.is_idle()
    assert saved2.cooldownRemaining == 6


# --- deterministic seed: 같은 seed면 같은 결과 ---

def test_same_seed_produces_identical_sequence_of_decisions():
    def run_sequence(seed: int) -> list[str]:
        rng = np.random.default_rng(seed)
        state = SimulatorState()
        outcomes = []
        for _ in range(20):
            _, state = decide_next_frame(state, rng, "90001", event_probability=0.3, warning_range=(2, 4), danger_range=(2, 4))
            outcomes.append(f"{state.state}:{state.phase}:{state.phaseIndex}")
        return outcomes

    assert run_sequence(123) == run_sequence(123)


# --- CLI: --scenario-pool ---

def test_cli_scenario_pool_omitted_uses_demo_default(tmp_path, _patch_send_frame):
    state_path = str(tmp_path / "state.json")
    rc = main([
        "--base-url", "http://example.invalid", "--m-no", "90001", "--state-path", state_path,
        "--event-probability", "1.0", "--warning-min", "2", "--warning-max", "3",
        "--danger-min", "2", "--danger-max", "3", "--seed", "1",
    ])
    assert rc == 0
    assert load_state(state_path).scenario in DEFAULT_DEMO_SCENARIO_POOL


def test_cli_scenario_pool_explicit_override_allows_over_current(tmp_path, _patch_send_frame):
    state_path = str(tmp_path / "state.json")
    rc = main([
        "--base-url", "http://example.invalid", "--m-no", "90001", "--state-path", state_path,
        "--event-probability", "1.0", "--warning-min", "2", "--warning-max", "3",
        "--danger-min", "2", "--danger-max", "3", "--seed", "1",
        "--scenario-pool", "OVER_CURRENT",
    ])
    assert rc == 0
    assert load_state(state_path).scenario == "OVER_CURRENT"


def test_cli_scenario_pool_unknown_name_rejected(tmp_path, _patch_send_frame):
    state_path = str(tmp_path / "state.json")
    rc = main([
        "--base-url", "http://example.invalid", "--m-no", "90001", "--state-path", state_path,
        "--scenario-pool", "NOT_A_SCENARIO",
    ])
    assert rc == 2
    assert not state_path or load_state(state_path).is_idle()  # 상태 파일이 만들어지지 않았거나 IDLE 그대로
