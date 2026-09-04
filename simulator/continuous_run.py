"""Continuous Demo Simulator. `python -m simulator.continuous_run`로 실행하며,
매 실행(one-shot process)마다 정확히 sensor frame 1개만 Backend로 전송하고 종료한다.

systemd timer가 이 스크립트를 약 60초 간격으로 반복 실행하는 것을 전제로 한다 - 이 파일 자체는
루프/sleep을 갖지 않는다(상주 daemon이 아님, 저사양 EC2에서 메모리를 계속 점유하지 않기 위함).

상태(state.json)는 continuous_state.py가 관리하고, 위험 이벤트의 물리값 시퀀스는
continuous_scenario.py가 training.scenario 기존 상수를 그대로 재사용해 만든다. 이 파일은
"IDLE이면 NORMAL 1프레임 또는 낮은 확률로 이벤트 시작, IN_EVENT면 현재 phase의 다음 프레임"을
결정하고 전송하는 오케스트레이션만 담당한다.

중요: Backend 전송이 성공한 경우에만 상태를 다음 프레임으로 넘긴다(advance). 실패하면 state를
그대로 두고 종료해서, 다음 timer 실행이 같은 지점부터 다시 시도한다.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import client
from .continuous_scenario import EVENT_SCENARIOS, build_event_series
from .continuous_state import DEFAULT_COOLDOWN_FRAMES, STATE_IN_EVENT, SimulatorState, load_state, write_state
from .frame_builder import build_frame_params, build_panel_frame_series

DEFAULT_STATE_PATH = "./demo-simulator-state.json"
DEFAULT_EVENT_PROBABILITY = 0.003  # 쿨다운이 끝난 IDLE일 때 새 위험 이벤트를 시작할 확률 - "확정 발생량"이
# 아니라 "하루 수 건 수준을 목표로 한 정책값"이다(1440회/day 실행 기준 대략적인 목표치일 뿐, 실제 발생
# 횟수는 쿨다운으로 그 사이 시행 자체가 줄어들기 때문에 이 값과 정확히 비례하지 않는다).
DEFAULT_WARNING_FRAMES_RANGE = (5, 15)  # inclusive-low, exclusive-high (np.random.Generator.integers 규칙)
DEFAULT_DANGER_FRAMES_RANGE = (5, 15)
MIN_CIRCUIT = 1
MAX_CIRCUIT = 10

# Continuous Demo의 "자동 랜덤 선택" 전용 풀 - continuous_scenario.EVENT_SCENARIOS(전체 5종)의 부분집합이다.
# OVER_CURRENT/COMPLEX_RISK를 여기서만 뺀 이유: 실측(local pytest + artifacts 모델로 직접 추론)에서 두
# 시나리오의 DANGER 구간이 legacy ARC classifier를 오탐(pred=1)시켜, "OVER_CURRENT 데모인데 AI/ARC 경보가
# 뜨는" 혼동을 공개 Portfolio Demo에서 보여주고 싶지 않기 때문이다. Simulator CLI(--scenario-pool로 override
# 가능)/Dataset 생성/기존 테스트/수동 E2E 시나리오는 계속 5종 전체를 그대로 쓴다 - 이 튜플은 continuous_run
# 자동 선택 로직 한 곳에만 영향을 준다. ARC classifier의 cross-scenario 오탐이 개선되면 이 튜플에 두 값만
# 추가하면 되고, 그 외 아무 데도 손댈 필요가 없다.
DEFAULT_DEMO_SCENARIO_POOL: tuple[str, ...] = ("ARC", "LEAKAGE", "OVERHEATING")

# 90001(--m-no)은 위 SimulatorState 상태기계로 NORMAL/위험 이벤트를 관리하고, 나머지 데모 분전반들은
# 여기 heartbeat 대상으로만 등록한다 - state.json을 전혀 읽거나 쓰지 않는 무상태 NORMAL 1프레임 전송뿐이라
# WARNING/DANGER 이벤트가 발생할 수 없다(decide_next_frame을 거치지 않음). CommunicationMonitor의
# 1분 통신두절 판정을 피하려는 목적이므로, 이 목록에 포함된 분전반은 절대 위험 시나리오 대상이 되면 안 된다.
HEARTBEAT_MNOS: tuple[str, ...] = ("90011", "90012", "90013", "90014", "90015", "90016", "90017")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Continuous Demo Simulator - one-shot 실행, sensor frame 1개만 전송")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--m-no", required=True, help="분전반 장비번호 (5자리)")
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="상태 저장 경로(JSON)")
    parser.add_argument("--event-probability", type=float, default=DEFAULT_EVENT_PROBABILITY)
    parser.add_argument("--warning-min", type=int, default=DEFAULT_WARNING_FRAMES_RANGE[0])
    parser.add_argument("--warning-max", type=int, default=DEFAULT_WARNING_FRAMES_RANGE[1])
    parser.add_argument("--danger-min", type=int, default=DEFAULT_DANGER_FRAMES_RANGE[0])
    parser.add_argument("--danger-max", type=int, default=DEFAULT_DANGER_FRAMES_RANGE[1])
    parser.add_argument("--cooldown-frames", type=int, default=DEFAULT_COOLDOWN_FRAMES, help="위험 이벤트 종료 후 다음 이벤트를 시작할 수 없는 최소 NORMAL 프레임 수")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None, help="지정하면 이번 실행의 IDLE 판단/신규 이벤트 뽑기가 결정론적이 된다(테스트용)")
    parser.add_argument(
        "--scenario-pool",
        type=str,
        default=None,
        help=(
            "자동 이벤트 시작 시 뽑을 시나리오 후보(쉼표 구분, 예: ARC,LEAKAGE,OVERHEATING,OVER_CURRENT). "
            f"생략하면 데모 기본 풀({','.join(DEFAULT_DEMO_SCENARIO_POOL)})을 쓴다 - "
            "OVER_CURRENT/COMPLEX_RISK는 수동 E2E/dataset/tests에서는 그대로 쓸 수 있지만 데모 자동 선택 기본값에서는 뺐다."
        ),
    )
    return parser.parse_args(argv)


# 이번 실행에서 보낼 frame params 1개를 결정한다. 반환된 next_state는 "전송 성공 시" 저장할 상태다 -
# 호출부가 실제 전송 성공 여부를 확인한 뒤에만 write_state를 호출해야 한다(실패 시맨틱은 이 함수 밖에서 처리).
def decide_next_frame(
    state: SimulatorState,
    rng: np.random.Generator,
    m_no: str,
    event_probability: float,
    warning_range: tuple[int, int],
    danger_range: tuple[int, int],
    cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    scenario_pool: tuple[str, ...] = DEFAULT_DEMO_SCENARIO_POOL,
) -> tuple[dict[str, str], SimulatorState]:
    if state.is_idle():
        if state.can_start_new_event() and rng.random() < event_probability:
            scenario = str(rng.choice(scenario_pool))
            target_circuit = int(rng.integers(MIN_CIRCUIT, MAX_CIRCUIT + 1))
            warning_frames = int(rng.integers(warning_range[0], warning_range[1]))
            danger_frames = int(rng.integers(danger_range[0], danger_range[1]))
            seed = int(rng.integers(0, 2**31 - 1))

            started = SimulatorState.start_event(scenario, target_circuit, warning_frames, danger_frames, seed)
            series = build_event_series(scenario, m_no, warning_frames, danger_frames, target_circuit, seed)
            params = build_frame_params(series, started.sequence_index())
            return params, started.advanced()

        idle_seed = int(rng.integers(0, 2**31 - 1))
        series = build_panel_frame_series("NORMAL", m_no, samples=1, seed=idle_seed, target_circuit=MIN_CIRCUIT)
        params = build_frame_params(series, 0)
        return params, state.advanced()

    # IN_EVENT
    assert state.scenario is not None and state.targetCircuit is not None
    assert state.warningFrames is not None and state.dangerFrames is not None and state.seed is not None
    series = build_event_series(state.scenario, m_no, state.warningFrames, state.dangerFrames, state.targetCircuit, state.seed)
    params = build_frame_params(series, state.sequence_index())
    return params, state.advanced(cooldown_frames)


# heartbeat 대상 분전반들에 NORMAL 1프레임씩 전송 - state.json을 읽거나 쓰지 않는 완전 무상태 경로.
# 개별 실패는 로그만 남기고 다음 mNo로 넘어간다(한 분전반의 실패가 나머지 heartbeat나 90001 처리를 막지 않는다).
def send_heartbeats(base_url: str, mnos: tuple[str, ...], timeout_s: float, rng: np.random.Generator) -> None:
    for mno in mnos:
        seed = int(rng.integers(0, 2**31 - 1))
        series = build_panel_frame_series("NORMAL", mno, samples=1, seed=seed, target_circuit=MIN_CIRCUIT)
        params = build_frame_params(series, 0)
        try:
            result = client.send_frame(base_url, params, timeout_s=timeout_s)
            print(f"[heartbeat] {mno} -> {result.get('message', 'OK')}")
        except client.FrameSendError as e:
            print(f"[heartbeat] {mno} 전송 실패 - {e}", file=sys.stderr)


# 1회 실행의 전체 흐름 - state 로드 -> 다음 프레임 결정 -> 전송 -> 성공 시에만 state 저장 -> heartbeat 전송.
# 반환값은 90001 처리 성공 여부(0/1은 main()의 exit code로 이어짐, 실패해도 예외를 던지지 않는다) - heartbeat
# 결과는 이 반환값에 영향을 주지 않는다(요구사항: 실패해도 로그만, 90001 판정을 막지 않는다).
def run_once(
    base_url: str,
    m_no: str,
    state_path: str,
    event_probability: float = DEFAULT_EVENT_PROBABILITY,
    warning_range: tuple[int, int] = DEFAULT_WARNING_FRAMES_RANGE,
    danger_range: tuple[int, int] = DEFAULT_DANGER_FRAMES_RANGE,
    cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    scenario_pool: tuple[str, ...] = DEFAULT_DEMO_SCENARIO_POOL,
    heartbeat_mnos: tuple[str, ...] = HEARTBEAT_MNOS,
    timeout_s: float = 5.0,
    rng: np.random.Generator | None = None,
) -> bool:
    rng = rng if rng is not None else np.random.default_rng()
    state = load_state(state_path)

    params, next_state = decide_next_frame(state, rng, m_no, event_probability, warning_range, danger_range, cooldown_frames, scenario_pool)
    # 로그 라벨은 "이번에 실제로 보낸 프레임"을 가리켜야 하므로 advance 전(원본) state 기준으로 만든다 -
    # next_state는 이미 다음 프레임으로 넘어간 상태라 전환 경계에서 라벨이 한 프레임 어긋나 보일 수 있다.
    if state.state == STATE_IN_EVENT:
        label = f"{state.scenario}/{state.phase}#{state.phaseIndex}"
    elif next_state.state == STATE_IN_EVENT:
        label = f"{next_state.scenario}/{next_state.phase}#0 (신규 이벤트 시작)"
    elif state.cooldownRemaining > 0:
        label = f"IDLE/NORMAL (cooldown {state.cooldownRemaining}->{next_state.cooldownRemaining})"
    else:
        label = "IDLE/NORMAL"

    ok = True
    try:
        result = client.send_frame(base_url, params, timeout_s=timeout_s)
        print(f"[continuous] {label} -> {result.get('message', 'OK')}")
        write_state(state_path, next_state)
    except client.FrameSendError as e:
        # 실패 시맨틱: state는 절대 저장하지 않는다(advance 금지, 새 이벤트 선택도 없었던 일이 된다) -
        # 다음 timer 실행이 이번과 정확히 같은 지점부터 다시 시도한다.
        print(f"[continuous] {label} 전송 실패 - {e}", file=sys.stderr)
        ok = False

    send_heartbeats(base_url, heartbeat_mnos, timeout_s, rng)
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scenario_pool is None:
        scenario_pool = DEFAULT_DEMO_SCENARIO_POOL
    else:
        scenario_pool = tuple(name.strip() for name in args.scenario_pool.split(",") if name.strip())
        unknown = [name for name in scenario_pool if name not in EVENT_SCENARIOS]
        if unknown:
            print(f"[continuous] 알 수 없는 시나리오: {unknown} (허용: {EVENT_SCENARIOS})", file=sys.stderr)
            return 2

    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()
    start = time.monotonic()
    ok = run_once(
        base_url=args.base_url,
        m_no=args.m_no,
        state_path=args.state_path,
        event_probability=args.event_probability,
        warning_range=(args.warning_min, args.warning_max),
        danger_range=(args.danger_min, args.danger_max),
        cooldown_frames=args.cooldown_frames,
        scenario_pool=scenario_pool,
        timeout_s=args.timeout_s,
        rng=rng,
    )
    elapsed = time.monotonic() - start
    print(f"[continuous] 종료 - {'성공' if ok else '실패'} ({elapsed:.2f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
