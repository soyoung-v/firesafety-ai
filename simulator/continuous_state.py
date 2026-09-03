"""Continuous Demo Simulator의 상태(state.json) 스키마 + 로드/저장.

한 시점에 진행 중인 위험 이벤트는 최대 1개다(IDLE 또는 IN_EVENT). 이벤트는
NORMAL(공유 baseline) -> WARNING(ramp) -> DANGER(hold) -> NORMAL cooldown -> IDLE(신규 이벤트
시작 가능) 순서로 진행하며, phase/phaseIndex로 지금 어디까지 왔는지를 기록해 매 실행(one-shot
process)이 끝나도 다음 실행이 이어받을 수 있게 한다.

Cooldown은 새 상태(state 값)를 추가하지 않고 IDLE에 `cooldownRemaining` 카운터 하나만 더해
표현한다 - DANGER 마지막 프레임 이후 이 값을 DEFAULT_COOLDOWN_FRAMES로 채워두면, 그 값이 0이 될
때까지는 매 실행이 NORMAL 프레임만 보내고 새 이벤트 시작 판단 자체를 하지 않는다(decide_next_frame
쪽 게이팅). 상태 종류를 늘리지 않아 전이 로직이 단순하게 유지된다.

프로세스가 중간에 죽어도 state.json이 절반만 써진 채로 남지 않도록 temp file
작성 후 atomic rename으로 저장한다(write_state). 파일이 없거나 깨져 있으면
IDLE 상태로 안전하게 취급한다(load_state) - 예외를 던져서 timer 실행 자체를
실패시키지 않는다.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

STATE_IDLE = "IDLE"
STATE_IN_EVENT = "IN_EVENT"

PHASE_WARNING = "WARNING"
PHASE_DANGER = "DANGER"

_VALID_STATES = (STATE_IDLE, STATE_IN_EVENT)
_VALID_PHASES = (PHASE_WARNING, PHASE_DANGER)

# 위험 이벤트 종료 직후 다음 이벤트가 바로 시작되지 않도록 두는 최소 NORMAL 구간(frame 수) -
# 60초 간격 기준 약 30~60분 상당. CLI(--cooldown-frames)로 조정 가능.
DEFAULT_COOLDOWN_FRAMES = 45


@dataclass
class SimulatorState:
    state: str = STATE_IDLE
    scenario: str | None = None
    targetCircuit: int | None = None
    phase: str | None = None
    phaseIndex: int = 0
    warningFrames: int | None = None
    dangerFrames: int | None = None
    seed: int | None = None
    cooldownRemaining: int = 0  # IDLE일 때만 의미 있음 - >0이면 새 이벤트 시작 판단 자체를 건너뛴다
    updatedAt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_idle(self) -> bool:
        return self.state == STATE_IDLE

    def can_start_new_event(self) -> bool:
        return self.is_idle() and self.cooldownRemaining <= 0

    # 이벤트 시작 시 - 이번 실행에서 HTTP 전송이 성공해야만 이 값으로 저장한다(호출부 책임)
    @staticmethod
    def start_event(scenario: str, target_circuit: int, warning_frames: int, danger_frames: int, seed: int) -> "SimulatorState":
        return SimulatorState(
            state=STATE_IN_EVENT,
            scenario=scenario,
            targetCircuit=target_circuit,
            phase=PHASE_WARNING,
            phaseIndex=0,
            warningFrames=warning_frames,
            dangerFrames=danger_frames,
            seed=seed,
        )

    # 이번 프레임 전송이 성공했을 때만 호출 - 다음 상태로 advance한 새 인스턴스를 반환한다(원본은 불변으로 취급).
    # cooldown_frames는 이벤트가 이번 호출로 막 끝났을 때만 쓰인다(그 외에는 무시).
    def advanced(self, cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES) -> "SimulatorState":
        if self.is_idle():
            # cooldown 중이면(cooldownRemaining > 0) 매 프레임 하나씩 소진, 다 쓰면 신규 이벤트 시작 가능
            return SimulatorState(cooldownRemaining=max(self.cooldownRemaining - 1, 0))

        assert self.phase in _VALID_PHASES and self.warningFrames is not None and self.dangerFrames is not None
        next_index = self.phaseIndex + 1

        if self.phase == PHASE_WARNING:
            if next_index >= self.warningFrames:
                return SimulatorState(
                    state=STATE_IN_EVENT,
                    scenario=self.scenario,
                    targetCircuit=self.targetCircuit,
                    phase=PHASE_DANGER,
                    phaseIndex=0,
                    warningFrames=self.warningFrames,
                    dangerFrames=self.dangerFrames,
                    seed=self.seed,
                )
            return SimulatorState(
                state=STATE_IN_EVENT,
                scenario=self.scenario,
                targetCircuit=self.targetCircuit,
                phase=PHASE_WARNING,
                phaseIndex=next_index,
                warningFrames=self.warningFrames,
                dangerFrames=self.dangerFrames,
                seed=self.seed,
            )

        # phase == DANGER
        if next_index >= self.dangerFrames:
            return SimulatorState(cooldownRemaining=cooldown_frames)  # 이벤트 종료 -> IDLE + cooldown 시작
        return SimulatorState(
            state=STATE_IN_EVENT,
            scenario=self.scenario,
            targetCircuit=self.targetCircuit,
            phase=PHASE_DANGER,
            phaseIndex=next_index,
            warningFrames=self.warningFrames,
            dangerFrames=self.dangerFrames,
            seed=self.seed,
        )

    # 이벤트 전체 시퀀스(NORMAL 앵커 1프레임 + WARNING N + DANGER M) 안에서, 지금 보낼 프레임의 절대 인덱스.
    # 앵커 프레임(index 0)은 ramp 시작점 계산용일 뿐 실제로 전송하지 않는다.
    def sequence_index(self) -> int:
        if self.phase == PHASE_WARNING:
            return 1 + self.phaseIndex
        if self.phase == PHASE_DANGER:
            assert self.warningFrames is not None
            return 1 + self.warningFrames + self.phaseIndex
        raise ValueError(f"sequence_index는 IN_EVENT 상태에서만 호출한다: {self}")

    def total_event_frames(self) -> int:
        assert self.warningFrames is not None and self.dangerFrames is not None
        return 1 + self.warningFrames + self.dangerFrames


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 상태 파일이 없거나 JSON이 깨졌거나 스키마가 안 맞으면 IDLE로 안전하게 취급한다(예외를 던지지 않음) -
# 이 판단 하나 때문에 timer 실행 전체가 실패하면 안 된다.
def load_state(path: str) -> SimulatorState:
    if not os.path.exists(path):
        return SimulatorState()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        state = SimulatorState(**{k: v for k, v in raw.items() if k in SimulatorState.__dataclass_fields__})
        if state.state not in _VALID_STATES:
            return SimulatorState()
        if state.state == STATE_IN_EVENT and state.phase not in _VALID_PHASES:
            return SimulatorState()
        return state
    except (json.JSONDecodeError, TypeError, ValueError, OSError):
        return SimulatorState()


# temp file 작성 -> os.replace(atomic rename)로 저장 - 중간에 프로세스가 죽어도 기존 state.json은 그대로 남는다
def write_state(path: str, state: SimulatorState) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    payload = asdict(state)
    payload["updatedAt"] = _now_iso()

    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".state-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
