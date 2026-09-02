"""Sensor Simulator CLI. `python -m simulator.run`로 실행한다.

실제 하드웨어처럼 sample 1개당 HTTP GET 요청 1번씩 연속 전송한다. 전송 주기(--interval-ms)는
실제 하드웨어 전송 주기가 아니라 데모용 설정값이다(TBD).
"""

from __future__ import annotations

import argparse
import sys
import time

from training.scenario.registry import SCENARIOS

from . import client
from .frame_builder import build_frame_params, build_panel_frame_series

DEFAULT_SAMPLES = 180
DEFAULT_INTERVAL_MS = 200
DEFAULT_SEED = 42
DEFAULT_TARGET_CIRCUIT = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcGuard Sensor Simulator - /m_noUpload.php HTTP 전송")
    parser.add_argument("--base-url", required=True, help="firesafety-be base URL (예: http://localhost:8080)")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS.keys()))
    parser.add_argument("--m-no", required=True, help="분전반 장비번호 (5자리)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="전송할 프레임(sample) 수")
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS, help="프레임 간 전송 간격(ms, 데모용 설정값)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-circuit", type=int, default=DEFAULT_TARGET_CIRCUIT, help="시나리오를 적용할 회로 번호(1~10)")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    series = build_panel_frame_series(
        scenario_name=args.scenario,
        m_no=args.m_no,
        samples=args.samples,
        seed=args.seed,
        target_circuit=args.target_circuit,
    )

    print(
        f"[simulator] scenario={args.scenario} m_no={args.m_no} target_circuit={args.target_circuit} "
        f"samples={args.samples} interval_ms={args.interval_ms} seed={args.seed} -> {args.base_url}"
    )

    sent = 0
    failed = 0
    for sample_index in range(args.samples):
        params = build_frame_params(series, sample_index)
        try:
            result = client.send_frame(args.base_url, params, timeout_s=args.timeout_s)
            sent += 1
            print(
                f"[{sample_index + 1}/{args.samples}] am{args.target_circuit}={params[f'am{args.target_circuit}']} "
                f"count{args.target_circuit}={params[f'count{args.target_circuit}']} aerror={params['aerror']} "
                f"-> {result.get('message', 'OK')}"
            )
        except client.FrameSendError as e:
            failed += 1
            print(f"[{sample_index + 1}/{args.samples}] 전송 실패 - {e}", file=sys.stderr)

        if sample_index < args.samples - 1:
            time.sleep(args.interval_ms / 1000)

    print(f"[simulator] 완료 - 전송 {sent}건, 실패 {failed}건")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
