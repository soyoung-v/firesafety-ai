"""training/features/window.py 테스트."""

import pandas as pd

from training.features.window import build_windows


def _make_run(run_id: str, scenario: str, n_samples: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": run_id,
            "scenario": scenario,
            "sample_index": range(n_samples),
            "current": [5.0] * n_samples,
            "arc_count": [0] * n_samples,
        }
    )


def test_exact_60_samples_produces_one_window():
    df = _make_run("R-1", "NORMAL", 60)
    windows = build_windows(df, window_size=60, stride=60)
    assert len(windows) == 1
    assert len(windows[0].rows) == 60
    assert windows[0].window_start == 0
    assert windows[0].window_end == 59


def test_tail_remainder_is_dropped():
    df = _make_run("R-1", "NORMAL", 130)  # 60+60+10
    windows = build_windows(df, window_size=60, stride=60)
    assert len(windows) == 2
    assert sum(len(w.rows) for w in windows) == 120


def test_windows_never_cross_run_boundary():
    df = pd.concat([_make_run("A", "NORMAL", 70), _make_run("B", "ARC", 70)], ignore_index=True)
    windows = build_windows(df, window_size=60, stride=60)
    run_ids_seen = {w.run_id for w in windows}
    assert run_ids_seen == {"A", "B"}
    for w in windows:
        assert (w.rows["run_id"] == w.run_id).all()


def test_window_index_and_bounds_sequential():
    df = _make_run("R-1", "NORMAL", 180)
    windows = build_windows(df, window_size=60, stride=60)
    assert [w.window_index for w in windows] == [0, 1, 2]
    assert [(w.window_start, w.window_end) for w in windows] == [(0, 59), (60, 119), (120, 179)]


def test_stride_smaller_than_window_creates_overlap():
    df = _make_run("R-1", "NORMAL", 120)
    windows = build_windows(df, window_size=60, stride=30)
    assert len(windows) == 3  # start=0,30,60 (60+60<=120 fails at start=60? 60+60=120<=120 ok)
