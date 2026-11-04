"""Offline tests for vol_snapshots/auto_capture.py and the capture meta file.

The runner is fired every 30 minutes and must decide on its own whether to do
anything. These pin the decisions (phase windows in New York time, the
intraday -> close upgrade, legacy days staying final) and the meta round
trip that lets the studies tell intraday snapshots from closes. No network.

Run:  python -m pytest tests/test_auto_capture.py -q
"""

import datetime as dt

import pandas as pd

from vol_snapshots import auto_capture as A
from vol_snapshots import capture as C

NY = A.NY


def ny(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=NY)


def test_capture_phase_windows_in_new_york_time():
    wed = (2026, 11, 4)
    assert A.capture_phase(ny(*wed, 9, 45)) is None          # pre-/just-open
    assert A.capture_phase(ny(*wed, 10, 0)) == "intraday"
    assert A.capture_phase(ny(*wed, 16, 14)) == "intraday"   # quotes still delayed
    assert A.capture_phase(ny(*wed, 16, 15)) == "close"
    assert A.capture_phase(ny(*wed, 23, 59)) == "close"
    assert A.capture_phase(ny(2026, 11, 5, 0, 30)) is None   # overnight
    assert A.capture_phase(ny(2026, 11, 7, 18, 0)) is None   # Saturday


def test_irish_small_hours_are_still_the_previous_us_session():
    """01:00 Dublin on Thursday is 20:00 New York on Wednesday: a close
    capture filed under Wednesday's session date, not Thursday's."""
    dublin = dt.datetime(2026, 11, 5, 1, 0, tzinfo=dt.timezone.utc)  # GMT in Nov
    as_ny = dublin.astimezone(NY)
    assert A.capture_phase(as_ny) == "close"
    assert as_ny.date().isoformat() == "2026-11-04"


def test_needs_capture_upgrades_intraday_only():
    assert A.needs_capture(False, None, "intraday")
    assert A.needs_capture(False, None, "close")
    intra = {"phase": "intraday"}
    close = {"phase": "close"}
    assert A.needs_capture(True, intra, "close")              # upgrade to the close
    assert not A.needs_capture(True, intra, "intraday")       # once per phase
    assert not A.needs_capture(True, close, "close")
    assert not A.needs_capture(True, close, "intraday")
    assert not A.needs_capture(True, None, "close")           # legacy file is final


def test_plan_picks_missing_and_upgradable_tickers():
    meta = {"SPY": {"phase": "intraday"}, "QQQ": {"phase": "close"}}
    todo = A.plan(["SPY", "QQQ", "IWM", "OLD"], ["spy", "QQQ", "OLD"], meta, "close")
    assert todo == ["SPY", "IWM"]
    assert A.plan(["SPY", "QQQ", "IWM"], ["SPY", "QQQ"], meta, "intraday") == ["IWM"]


def test_capture_meta_round_trip_and_phases(tmp_path):
    root = str(tmp_path)
    day = "2026-11-04"
    for t in ("SPY", "AAPL", "LEGACY"):
        p = tmp_path / day / f"{t}.csv.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    C.write_capture_meta(root, day, "spy", "intraday", "2026-11-04T16:00:00+00:00")
    C.write_capture_meta(root, day, "AAPL", "close", "2026-11-04T21:30:00+00:00")
    meta = C.read_capture_meta(root, day)
    assert meta["SPY"]["phase"] == "intraday" and meta["AAPL"]["phase"] == "close"
    C.write_capture_meta(root, day, "SPY", "close", "2026-11-04T21:31:00+00:00")
    assert C.read_capture_meta(root, day)["SPY"]["phase"] == "close"   # upgraded
    ph = C.load_capture_phases(root).set_index("ticker")["phase"]
    assert ph.to_dict() == {"AAPL": "close", "LEGACY": "close", "SPY": "close"}
    assert C.read_capture_meta(root, "2026-01-01") == {}
    assert not C.META_FILE.endswith(".csv.gz")   # never read as a snapshot


def test_run_is_silent_outside_the_windows(monkeypatch):
    """Outside market windows run() must return before importing the capture
    layer or touching the network."""
    import sys
    monkeypatch.delitem(sys.modules, "vol_snapshots.capture", raising=False)
    assert A.run(now_ny=ny(2026, 11, 7, 12, 0)) == 0            # Saturday
    assert A.run(now_ny=ny(2026, 11, 4, 6, 0)) == 0             # pre-open
    assert "vol_snapshots.capture" not in sys.modules
