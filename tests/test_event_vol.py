"""Offline tests for the event-vol study (vol_snapshots/event_vol.py).

A synthetic history injects a KNOWN event - vol builds up, the term structure
inverts, then vol crushes with a spot gap - alongside a flat control that must
stay silent. Detection, the implied-event-move decomposition, and the recorded
window are all pinned.

Run:  python -m pytest tests/test_event_vol.py -q
"""

import numpy as np
import pandas as pd

from vol_snapshots import event_vol as EV
from vol_snapshots.fit_history import HISTORY_COLUMNS

BASE_VOL = 0.25


def _row(day_offset, ticker, spot, vol30, slope=0.0, vol182=BASE_VOL):
    row = {c: np.nan for c in HISTORY_COLUMNS}
    row.update({
        "snapshot_date": (pd.Timestamp("2026-08-03")
                          + pd.Timedelta(days=day_offset)).date().isoformat(),
        "ticker": ticker, "spot": spot, "riskfree": 0.04, "n_expiries": 8,
        "n_quotes": 500, "rho": -0.4, "eta": 1.0, "gamma": 0.4,
        "atm_vol_30d": vol30, "atm_vol_91d": vol30, "atm_vol_182d": vol182,
        "term_slope": slope, "min_butterfly_g": 0.3, "calendar_min_gap": 1e-4,
    })
    return row


def event_history():
    """Ten days: flat, then a 5-day vol build-up into a crush with a 4% gap."""
    rows = []
    spot = 100.0
    ramp = {5: 0.30, 6: 0.34, 7: 0.39, 8: 0.45}       # build-up
    for i in range(10):
        if i == 9:                                    # crush day: gap + reset
            spot *= np.exp(0.04)
            vol30 = 0.26
        else:
            vol30 = ramp.get(i, BASE_VOL)
        rows.append(_row(i, "SYN", spot, vol30, slope=BASE_VOL - vol30))
        spot *= np.exp(0.001)                         # drift between days
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def flat_history():
    rows = [_row(i, "CTRL", 100.0 * np.exp(0.002 * i), BASE_VOL)
            for i in range(10)]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def test_flat_history_detects_nothing():
    assert EV.run_study(flat_history()) == []


def test_event_is_detected_once_with_correct_numbers():
    events = EV.run_study(event_history())
    assert len(events) == 1
    e = events[0]
    assert e.ticker == "SYN"
    assert e.pre_vol_30d == 0.45 and e.post_vol_30d == 0.26
    assert abs(e.crush - 0.19) < 1e-9
    # implied move from the pre-event term structure, 182d as the base
    expected = np.sqrt((0.45**2 - BASE_VOL**2) * 30 / 365)
    assert abs(e.implied_event_move - expected) < 1e-6
    # realized move is the crush-day gap (4% jump + 0.1% drift accrued)
    assert abs(e.realized_move - 0.041) < 1e-3
    # window is aligned at the crush day and covers the build-up
    assert 0 in e.window_offsets and min(e.window_offsets) <= -4
    assert max(e.window_vol_30d) == 0.45


def test_small_dips_are_not_events():
    """A 2-vol-pt wobble (below both thresholds) must not fire."""
    rows = [_row(0, "SYN", 100.0, 0.25), _row(1, "SYN", 100.2, 0.23)]
    assert EV.run_study(pd.DataFrame(rows, columns=HISTORY_COLUMNS)) == []


def test_implied_event_move_floors_at_zero():
    # upward-sloping term structure: no excess short-dated variance
    assert EV.implied_event_move(0.18, 0.22) == 0.0
