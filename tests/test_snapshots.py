"""Offline tests for the option-chain snapshot capture (vol_snapshots/).

Everything network-facing is excluded; what IS tested is what the future
surface-dynamics work will depend on: the schema, the normalization of raw
yfinance chain frames into it, and the write/load round-trip across days.

Run:  python -m pytest tests/test_snapshots.py -q
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vol_snapshots import capture
from vol_snapshots.capture import (
    COLUMNS,
    load_snapshots,
    normalize_chain,
    save_snapshot,
)


class _FixedDate:
    """Stand-in for the ``datetime`` module whose ``date.today()`` is pinned,
    so the non-trading-day guard can be tested on a chosen weekday or holiday
    without waiting for one to come round.

    Patched over ``capture._dt`` rather than over ``datetime.date`` itself:
    the latter is a shared module attribute, and mutating it would pin
    ``today()`` for the whole test session, not just this module.
    """

    def __init__(self, fixed):
        self.date = type("_D", (), {"today": staticmethod(lambda: fixed)})


def fake_yf_chain(strikes, bids, asks, iv=0.2):
    """A frame shaped like one side of a yfinance option chain."""
    n = len(strikes)
    return pd.DataFrame({
        "contractSymbol": [f"XYZ2601{i}" for i in range(n)],
        "strike": strikes,
        "bid": bids,
        "ask": asks,
        "lastPrice": [(b + a) / 2 for b, a in zip(bids, asks)],
        "volume": [10] * n,
        "openInterest": [100] * n,
        "impliedVolatility": [iv] * n,
        "inTheMoney": [False] * n,
    })


def test_normalize_schema_and_order():
    raw = fake_yf_chain([110.0, 90.0, 100.0], [1.0, 2.0, 1.5], [1.2, 2.4, 1.7])
    out = normalize_chain(raw, ticker="XYZ", spot=100.0, expiry="2026-09-18",
                          otype="call", snapshot_date="2026-07-25", riskfree=0.04)
    assert list(out.columns) == COLUMNS
    assert list(out["strike"]) == [90.0, 100.0, 110.0]      # sorted
    assert (out["ticker"] == "XYZ").all()
    assert (out["otype"] == "call").all()
    assert np.allclose(out["spot"], 100.0)
    assert np.allclose(out["riskfree"], 0.04)


def test_normalize_tolerates_missing_columns_and_bad_rows():
    raw = pd.DataFrame({"strike": [100.0, None, "bad"], "bid": [1.0, 2.0, 3.0]})
    out = normalize_chain(raw, ticker="XYZ", spot=100.0, expiry="2026-09-18",
                          otype="put", snapshot_date="2026-07-25")
    assert len(out) == 1                                     # unparseable strikes dropped
    assert list(out.columns) == COLUMNS                      # schema still exact
    assert np.isnan(out["ask"].iloc[0])                      # absent column -> NaN
    assert np.isnan(out["riskfree"].iloc[0])                 # default riskfree -> NaN


def test_save_and_load_round_trip(tmp_path):
    root = str(tmp_path)
    for day in ("2026-07-24", "2026-07-25"):
        for tkr in ("AAA", "BBB"):
            raw = fake_yf_chain([95.0, 105.0], [1.0, 0.5], [1.2, 0.7])
            df = normalize_chain(raw, ticker=tkr, spot=100.0, expiry="2026-09-18",
                                 otype="call", snapshot_date=day, riskfree=0.04)
            save_snapshot(df, root)
    allrows = load_snapshots(root)
    assert list(allrows.columns) == COLUMNS
    assert len(allrows) == 2 * 2 * 2
    assert set(allrows["snapshot_date"]) == {"2026-07-24", "2026-07-25"}
    only_a = load_snapshots(root, tickers=["aaa"])           # case-insensitive filter
    assert set(only_a["ticker"]) == {"AAA"}
    assert len(only_a) == 4


def test_save_rejects_wrong_schema(tmp_path):
    with pytest.raises(ValueError):
        save_snapshot(pd.DataFrame({"strike": [1.0]}), str(tmp_path))


def test_load_empty_root_is_empty_frame(tmp_path):
    out = load_snapshots(str(tmp_path / "nothing_here"))
    assert list(out.columns) == COLUMNS
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# Non-trading-day guard                                                        #
# --------------------------------------------------------------------------- #
def test_capture_skips_weekend_without_touching_the_network(monkeypatch, capsys):
    """Saturday short-circuits before any fetch: no network call at all."""
    monkeypatch.setattr(capture, "_dt", _FixedDate(dt.date(2026, 8, 1)))
    monkeypatch.setattr(capture, "last_session_date",
                        lambda *a, **k: pytest.fail("should not query the market"))
    monkeypatch.setattr(capture, "capture_ticker",
                        lambda *a, **k: pytest.fail("should not capture"))
    capture.main([])
    assert "weekend" in capsys.readouterr().out.lower()


def test_capture_skips_market_holiday(monkeypatch, capsys):
    """A holiday is a WEEKDAY, so the weekend guard alone would have written
    a stale snapshot. The last-session check is what refuses it: on Labor Day
    2026-09-07 the freshest bar is still Friday the 4th."""
    monkeypatch.setattr(capture, "_dt", _FixedDate(dt.date(2026, 9, 7)))
    monkeypatch.setattr(capture, "last_session_date",
                        lambda *a, **k: dt.date(2026, 9, 4))
    monkeypatch.setattr(capture, "capture_ticker",
                        lambda *a, **k: pytest.fail("should not capture"))
    capture.main([])
    out = capsys.readouterr().out
    assert "not a trading session" in out and "2026-09-04" in out


def test_capture_proceeds_on_a_normal_session(monkeypatch):
    monkeypatch.setattr(capture, "_dt", _FixedDate(dt.date(2026, 9, 8)))
    monkeypatch.setattr(capture, "last_session_date",
                        lambda *a, **k: dt.date(2026, 9, 8))
    monkeypatch.setattr(capture, "_fetch_riskfree", lambda: 0.04)
    seen = []
    monkeypatch.setattr(capture, "capture_ticker",
                        lambda t, d, r, **k: seen.append((t, d)))
    capture.main(["SPY"])
    assert seen == [("SPY", "2026-09-08")]


def test_capture_proceeds_when_the_session_check_is_unavailable(monkeypatch):
    """Network trouble must not cost a real trading day - fail open."""
    monkeypatch.setattr(capture, "_dt", _FixedDate(dt.date(2026, 9, 8)))
    monkeypatch.setattr(capture, "last_session_date", lambda *a, **k: None)
    monkeypatch.setattr(capture, "_fetch_riskfree", lambda: 0.04)
    seen = []
    monkeypatch.setattr(capture, "capture_ticker",
                        lambda t, d, r, **k: seen.append(t))
    capture.main(["SPY"])
    assert seen == ["SPY"]


def test_force_overrides_the_guard(monkeypatch):
    monkeypatch.setattr(capture, "_dt", _FixedDate(dt.date(2026, 8, 1)))
    monkeypatch.setattr(capture, "_fetch_riskfree", lambda: 0.04)
    seen = []
    monkeypatch.setattr(capture, "capture_ticker",
                        lambda t, d, r, **k: seen.append(t))
    capture.main(["SPY", "--force"])
    assert seen == ["SPY"]
