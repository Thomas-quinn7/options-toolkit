"""Offline tests for the daily surface-fitting pipeline (vol_snapshots/fit_history.py).

A synthetic snapshot day is generated from a KNOWN arbitrage-free SSVI surface
priced into bid-ask option quotes in the capture schema, so the whole pipeline
- forward from put-call parity, IV inversion from prices, OTM slice building,
global band fit, no-arb diagnostics, history upsert - is pinned end to end
with no network.

Run:  python -m pytest tests/test_fit_history.py -q
"""

import datetime as dt

import numpy as np
import pandas as pd

from vol_snapshots import fit_history as F
from pricing_and_vol_surface import vol_surface as V
from vol_snapshots.capture import COLUMNS

TRUE = V.SSVIParams(rho=-0.4, eta=1.0, gamma=0.4)
S0, R, ATM_VOL = 100.0, 0.03, 0.20
SNAP_DATE = "2026-07-01"
MATURITIES = [0.06, 0.15, 0.35, 0.70]     # straddle the 30/91/182d horizons


def synthetic_day(ticker="SYN", spot=S0, r=R, true=TRUE) -> pd.DataFrame:
    """One (day, ticker) chain in the capture schema, priced off ``true``.

    Both a call and a put row at every strike (parity-consistent, zero
    dividend so F = S*e^{rT}), with a vega-scaled bid-ask spread that widens
    in the wings - the shape the band fit is built for.
    """
    day = dt.date.fromisoformat(SNAP_DATE)
    rows = []
    for T in MATURITIES:
        theta = ATM_VOL**2 * T
        expiry = (day + dt.timedelta(days=round(T * 365))).isoformat()
        F_true = spot * np.exp(r * T)
        for k in np.linspace(-0.45, 0.35, 31):
            K = F_true * np.exp(k)
            iv = float(np.sqrt(V.ssvi_w(k, theta, true) / T))
            call = V.bs_call(spot, K, T, r, iv)
            put = call - spot + K * np.exp(-r * T)          # parity, q=0
            half = V.bs_vega(spot, K, T, r, iv) * (0.002 + 0.008 * abs(k))
            half = max(float(half), 0.005)
            for otype, price in (("call", call), ("put", put)):
                bid, ask = price - half, price + half
                if bid <= 0:
                    bid, ask = 0.0, max(ask, 0.01)          # one-sided quote
                rows.append({
                    "snapshot_date": SNAP_DATE, "ticker": ticker, "spot": spot,
                    "riskfree": r, "expiry": expiry, "otype": otype,
                    "strike": round(float(K), 4), "bid": round(float(bid), 4),
                    "ask": round(float(ask), 4), "last": round(float(price), 4),
                    "volume": 100, "open_interest": 500, "iv_yf": np.nan,
                })
    return pd.DataFrame(rows)[COLUMNS]


def test_implied_forward_recovers():
    df = synthetic_day()
    T = MATURITIES[1]
    day = dt.date.fromisoformat(SNAP_DATE)
    expiry = (day + dt.timedelta(days=round(T * 365))).isoformat()
    sl = df[df["expiry"] == expiry]
    F_true = S0 * np.exp(R * T)
    F_impl = F.implied_forward(sl, R, T, S0)
    assert abs(F_impl / F_true - 1.0) < 3e-3


def test_build_slice_is_otm_band():
    df = synthetic_day()
    T = MATURITIES[2]
    day = dt.date.fromisoformat(SNAP_DATE)
    expiry = (day + dt.timedelta(days=round(T * 365))).isoformat()
    built = F.build_slice(df[df["expiry"] == expiry], SNAP_DATE, expiry, S0, R)
    assert built is not None
    assert built["k"][0] < 0.0 < built["k"][-1]          # straddles ATM
    assert np.all(built["w_ask"] >= built["w_bid"])      # genuine intervals
    assert abs(built["T"] - T) < 2 / 365
    # theta ~ true ATM total variance
    assert abs(built["theta"] - ATM_VOL**2 * T) < 0.15 * ATM_VOL**2 * T


def test_build_slice_rejects_absurd_implied_carry():
    """Stale quotes that push the parity forward far off spot imply a huge
    dividend/borrow; the slice must be refused, not fit. (Day one of real
    data produced AAPL q=-37% from weekend marks - this pins the gate.)"""
    df = synthetic_day()
    T = MATURITIES[2]
    day = dt.date.fromisoformat(SNAP_DATE)
    expiry = (day + dt.timedelta(days=round(T * 365))).isoformat()
    sl = df[df["expiry"] == expiry].copy()
    calls = sl["otype"] == "call"
    sl.loc[calls, ["bid", "ask", "last"]] += 5.0     # C-P inflated -> F ~ +5%
    assert F.build_slice(sl, SNAP_DATE, expiry, S0, R) is None


def test_fit_day_recovers_and_is_arb_free():
    row = F.fit_day_ticker(synthetic_day())
    assert row is not None
    assert row["n_expiries"] == len(MATURITIES)
    assert abs(row["rho"] - TRUE.rho) < 0.15
    assert abs(row["atm_vol_30d"] - ATM_VOL) < 0.01
    assert abs(row["atm_vol_182d"] - ATM_VOL) < 0.01
    assert abs(row["term_slope"]) < 0.01                 # flat true term structure
    assert row["min_butterfly_g"] >= 0.0
    assert row["calendar_min_gap"] >= 0.0


def test_chart_tickers_caps_and_prioritises():
    # narrower than the cap: everything charts, priority names first
    got = F.chart_tickers(["AAPL", "MSFT", "SPY"])
    assert got == ["SPY", "AAPL", "MSFT"]
    # wider than the cap: at most MAX_CHART_SERIES, priority list wins
    wide = ["AAPL", "GLD", "IWM", "MSFT", "NVDA", "QQQ", "SPY", "TLT",
            "TSLA", "VXX", "XLE", "XLF", "XLK"]
    got = F.chart_tickers(wide)
    assert len(got) == F.MAX_CHART_SERIES
    assert got == [t for t in F.CHART_PRIORITY if t in wide]


def test_build_slice_allows_a_discrete_dividend_near_dated():
    """A normal discrete dividend inside a short window must NOT delete the
    slice. q is a continuous-yield abstraction, so a ~40bp drop in the forward
    annualises to |q| ~ 9% at 16 days; the old flat |q| <= 8% gate deleted
    every near-dated expiry of the dividend payers (TLT's whole 16-37d strip),
    and with them the 30d ATM vol the history exists to record."""
    df = synthetic_day()
    T = MATURITIES[0]                                    # ~22 days
    day = dt.date.fromisoformat(SNAP_DATE)
    expiry = (day + dt.timedelta(days=round(T * 365))).isoformat()
    sl = df[df["expiry"] == expiry].copy()
    # Push the forward ~60bp below spot the way a dividend does - through
    # C - P, so put-call parity still holds and the quotes stay honest.
    half = 0.004 * S0
    sl.loc[sl["otype"] == "call", ["bid", "ask", "last"]] -= half
    sl.loc[sl["otype"] == "put", ["bid", "ask", "last"]] += half
    F_impl = F.implied_forward(sl, R, T, S0)
    T_act = (dt.date.fromisoformat(expiry) - day).days / 365.0
    assert abs(F_impl / S0 - 1) < 0.01                   # a plausible dividend
    # ~22 days, so that 60bp annualises well past the old flat |q| bound...
    assert abs(R - np.log(F_impl / S0) / T_act) > F.Q_ABS_MAX
    # ...yet the slice survives, because the gate is on the basis now.
    assert F.build_slice(sl, SNAP_DATE, expiry, S0, R) is not None


def test_fit_is_usable_flags_collapsed_fits():
    theta = ATM_VOL**2 * MATURITIES[-1]
    assert F.fit_is_usable(-0.4, 0.3, 0.001, theta)
    # rho on fit_ssvi_band's own bound: the optimiser ran out of room
    assert not F.fit_is_usable(0.999, 0.3, 0.001, theta)
    assert not F.fit_is_usable(-0.999, 0.3, 0.001, theta)
    # total variance falling with maturity by more than half a theta
    assert not F.fit_is_usable(-0.4, 0.3, -0.9 * theta, theta)
    # a mild wing violation is still readable and stays in the studies
    assert F.fit_is_usable(-0.4, 0.3, -0.15 * theta, theta)
    # butterfly arbitrage in the fitted surface
    assert not F.fit_is_usable(-0.4, -0.01, 0.001, theta)
    assert not F.fit_is_usable(np.nan, 0.3, 0.001, theta)


def test_fit_day_marks_its_fit_usable():
    row = F.fit_day_ticker(synthetic_day())
    assert row["fit_ok"] is True


def test_usable_history_filters_but_csv_keeps_the_bad_row(tmp_path):
    """The degenerate row must stay ON DISK - the no-arb diagnostics are the
    point of the history - while the studies stop reading it."""
    path = str(tmp_path / "hist.csv")
    good = {c: 0 for c in F.HISTORY_COLUMNS}
    good.update(snapshot_date="2026-07-01", ticker="GOOD", atm_vol_30d=0.2,
                rho=-0.4, min_butterfly_g=0.3, calendar_min_gap=0.001,
                atm_vol_182d=0.2, fit_ok=True)
    bad = dict(good, ticker="BAD", rho=0.999, calendar_min_gap=-1.3,
               fit_ok=False)
    F.update_history([good, bad], path)
    hist = F.load_history(path)
    assert len(hist) == 2
    assert list(F.usable_history(hist)["ticker"]) == ["GOOD"]


def test_load_history_backfills_fit_ok_for_legacy_csv(tmp_path):
    """A history written before the flag existed must not be assumed clean:
    the flag is recovered from the diagnostics already on each row."""
    path = str(tmp_path / "legacy.csv")
    legacy = pd.DataFrame([
        dict(snapshot_date="2026-07-30", ticker="OK", spot=1.0, riskfree=0.04,
             n_expiries=5, n_quotes=100, rho=-0.4, eta=1.0, gamma=0.4,
             atm_vol_30d=0.2, atm_vol_91d=0.2, atm_vol_182d=0.2,
             term_slope=0.0, min_butterfly_g=0.3, calendar_min_gap=0.001),
        dict(snapshot_date="2026-07-30", ticker="XLK", spot=1.0, riskfree=0.04,
             n_expiries=6, n_quotes=119, rho=0.999, eta=1.0, gamma=0.4,
             atm_vol_30d=np.nan, atm_vol_91d=np.nan, atm_vol_182d=0.325,
             term_slope=np.nan, min_butterfly_g=0.223,
             calendar_min_gap=-1.275256),
    ])
    legacy.to_csv(path, index=False)
    hist = F.load_history(path)
    assert "fit_ok" in hist.columns
    assert dict(zip(hist["ticker"], hist["fit_ok"])) == {"OK": True, "XLK": False}


def test_history_upsert_dedups(tmp_path):
    path = str(tmp_path / "hist.csv")
    row = {c: 0 for c in F.HISTORY_COLUMNS}
    row.update(snapshot_date="2026-07-01", ticker="SYN", atm_vol_30d=0.2)
    hist = F.update_history([row], path)
    assert len(hist) == 1
    row2 = dict(row, atm_vol_30d=0.25)                   # refit same key
    hist = F.update_history([row2], path)
    assert len(hist) == 1
    assert float(hist["atm_vol_30d"].iloc[0]) == 0.25
    row3 = dict(row, ticker="OTHER")                     # new key appends
    hist = F.update_history([row3], path)
    assert len(hist) == 2
