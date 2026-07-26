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
