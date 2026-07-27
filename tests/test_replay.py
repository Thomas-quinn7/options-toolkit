"""Offline tests for the realised-vs-implied replay (vol_snapshots/replay.py).

Synthetic snapshot histories are generated from a GBM with KNOWN realised vol,
quoted at a KNOWN implied vol, so the whole pipeline - entry selection, daily
hedging at entry IV, settlement, and the per-position gamma-P&L theory - is
pinned end to end with no network. Daily hedging leaves real discrete-hedging
noise per position, so the sign tests average a handful of seeds, exactly as
the write-up says the real replay must as history accumulates.

Run:  python -m pytest tests/test_replay.py -q
"""

import datetime as dt

import numpy as np
import pandas as pd

from vol_snapshots import replay as R
from vol_snapshots.capture import COLUMNS
from market_making.mm_sim import bs_price

S0, RATE = 100.0, 0.03
DAY0 = dt.date(2026, 7, 1)
EXPIRY_DAYS = 35


def make_history(sigma_real, sigma_impl, n_days, seed, ticker="SYN",
                 decoy_expiry_days=None):
    """Daily snapshot frames for one ticker: GBM spot at ``sigma_real``,
    quoted tight at ``sigma_impl`` (q=0). ``decoy_expiry_days`` adds a second,
    nearer expiry listed FIRST - real chains always have many expiries, and
    position marking must never pick a row from the wrong one."""
    rng = np.random.default_rng(seed)
    expiries = ([DAY0 + dt.timedelta(days=decoy_expiry_days)]
                if decoy_expiry_days else [])
    expiries.append(DAY0 + dt.timedelta(days=EXPIRY_DAYS))
    strikes = np.arange(60.0, 141.0, 1.0)
    frames = []
    S = S0
    for i in range(n_days):
        day = DAY0 + dt.timedelta(days=i)
        if i > 0:
            dt_step = 1.0 / 365.0
            z = rng.standard_normal()
            S = S * np.exp((RATE - 0.5 * sigma_real**2) * dt_step
                           + sigma_real * np.sqrt(dt_step) * z)
        rows = []
        for expiry in expiries:
            tau = max((expiry - day).days / 365.0, 0.0)
            for K in strikes:
                for otype in ("call", "put"):
                    price = float(bs_price(S, K, tau, RATE, sigma_impl, 0.0, otype))
                    half = max(0.002 * price, 0.02)
                    rows.append({
                        "snapshot_date": day.isoformat(), "ticker": ticker,
                        "spot": float(S), "riskfree": RATE,
                        "expiry": expiry.isoformat(), "otype": otype,
                        "strike": float(K), "bid": max(price - half, 0.0),
                        "ask": price + half, "last": price, "volume": 100,
                        "open_interest": 500, "iv_yf": np.nan,
                    })
        frames.append(pd.DataFrame(rows)[COLUMNS])
    return pd.concat(frames, ignore_index=True)


def test_entry_selection_is_atm_and_at_market_vol():
    snaps = make_history(0.2, 0.25, 1, seed=0)
    entry = R.pick_entry(snaps, DAY0.isoformat(), RATE)
    assert entry is not None
    F = S0 * np.exp(RATE * EXPIRY_DAYS / 365.0)
    assert abs(entry["strike"] - F) <= 0.5          # nearest listed strike
    assert abs(entry["sigma_entry"] - 0.25) < 5e-3  # inverted from the quotes
    assert abs(entry["T0"] - EXPIRY_DAYS / 365.0) < 1e-9
    assert abs(entry["div_yield"]) < 5e-3           # q=0 world


def test_entry_refused_when_implied_carry_is_absurd():
    """Distorted call quotes drag the parity forward ~2% off spot over 35
    days - an implied |q| far beyond any real carry. pick_entry must refuse
    the day rather than strike a straddle off a poisoned forward."""
    snaps = make_history(0.2, 0.25, 1, seed=0)
    calls = snaps["otype"] == "call"
    snaps.loc[calls, ["bid", "ask", "last"]] += 2.0
    assert R.pick_entry(snaps, DAY0.isoformat(), RATE) is None


def test_stale_weekend_days_are_dropped():
    snaps = make_history(0.2, 0.25, 3, seed=1)
    # append a "Saturday" serving the same spot/quotes under a new date
    stale = snaps[snaps["snapshot_date"] == snaps["snapshot_date"].max()].copy()
    stale["snapshot_date"] = (DAY0 + dt.timedelta(days=3)).isoformat()
    days = R.ticker_days(pd.concat([snaps, stale], ignore_index=True))
    assert len(days) == 3


def test_open_position_marks_at_market():
    snaps = make_history(0.2, 0.25, 10, seed=2)
    results = R.run_replay(snaps)
    assert len(results) == 1
    x = results[0]
    assert x.status == "open"
    assert x.mark_source == "market_mid"
    assert x.days_held == 9
    assert np.isfinite(x.pnl)


def test_day0_open_pnl_is_zero_despite_other_expiries():
    """Entered at mid and marked at mid the same day => P&L exactly the
    (zero) round trip, even when a nearer decoy expiry is listed first -
    marking must read the position's own expiry."""
    snaps = make_history(0.2, 0.25, 1, seed=3, decoy_expiry_days=7)
    (x,) = R.run_replay(snaps)
    assert x.status == "open"
    assert x.mark_source == "market_mid"
    assert abs(x.pnl) < 1e-9


def test_short_straddle_wins_when_realised_below_implied():
    pnls, theories = [], []
    for seed in range(6):
        snaps = make_history(0.10, 0.30, EXPIRY_DAYS + 1, seed=seed)
        (x,) = R.run_replay(snaps)
        assert x.status == "closed" and x.mark_source == "settled"
        pnls.append(x.pnl)
        theories.append(x.theory_pnl)
    assert np.mean(pnls) > 0
    assert np.mean(theories) > 0
    assert sum(p > 0 for p in pnls) >= 5


def test_short_straddle_loses_when_realised_above_implied():
    pnls = []
    for seed in range(6):
        snaps = make_history(0.50, 0.20, EXPIRY_DAYS + 1, seed=seed)
        (x,) = R.run_replay(snaps)
        pnls.append(x.pnl)
    assert np.mean(pnls) < 0
    assert sum(p < 0 for p in pnls) >= 5


def test_pnl_tracks_gamma_pnl_theory_per_path():
    """Per path, hedged P&L = theory accrual + discrete-hedging noise; with
    daily steps the noise is material but the two must stay close relative to
    the position's premium scale."""
    errs, premium = [], None
    for seed in range(6):
        snaps = make_history(0.15, 0.25, EXPIRY_DAYS + 1, seed=seed)
        (x,) = R.run_replay(snaps)
        errs.append(abs(x.pnl - x.theory_pnl))
        premium = x.premium_mid
    assert np.mean(errs) < 0.35 * premium
    assert x.realised_vol == x.realised_vol         # recorded, not nan


def test_realised_vol_estimate_is_sane():
    vols = []
    for seed in range(8):
        snaps = make_history(0.30, 0.30, EXPIRY_DAYS + 1, seed=seed)
        (x,) = R.run_replay(snaps)
        vols.append(x.realised_vol)
    # 35 daily observations: mean within ~5 vol points of truth
    assert abs(np.mean(vols) - 0.30) < 0.05
