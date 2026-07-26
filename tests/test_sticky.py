"""Offline tests for the sticky-strike vs sticky-delta study (vol_snapshots/sticky.py).

The core comparisons are pure functions of smile callables, so both regimes
are synthesised EXACTLY (a smile that repositions with moneyness vs one pinned
to strikes) and the study must recover each: demeaned-RMS discriminator and
beta both. The history adapter and the data-threshold gate are pinned on
constructed surface_history rows.

Run:  python -m pytest tests/test_sticky.py -q
"""

import numpy as np
import pandas as pd

from vol_snapshots import sticky as ST
from vol_snapshots.fit_history import HISTORY_COLUMNS

RNG = np.random.default_rng(7)
B, C = -0.5, 2.0          # smile slope and curvature in log-moneyness
BAND = 0.065              # ~1.5 * 0.15 * sqrt(30/365)


def smile(a):
    """A quadratic smile with ATM level ``a`` (shape shared by both worlds)."""
    return lambda k: a + B * np.asarray(k) + C * np.asarray(k) ** 2


def make_pairs(world, n=60, level_noise=0.002, seed=1):
    """Day-pairs from an exact regime: 'delta' keeps the smile a function of
    moneyness; 'strike' keeps vol pinned to strikes (smile shifts by the move)."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        d = rng.choice([-1, 1]) * rng.uniform(0.005, 0.03)
        a_t = 0.15 + rng.normal(0, level_noise)
        a_t1 = 0.15 + rng.normal(0, level_noise)
        sig_t = smile(a_t)
        if world == "delta":
            sig_t1 = smile(a_t1)
        else:  # sticky-strike: today's smile is yesterday's, shifted by d
            sig_t1 = lambda k, a=a_t1, dd=d: smile(a)(np.asarray(k) + dd)
        pairs.append(ST.pair_metrics(sig_t, sig_t1, d, BAND))
    return pairs


def test_sticky_delta_world_is_recovered():
    m = ST.regime_summary(make_pairs("delta"))
    assert m["rms_delta"] < 0.5 * m["rms_strike"]      # primary discriminator
    assert 0.8 < m["beta"] < 1.2                       # classic regression


def test_sticky_strike_world_is_recovered():
    m = ST.regime_summary(make_pairs("strike"))
    assert m["rms_strike"] < 0.5 * m["rms_delta"]
    assert -0.3 < m["beta"] < 0.3


def _history_rows(n_days, atm=0.15, rho=-0.55, eta=1.0, gamma=0.4, seed=2):
    """surface_history rows for one ticker: constant SSVI shape (an exactly
    sticky-delta world up to ATM level noise), spot moving 0.5% a day."""
    rng = np.random.default_rng(seed)
    rows, spot = [], 500.0
    for i in range(n_days):
        spot *= np.exp(rng.choice([-1, 1]) * 0.005)
        row = {c: np.nan for c in HISTORY_COLUMNS}
        row.update({
            "snapshot_date": (pd.Timestamp("2026-08-03")
                              + pd.Timedelta(days=i)).date().isoformat(),
            "ticker": "SYN", "spot": round(spot, 4), "riskfree": 0.04,
            "n_expiries": 8, "n_quotes": 500, "rho": rho, "eta": eta,
            "gamma": gamma, "atm_vol_30d": atm + rng.normal(0, 0.001),
            "atm_vol_91d": atm, "atm_vol_182d": atm, "term_slope": 0.0,
            "min_butterfly_g": 0.3, "calendar_min_gap": 1e-4,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def test_smile_from_row_reproduces_fit():
    row = _history_rows(1).iloc[0]
    sig = ST.smile_from_row(row)
    assert abs(float(sig(0.0)) - row["atm_vol_30d"]) < 1e-12
    # negative rho -> puts richer: vol falls as k rises through ATM
    assert float(sig(-0.05)) > float(sig(0.05))


def test_gate_waits_below_min_pairs():
    studies, waiting = ST.run_study(_history_rows(5))
    assert studies == []
    assert len(waiting) == 1 and waiting[0].startswith("SYN")


def test_full_pipeline_on_constant_shape_history_reads_sticky_delta():
    """A constant-(rho,eta,gamma) SSVI history IS a sticky-delta world (the
    smile is the same function of moneyness every day, up to ATM noise); the
    end-to-end study over the history rows must read it that way."""
    studies, waiting = ST.run_study(_history_rows(25))
    assert waiting == []
    (s,) = studies
    assert s.summary["n_pairs"] >= ST.MIN_PAIRS
    assert s.summary["rms_delta"] < s.summary["rms_strike"]
