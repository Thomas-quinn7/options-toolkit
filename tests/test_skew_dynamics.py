"""Offline tests for the skew-dynamics study (vol_snapshots/skew_dynamics.py).

Synthetic histories inject KNOWN dynamics - the SSVI rho wired to the market's
cumulative return (skew steepens when the market falls) and ATM vol wired to
the ticker's own price (leverage effect) - and the study's regressions must
recover both signs, with the data gate pinned as well.

Run:  python -m pytest tests/test_skew_dynamics.py -q
"""

import numpy as np
import pandas as pd

from vol_snapshots import skew_dynamics as SD
from vol_snapshots.fit_history import HISTORY_COLUMNS


def make_history(n_days=30, seed=5, lev=-1.5, rho_sens=3.0):
    """SPY rows with +-1% daily moves; rho tracks the cumulative market
    return (scaled by ``rho_sens``) and ATM vol tracks own price (``lev``)."""
    rng = np.random.default_rng(seed)
    rows, spot, s0 = [], 500.0, 500.0
    for i in range(n_days):
        spot *= np.exp(rng.choice([-1.0, 1.0]) * 0.01)
        level = np.log(spot / s0)
        row = {c: np.nan for c in HISTORY_COLUMNS}
        row.update({
            "snapshot_date": (pd.Timestamp("2026-08-03")
                              + pd.Timedelta(days=i)).date().isoformat(),
            "ticker": "SPY", "spot": round(spot, 4), "riskfree": 0.04,
            "n_expiries": 8, "n_quotes": 500,
            "rho": float(np.clip(-0.5 + rho_sens * level, -0.95, 0.95)),
            "eta": 1.0, "gamma": 0.4,
            "atm_vol_30d": float(np.clip(0.18 + lev * level, 0.05, 1.0)),
            "atm_vol_91d": 0.18, "atm_vol_182d": 0.18, "term_slope": 0.0,
            "min_butterfly_g": 0.3, "calendar_min_gap": 1e-4,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def test_ols_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 200)
    y = -2.0 * x + rng.normal(0, 0.001, 200) + 0.005   # intercept must not bias
    beta, se = SD.ols(x, y)
    assert abs(beta + 2.0) < 3 * se


def test_gate_waits_below_min_pairs():
    studies, waiting = SD.run_study(make_history(n_days=6))
    assert studies == []
    assert len(waiting) == 1 and waiting[0].startswith("SPY")


def test_known_dynamics_are_recovered():
    studies, waiting = SD.run_study(make_history(n_days=40))
    assert waiting == []
    (s,) = studies
    assert s.n_pairs >= SD.MIN_PAIRS
    # skew steepens when the market falls: rho rises with the market, the
    # put-minus-call skew measure falls -> negative regression slope
    assert s.beta_skew < 0
    assert s.beta_skew + 2 * s.se_skew < 0             # significantly so
    # leverage effect injected at -1.5 vol per unit log return
    assert abs(s.beta_lev - (-1.5)) < 2 * s.se_lev + 0.15


def test_skew_measure_sign_matches_rho():
    steep = make_history(n_days=1, rho_sens=0.0).iloc[0]
    assert SD.skew_measure(steep) > 0                  # rho<0 -> puts richer
