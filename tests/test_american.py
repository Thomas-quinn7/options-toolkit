"""Tests for the CRR American pricer (pricing_and_vol_surface/american.py).

The correctness anchor is the European mode converging to closed-form
Black-Scholes; the American-specific facts (call = European when q=0, put
premium positive and increasing in r, deep-ITM put pinned to intrinsic) are
then pinned on top. The final test quantifies the claim the README makes:
for the short-dated OTM quotes the skew scanner consumes, the
early-exercise premium is small next to real bid-ask spreads.

Run:  python -m pytest tests/test_american.py -q
"""

import numpy as np
import pytest

from pricing_and_vol_surface.american import (
    american_iv,
    american_iv_vector,
    crr_price,
    early_exercise_premium,
)
from pricing_and_vol_surface.vol_surface import bs_call, iv_from_price


def bs_put(S, K, T, r, sigma, q=0.0):
    return bs_call(S, K, T, r, sigma, q) - S * np.exp(-q * T) + K * np.exp(-r * T)


# --------------------------------------------------------------------------- #
# Correctness anchor: European binomial -> Black-Scholes                       #
# --------------------------------------------------------------------------- #
def test_european_mode_converges_to_black_scholes():
    for S, K, T, r, sigma, q, otype in [
        (100.0, 105.0, 0.5, 0.04, 0.25, 0.01, "call"),
        (100.0, 90.0, 1.0, 0.02, 0.35, 0.00, "put"),
        (100.0, 100.0, 0.1, 0.05, 0.15, 0.03, "call"),
    ]:
        tree = crr_price(S, K, T, r, sigma, q, otype, "european", n_steps=1000)
        bs = bs_call(S, K, T, r, sigma, q) if otype == "call" else bs_put(S, K, T, r, sigma, q)
        assert abs(tree - bs) < 5e-3, (otype, tree, bs)


def test_expiry_is_intrinsic():
    assert crr_price(110.0, 100.0, 0.0, 0.05, 0.2, otype="call") == pytest.approx(10.0)
    assert crr_price(110.0, 100.0, 0.0, 0.05, 0.2, otype="put") == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# American-exercise facts                                                     #
# --------------------------------------------------------------------------- #
def test_american_call_without_dividends_is_european():
    """Merton: never exercise an American call on a non-dividend payer early."""
    for K in (80.0, 100.0, 120.0):
        amer = crr_price(100.0, K, 0.75, 0.05, 0.3, 0.0, "call", "american")
        euro = crr_price(100.0, K, 0.75, 0.05, 0.3, 0.0, "call", "european")
        assert amer == pytest.approx(euro, abs=1e-10)


def test_put_premium_positive_and_increases_with_rates():
    prem_lo = early_exercise_premium(100.0, 100.0, 1.0, 0.02, 0.25, 0.0, "put")
    prem_hi = early_exercise_premium(100.0, 100.0, 1.0, 0.08, 0.25, 0.0, "put")
    assert 0.0 < prem_lo < prem_hi


def test_deep_itm_put_pinned_to_intrinsic():
    price = crr_price(60.0, 100.0, 0.5, 0.08, 0.2, 0.0, "put", "american")
    assert price == pytest.approx(40.0, abs=1e-9)
    # while the European put is worth strictly less than intrinsic here
    assert crr_price(60.0, 100.0, 0.5, 0.08, 0.2, 0.0, "put", "european") < 40.0


def test_american_never_below_european_or_intrinsic():
    Ks = np.linspace(70, 130, 13)
    for otype in ("call", "put"):
        amer = crr_price(100.0, Ks, 0.5, 0.05, 0.3, 0.02, otype, "american")
        euro = crr_price(100.0, Ks, 0.5, 0.05, 0.3, 0.02, otype, "european")
        intr = np.maximum(100.0 - Ks, 0.0) if otype == "call" else np.maximum(Ks - 100.0, 0.0)
        assert np.all(amer >= euro - 1e-10)
        assert np.all(amer >= intr - 1e-10)


# --------------------------------------------------------------------------- #
# IV inversion                                                                #
# --------------------------------------------------------------------------- #
def test_american_iv_round_trip():
    price = crr_price(100.0, 95.0, 0.35, 0.045, 0.32, 0.005, "put", "american", 200)
    iv = american_iv(price, 100.0, 95.0, 0.35, 0.045, 0.005, "put")
    assert iv == pytest.approx(0.32, abs=1e-6)


def test_american_iv_impossible_price_is_nan():
    # below intrinsic for a deep-ITM American put
    assert np.isnan(american_iv(30.0, 60.0, 100.0, 0.5, 0.05, 0.0, "put"))


def test_vector_iv_matches_scalar():
    Ks = np.array([85.0, 95.0, 105.0])
    ivs_true = np.array([0.35, 0.28, 0.24])
    prices = crr_price(100.0, Ks, 0.25, 0.045, ivs_true, 0.0, "put", "american", 200)
    ivs = american_iv_vector(prices, 100.0, Ks, 0.25, 0.045, 0.0, "put")
    assert np.allclose(ivs, ivs_true, atol=1e-6)


# --------------------------------------------------------------------------- #
# The quantified caveat the README leans on                                   #
# --------------------------------------------------------------------------- #
def test_otm_early_exercise_premium_is_small():
    """For the skew scanner's typical quotes (90%-moneyness puts, 30-60 DTE,
    r ~ 4.5%), the American premium is under ~2% of the option price and the
    IV error from inverting an American price with the European inverter is
    under one vol point - far inside real OTM bid-ask spreads (>= 5-10% of
    mid). This is what justifies European inversion as the fast default."""
    worst_frac, worst_iv_err = 0.0, 0.0
    for dte in (30, 45, 60):
        for iv in (0.20, 0.35, 0.60):
            T, K = dte / 365.0, 90.0
            amer = crr_price(100.0, K, T, 0.045, iv, 0.0, "put", "american", 400)
            euro = crr_price(100.0, K, T, 0.045, iv, 0.0, "put", "european", 400)
            if euro < 0.02:            # sub-2-cent quotes fail liquidity gates anyway
                continue
            worst_frac = max(worst_frac, (amer - euro) / euro)
            iv_euro = iv_from_price(amer, 100.0, K, T, 0.045, otype="put")
            worst_iv_err = max(worst_iv_err, abs(iv_euro - iv))
    assert worst_frac < 0.02
    assert worst_iv_err < 0.01
