"""Tests for the JAX Black-Scholes pricer (black.py).

The Greeks are closed-form under @jit, so each formula is pinned against a
central finite difference of the pricer itself (a sign error or a missing
e^{-qT} factor fails loudly). The Newton implied-vol solver is tested for
convergence, recovery from a bad starting guess, and clean nan failure on
impossible prices.

Run:  python -m pytest tests/test_black.py -q
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("jax")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing_and_vol_surface"))

from black import black_scholes, greeks, implied_volatility  # noqa: E402

# S, K, T, r, sigma, q, otype - ITM/ATM/OTM, with and without dividends
CASES = [
    (100.0, 100.0, 0.5, 0.03, 0.20, 0.00, "call"),
    (100.0, 110.0, 1.0, 0.05, 0.25, 0.02, "call"),
    (100.0, 90.0, 0.75, 0.01, 0.35, 0.03, "put"),
    (100.0, 100.0, 2.0, 0.04, 0.15, 0.01, "put"),
]


def _price(S, K, T, r, sigma, q, otype):
    return float(black_scholes(S, K, T, r, sigma, q, otype))


# --------------------------------------------------------------------------- #
# Pricing identities                                                          #
# --------------------------------------------------------------------------- #
def test_put_call_parity():
    for S, K, T, r, sigma, q, _ in CASES:
        call = _price(S, K, T, r, sigma, q, "call")
        put = _price(S, K, T, r, sigma, q, "put")
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert np.isclose(call - put, parity, atol=5e-4)


def test_delta_parity():
    """delta_call - delta_put == e^{-qT}."""
    for S, K, T, r, sigma, q, _ in CASES:
        d_call = float(greeks(S, K, T, r, sigma, q, "call")[0])
        d_put = float(greeks(S, K, T, r, sigma, q, "put")[0])
        assert np.isclose(d_call - d_put, np.exp(-q * T), atol=1e-4)


# --------------------------------------------------------------------------- #
# Greeks vs finite differences of the pricer                                  #
# --------------------------------------------------------------------------- #
# Bump sizes chosen large enough that float32 price rounding is negligible
# next to the difference, small enough that truncation error stays tiny.
def test_greeks_match_finite_difference():
    for S, K, T, r, sigma, q, otype in CASES:
        delta, gamma, theta, vega, rho = (float(g) for g in greeks(S, K, T, r, sigma, q, otype))

        h = 0.5
        fd_delta = (_price(S + h, K, T, r, sigma, q, otype)
                    - _price(S - h, K, T, r, sigma, q, otype)) / (2 * h)
        assert np.isclose(delta, fd_delta, atol=2e-3), (otype, "delta")

        # gamma via the (exact) delta to avoid a noisy second difference
        fd_gamma = (float(greeks(S + h, K, T, r, sigma, q, otype)[0])
                    - float(greeks(S - h, K, T, r, sigma, q, otype)[0])) / (2 * h)
        assert np.isclose(gamma, fd_gamma, atol=2e-4), (otype, "gamma")

        h = 1.0 / 365.0  # theta is per calendar day: -dP/dT / 365
        fd_theta = (_price(S, K, T - h, r, sigma, q, otype)
                    - _price(S, K, T + h, r, sigma, q, otype)) / (2 * h) / 365.0
        assert np.isclose(theta, fd_theta, atol=5e-4), (otype, "theta")

        h = 0.005  # vega is per 1% of vol: dP/dsigma / 100
        fd_vega = (_price(S, K, T, r, sigma + h, q, otype)
                   - _price(S, K, T, r, sigma - h, q, otype)) / (2 * h) / 100.0
        assert np.isclose(vega, fd_vega, atol=2e-3), (otype, "vega")

        h = 0.001  # rho is per 1% of rate: dP/dr / 100
        fd_rho = (_price(S, K, T, r + h, sigma, q, otype)
                  - _price(S, K, T, r - h, sigma, q, otype)) / (2 * h) / 100.0
        assert np.isclose(rho, fd_rho, atol=2e-3), (otype, "rho")


# --------------------------------------------------------------------------- #
# Newton implied-vol solver                                                   #
# --------------------------------------------------------------------------- #
def test_implied_vol_round_trip():
    for S, K, T, r, sigma, q, otype in CASES:
        price = _price(S, K, T, r, sigma, q, otype)
        iv = float(implied_volatility(S, K, 0.4, price, r=r, T=T, q=q, otype=otype, E=1e-5))
        assert np.isclose(iv, sigma, atol=2e-3), (otype, sigma, iv)


def test_implied_vol_recovers_from_bad_start():
    """Deep ITM with a near-zero starting guess (vega ~ 0 there): the
    multi-start fallback must still find the true vol, not stall."""
    price = _price(100.0, 60.0, 1.0, 0.045, 0.60, 0.0, "call")
    iv = float(implied_volatility(100.0, 60.0, 0.05, price, r=0.045, T=1.0, E=1e-4))
    assert np.isclose(iv, 0.60, atol=5e-3)


def test_implied_vol_impossible_price_is_nan():
    # below the no-arb lower bound S - K*e^{-rT} ~ 42.6: no vol matches
    assert np.isnan(float(implied_volatility(100.0, 60.0, 0.3, 30.0, r=0.045, T=1.0, E=1e-5)))
    # negative price
    assert np.isnan(float(implied_volatility(100.0, 100.0, 0.2, -1.0, r=0.045, T=1.0, E=1e-5)))
