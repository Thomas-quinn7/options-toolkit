"""Tests for the options market-making simulator.

The load-bearing test is ``test_hedging_identity``: it checks the simulated
delta-hedged P&L against the closed-form Black-Scholes gamma-P&L, which is what
makes the simulator's vol P&L trustworthy rather than just plausible.

Run:  python -m pytest tests/test_mm.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "market_making"))

from mm_sim import (  # noqa: E402
    MMParams,
    bs_delta,
    bs_gamma,
    bs_price,
    experiment_adverse_selection,
    experiment_hedging_validation,
    experiment_mm_vol_sweep,
    experiment_toxic_spread,
    experiment_vol_informed_flow,
    experiment_vol_spread_defence,
    simulate_paths,
)


# --------------------------------------------------------------------------- #
# Closed-form Black-Scholes sanity                                            #
# --------------------------------------------------------------------------- #
def test_put_call_parity():
    S, K, tau, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.2
    call = bs_price(S, K, tau, r, sigma, otype="call")
    put = bs_price(S, K, tau, r, sigma, otype="put")
    assert np.isclose(call - put, S - K * np.exp(-r * tau), atol=1e-10)


def test_delta_matches_finite_difference():
    S, K, tau, r, sigma = 100.0, 100.0, 0.4, 0.01, 0.25
    h = 1e-4
    fd = (bs_price(S + h, K, tau, r, sigma) - bs_price(S - h, K, tau, r, sigma)) / (2 * h)
    assert np.isclose(bs_delta(S, K, tau, r, sigma), fd, atol=1e-5)


def test_gamma_matches_finite_difference():
    S, K, tau, r, sigma = 100.0, 100.0, 0.4, 0.01, 0.25
    h = 1e-3
    fd = (bs_delta(S + h, K, tau, r, sigma) - bs_delta(S - h, K, tau, r, sigma)) / (2 * h)
    assert np.isclose(bs_gamma(S, K, tau, r, sigma), fd, atol=1e-5)


def test_expiry_intrinsic():
    assert np.isclose(bs_price(110.0, 100.0, 0.0, 0.0, 0.2, otype="call"), 10.0)
    assert np.isclose(bs_price(90.0, 100.0, 0.0, 0.0, 0.2, otype="call"), 0.0)
    assert np.isclose(bs_delta(110.0, 100.0, 0.0, 0.0, 0.2, otype="call"), 1.0)
    assert np.isclose(bs_gamma(110.0, 100.0, 0.0, 0.0, 0.2), 0.0)


# --------------------------------------------------------------------------- #
# The hedging identity - the core correctness guarantee                       #
# --------------------------------------------------------------------------- #
def test_hedging_identity():
    """Simulated hedged P&L must match the BS gamma-P&L theory within noise."""
    params = MMParams(n_steps=126)
    grid = np.array([0.14, 0.20, 0.26])
    val = experiment_hedging_validation(params, grid, n_sims=3000, seed=7)
    for row in val["rows"]:
        err = abs(row["sim_mean"] - row["theory_mean"])
        assert err < 6 * row["sim_se"] + 1e-3, row  # within Monte-Carlo error


def test_short_is_long_vol_downside():
    """A hedged short option makes money when realised < implied, loses when >."""
    params = MMParams()
    grid = np.array([0.14, 0.26])
    val = experiment_hedging_validation(params, grid, n_sims=3000, seed=3)
    below, above = val["rows"][0], val["rows"][1]
    assert below["sim_mean"] > 0 > above["sim_mean"]


# --------------------------------------------------------------------------- #
# The MM decomposition behaves as a market-maker's book should                #
# --------------------------------------------------------------------------- #
def test_spread_flat_vol_slopes_down():
    params = MMParams(flow_imbalance=0.30)
    grid = np.linspace(0.14, 0.28, 5)
    sweep = experiment_mm_vol_sweep(params, grid, n_sims=2500, seed=1)
    spreads = np.array([r["spread_mean"] for r in sweep["rows"]])
    vols = np.array([r["vol_mean"] for r in sweep["rows"]])
    totals = np.array([r["total_mean"] for r in sweep["rows"]])
    # spread capture does not depend on realised vol (flat within a few %)
    assert spreads.std() / spreads.mean() < 0.05
    # a net-short desk's vol P&L and total P&L fall as realised vol rises
    assert vols[0] > vols[-1]
    assert totals[0] > totals[-1]


# --------------------------------------------------------------------------- #
# Adverse selection / toxic flow                                              #
# --------------------------------------------------------------------------- #
def test_adverse_selection_needs_hedge_lag():
    """At realised==implied vol: delta-hedging before the move neutralises the
    direction of informed flow (lag0 residual ~ 0 at any toxicity); the cost
    appears only when hedging lags the move (lag1 residual << 0 with toxicity)."""
    base = MMParams(flow_imbalance=0.0)
    rows = experiment_adverse_selection(base, [0.0, 0.6], n_sims=3000, seed=2)
    no_tox, toxic = rows[0], rows[1]
    # hedging before the move: no systematic residual, with or without toxicity
    assert abs(no_tox["lag0_resid"]) < 0.2
    assert abs(toxic["lag0_resid"]) < 0.2
    # hedging after the move: ~0 without toxicity, strongly negative with it
    assert abs(no_tox["lag1_resid"]) < 0.25
    assert toxic["lag1_resid"] < -1.0


def test_wider_spread_survives_toxic_flow():
    base = MMParams(flow_imbalance=0.0)
    grid = experiment_toxic_spread(base, [0.6], [0.10, 0.25], n_sims=3000, seed=3)
    assert grid[0.25][0] > grid[0.10][0]


# --------------------------------------------------------------------------- #
# Vol-informed (vega-toxic) flow                                              #
# --------------------------------------------------------------------------- #
def test_per_path_sigma_preserves_hedging_identity():
    """The gamma-P&L identity must hold path-by-path with per-path realised vols."""
    params = MMParams(n_steps=126)
    rng = np.random.default_rng(11)
    sig = np.where(rng.random(3000) < 0.5, 0.26, 0.14)
    res = simulate_paths(params, sig, 3000, rng, quoting=False, init_position=-1)
    err = abs(res["total_pnl"].mean() - res["vol_theory"].mean())
    se = res["total_pnl"].std(ddof=1) / np.sqrt(len(res["total_pnl"]))
    assert err < 6 * se + 1e-3


def test_vol_informed_flow_survives_instant_hedging():
    """Instant hedging neutralises direction-informed flow (residual ~ 0 at any
    toxicity - its total falls only because informed flow is one-sided volume)
    but NOT vol-informed flow, whose loss lands in the vol/hedging residual."""
    base = MMParams(flow_imbalance=0.0)
    rows = experiment_vol_informed_flow(base, [0.0, 0.6], vol_shock=0.06,
                                        n_sims=3000, seed=4)
    clean, toxic = rows[0], rows[1]
    # direction-informed flow, hedged instantly: no systematic vol residual
    assert abs(clean["dir_resid"]) < 0.3
    assert abs(toxic["dir_resid"]) < 0.3
    # vol-informed flow: a large negative residual instant hedging cannot remove
    assert toxic["vol_resid"] < -1.5
    # at the same toxicity (same one-sided volume geometry), the vega-toxic desk
    # does materially worse than the direction-toxic one
    assert toxic["vol_total"] < toxic["dir_total"] - 1.5


def test_vol_spread_defends_against_vega_toxicity():
    """The vol-space markup charges informed flow in its own currency: it
    shrinks the vega adverse-selection residual, an interior markup beats no
    defence under toxic flow, and the same markup is pure cost on clean flow."""
    base = MMParams(flow_imbalance=0.0)
    rows = experiment_vol_spread_defence(base, [0.0, 0.005, 0.02], tox=0.5,
                                         vol_shock=0.06, n_sims=3000, seed=5)
    none, small, wide = rows
    # the residual (the vega loss itself) shrinks as the markup widens
    assert wide["resid"] > none["resid"] + 1.0
    # under toxic flow, a small markup beats quoting none
    assert small["total"] > none["total"]
    # under clean flow the markup only costs volume - no free lunch
    assert small["clean_total"] < none["clean_total"]


def test_flow_fractions_validated():
    with pytest.raises(ValueError):
        simulate_paths(MMParams(toxicity=0.7, vol_toxicity=0.6), 0.2, 10,
                       np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Cross-check the vectorised BS against the repo's autodiff pricer (black.py)  #
# --------------------------------------------------------------------------- #
def test_cross_check_black_py():
    pytest.importorskip("jax")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing_and_vol_surface"))
    import black  # noqa: E402

    for S, K, tau, r, sigma in [(100, 100, 0.5, 0.02, 0.2), (95, 105, 0.25, 0.0, 0.35)]:
        assert np.isclose(bs_price(S, K, tau, r, sigma), float(black.black_scholes(S, K, tau, r, sigma)), atol=1e-6)
        d, g, *_ = black.greeks(S, K, tau, r, sigma)
        assert np.isclose(bs_delta(S, K, tau, r, sigma), float(d), atol=1e-6)
        assert np.isclose(bs_gamma(S, K, tau, r, sigma), float(g), atol=1e-6)
