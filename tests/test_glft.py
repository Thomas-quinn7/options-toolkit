"""Tests for the GLFT quoting engine (market_making/glft.py) and its
integration into the simulator.

The load-bearing tests are the cross-checks between the three routes to the
quotes - the exact finite-horizon solution (eigendecomposition), the paper's
literal matrix exponential, and the closed-form asymptotics - plus the
simulator-level fact the model exists to deliver: derived quotes control
inventory and win on the CARA objective they optimise.

Run:  python -m pytest tests/test_glft.py -q
"""

from dataclasses import replace

import numpy as np
import pytest

from market_making.glft import (
    GLFTParams,
    asymptotic_quotes,
    base_half_spread,
    ergodic_quotes,
    exact_quotes,
    glft_spread_skew,
    inventory_grid,
    skew_unit,
    value_function,
    value_function_expm,
)
from market_making.mm_sim import (
    MMParams,
    _regime_vols,
    certainty_equivalent,
    experiment_glft_quoting,
    experiment_hawkes_clustering,
    simulate_paths,
)

# The simulator's own fill model mapped to its ATM option (dollar vol ~ 11).
P_MM = GLFTParams(sigma=11.0, A=250.0, k=8.0, gamma=0.1, Q=25)

# A parameter set deep in the closed form's asymptotic regime: the stationary
# inventory distribution has width s = (eta/alpha)^(1/4) ~ 16 contracts, so
# the continuum approximation behind the closed form is comfortably valid and
# Q = 60 leaves the bound far away.
P_WIDE = GLFTParams(sigma=1.0, A=2000.0, k=2.0, gamma=0.01, Q=60)


# --------------------------------------------------------------------------- #
# Closed-form structure                                                       #
# --------------------------------------------------------------------------- #
def test_asymptotic_spread_constant_skew_linear():
    """The closed form's whole claim: total spread independent of inventory,
    quote centre shifted linearly by -c1 per contract."""
    q = np.arange(-20, 21)
    db, da = asymptotic_quotes(P_MM, q)
    c0 = base_half_spread(P_MM.gamma, P_MM.k)
    c1 = float(skew_unit(P_MM.sigma, P_MM.A, P_MM.k, P_MM.gamma))
    assert np.allclose(db + da, 2 * c0 + c1, atol=1e-12)          # constant spread
    assert np.allclose((da - db) / 2, -c1 * q, atol=1e-12)        # linear centre shift
    assert c0 > 0 and c1 > 0
    # long inventory: bid backs away, ask leans in (and symmetrically short)
    assert np.all(np.diff(db) > 0)
    assert np.all(np.diff(da) < 0)


def test_adapter_reconstructs_the_closed_form():
    """glft_spread_skew is the same quotes reparameterised for the simulator:
    bid distance = half + skew*q, ask distance = half - skew*q."""
    half, skew = glft_spread_skew(P_MM.sigma, P_MM.A, P_MM.k, P_MM.gamma)
    q = np.arange(-10, 11)
    db, da = asymptotic_quotes(P_MM, q)
    assert np.allclose(half + skew * q, db, atol=1e-12)
    assert np.allclose(half - skew * q, da, atol=1e-12)


def test_adapter_vectorises_over_sigma():
    sig = np.array([2.0, 11.0, 25.0])
    half, skew = glft_spread_skew(sig, P_MM.A, P_MM.k, P_MM.gamma)
    assert half.shape == skew.shape == sig.shape
    # skew scales linearly in sigma (c1 ~ sigma); the base c0 does not move
    assert np.allclose(skew / sig, skew[0] / sig[0], rtol=1e-12)
    assert np.all(np.diff(half) > 0)


# --------------------------------------------------------------------------- #
# Exact solution                                                              #
# --------------------------------------------------------------------------- #
def test_value_function_positive_and_terminal():
    """v must be positive wherever it is numerically resolved (true tails decay
    like a Gaussian in q and legitimately fall below double precision)."""
    qs = inventory_grid(P_MM)
    for tau in (0.0, 0.05, 0.5, 5.0):
        v = value_function(P_MM, tau)
        resolved = v > 1e-12 * v.max()
        assert np.all(v[resolved] > 0), f"v must be positive where resolved (tau={tau})"
        assert np.all(v > -1e-12), f"no significant negative entries (tau={tau})"
        assert resolved[P_MM.Q], "the flat-inventory state must always be resolved"
    # tau = 0 reproduces the terminal condition exactly (flat with no penalty)
    assert np.allclose(value_function(P_MM, 0.0), 1.0, atol=1e-10)
    p_liq = GLFTParams(sigma=11.0, A=250.0, k=8.0, gamma=0.1, Q=25, liq_penalty=0.05)
    vT = value_function(p_liq, 0.0)
    assert np.allclose(vT, np.exp(-p_liq.k * p_liq.liq_penalty * np.abs(qs)),
                       atol=1e-10)


def test_eigendecomposition_matches_paper_expm():
    """The stable eigendecomposition route must reproduce the paper's literal
    expm(-M tau) formula wherever the latter doesn't overflow."""
    p = GLFTParams(sigma=11.0, A=250.0, k=8.0, gamma=0.1, Q=15)
    for tau in (0.01, 0.1, 0.25):
        assert np.allclose(value_function(p, tau), value_function_expm(p, tau),
                           rtol=1e-8), f"tau={tau}"


def test_exact_symmetry_without_drift():
    """No drift: v is even in q, so delta_b(q) = delta_a(-q), where resolved."""
    v = value_function(P_MM, 0.3)
    # entries near the resolution floor carry amplified relative roundoff, so
    # the tight comparison runs on the well-resolved band only
    well = (v > 1e-6 * v.max()) & (v[::-1] > 1e-6 * v.max())
    assert well.sum() >= 9, "the central band must be well resolved"
    assert np.allclose(v[well], v[::-1][well], rtol=1e-9)
    _, db, da = exact_quotes(P_MM, 0.3)
    pair = well[:-1] & well[1:]          # a quote uses two neighbouring entries
    both = np.concatenate([pair, [False]]) & ~np.isnan(db) & ~np.isnan(da[::-1])
    assert np.allclose(db[both], da[::-1][both], atol=1e-8)


def test_terminal_limit_is_c0():
    """As tau -> 0 with no liquidation penalty, all skew fades: there is no
    time left to be stuck with inventory, so every level quotes c0."""
    _, db, da = exact_quotes(P_MM, 1e-9)
    c0 = base_half_spread(P_MM.gamma, P_MM.k)
    assert np.allclose(db[~np.isnan(db)], c0, atol=1e-4)
    assert np.allclose(da[~np.isnan(da)], c0, atol=1e-4)
    # and the skew is already material one trading day from the terminal time
    _, db_day, _ = exact_quotes(P_MM, 1.0 / 252.0)
    i = P_MM.Q
    assert db_day[i + 1] - db_day[i] > 0.01


def test_exact_converges_to_ergodic():
    """Far from the terminal time the finite-horizon quotes must sit on the
    stationary (ground-eigenvector) quotes, across the resolved band."""
    qs, db_erg, da_erg = ergodic_quotes(P_MM)
    _, db, da = exact_quotes(P_MM, 5.0)
    keep = ~np.isnan(db_erg) & ~np.isnan(db)
    assert keep.sum() >= 9, "the central band must be resolved"
    assert np.allclose(db[keep], db_erg[keep], atol=1e-6)
    keep = ~np.isnan(da_erg) & ~np.isnan(da)
    assert np.allclose(da[keep], da_erg[keep], atol=1e-6)


def test_ergodic_matches_closed_form_in_asymptotic_regime():
    """The closed-form quotes are the continuum approximation of the exact
    stationary ones; deep in their regime (inventory-distribution width
    s ~ 16 >> 1, bound Q ~ 4s away) they must agree tightly at central q."""
    qs, db_erg, _ = ergodic_quotes(P_WIDE)
    db_cf, _ = asymptotic_quotes(P_WIDE, qs)
    c1 = float(skew_unit(P_WIDE.sigma, P_WIDE.A, P_WIDE.k, P_WIDE.gamma))
    centre = np.abs(qs) <= 10
    err = np.abs(db_erg[centre] - db_cf[centre])
    assert np.max(err) < 0.05 * c1, f"max error {np.max(err):.2e} vs c1={c1:.2e}"


def test_param_validation():
    with pytest.raises(ValueError):
        GLFTParams(sigma=-1.0, A=250.0, k=8.0, gamma=0.1)
    with pytest.raises(ValueError):
        GLFTParams(sigma=1.0, A=250.0, k=8.0, gamma=0.0)
    with pytest.raises(ValueError):
        value_function(P_MM, -0.1)


# --------------------------------------------------------------------------- #
# Simulator integration                                                       #
# --------------------------------------------------------------------------- #
def test_glft_policy_controls_inventory_and_risk():
    """Under one-sided client flow with UNCERTAIN realised vol (the arena
    where a hedged desk's inventory is a live vega bet), the GLFT-derived
    skew must hold the book near flat, cut P&L dispersion, and win on the
    CARA utility the model optimises, against a no-inventory-control desk.

    (At realised == implied inventory is nearly free to carry and the no-skew
    desk's extra volume wins - which is model-consistent, not a bug: there is
    nothing for inventory control to pay for in that arena.)
    """
    base = MMParams(flow_imbalance=0.30)
    rng = np.random.default_rng
    sig = _regime_vols(base, 0.06, 1500, seed=8)
    no_skew = simulate_paths(replace(base, skew_coef=0.0), sig,
                             1500, rng(9), quoting=True)
    glft = simulate_paths(replace(base, quote_policy="glft", gamma_risk=0.1),
                          sig, 1500, rng(9), quoting=True)
    inv_ns = np.abs(no_skew["final_inventory"]).mean()
    inv_g = np.abs(glft["final_inventory"]).mean()
    assert inv_g < 0.3 * inv_ns
    assert glft["total_pnl"].std(ddof=1) < 0.5 * no_skew["total_pnl"].std(ddof=1)
    # the risk saved is worth the volume given up, under the model's own utility
    ce_g = certainty_equivalent(glft["total_pnl"], 0.1)
    ce_ns = certainty_equivalent(no_skew["total_pnl"], 0.1)
    assert ce_g > ce_ns + 0.5


def test_experiment_glft_quoting_runs_and_ranks():
    res = experiment_glft_quoting(MMParams(flow_imbalance=0.30), gammas=(0.1,),
                                  n_sims=1200, seed=8)
    by_desk = {r["desk"]: r for r in res["rows"]}
    assert set(by_desk) == {"no skew", "hand-tuned", "GLFT g=0.1"}
    # any inventory control beats none by a wide margin; the derived desk's
    # book is comparable to the hand-tuned one (not asserted strictly - the
    # two sit close on the frontier and Monte-Carlo noise can swap them)
    assert by_desk["GLFT g=0.1"]["abs_inv"] < 0.5 * by_desk["no skew"]["abs_inv"]
    assert by_desk["hand-tuned"]["abs_inv"] < 0.5 * by_desk["no skew"]["abs_inv"]
    assert by_desk["GLFT g=0.1"]["ce"] > by_desk["no skew"]["ce"] + 1.0
    # the derived desk is the most conservative point shown on the frontier
    assert by_desk["GLFT g=0.1"]["std"] < by_desk["hand-tuned"]["std"]
    # every desk still trades - the derived quotes are competitive, not shut
    assert all(r["avg_fills"] > 5 for r in res["rows"])


def test_quote_policy_validation():
    with pytest.raises(ValueError):
        simulate_paths(MMParams(quote_policy="nope"), 0.2, 10,
                       np.random.default_rng(0))
    with pytest.raises(ValueError):
        simulate_paths(MMParams(quote_policy="glft", gamma_risk=0.0), 0.2, 10,
                       np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Clustered (self-exciting) flow                                              #
# --------------------------------------------------------------------------- #
def test_hawkes_off_changes_nothing():
    """branching = 0 must reproduce the independent-flow desk exactly - same
    RNG draws, same fills, same P&L to the last bit."""
    base = MMParams(flow_imbalance=0.15)
    a = simulate_paths(base, 0.2, 400, np.random.default_rng(3), quoting=True)
    b = simulate_paths(replace(base, hawkes_branching=0.0, hawkes_decay=99.0),
                       0.2, 400, np.random.default_rng(3), quoting=True)
    assert np.array_equal(a["total_pnl"], b["total_pnl"])
    assert np.array_equal(a["fills"], b["fills"])


def test_hawkes_validation():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            simulate_paths(MMParams(hawkes_branching=bad), 0.2, 10,
                           np.random.default_rng(0))
    with pytest.raises(ValueError):
        simulate_paths(MMParams(hawkes_branching=0.5, hawkes_decay=0.0), 0.2,
                       10, np.random.default_rng(0))


def test_hawkes_clustering_raises_inventory_risk():
    """Volume-matched clustering must raise peak inventory and P&L dispersion
    - the cost the independent-fills assumption hides - while realised volume
    stays comparable at moderate branching (the skew extinguishes runs, so
    some volume loss is expected and honest)."""
    base = MMParams(flow_imbalance=0.0)
    sig = _regime_vols(base, 0.06, 2000, seed=10)
    rng = np.random.default_rng
    indep = simulate_paths(base, sig, 2000, rng(11), quoting=True)
    clust = simulate_paths(replace(base, hawkes_branching=0.6), sig, 2000,
                           rng(11), quoting=True)
    assert clust["max_abs_inventory"].mean() > 1.1 * indep["max_abs_inventory"].mean()
    assert clust["total_pnl"].std(ddof=1) > 1.2 * indep["total_pnl"].std(ddof=1)
    assert clust["fills"].mean() > 0.7 * indep["fills"].mean()


def test_hawkes_flips_the_desk_ranking():
    """Under independent flow the hand-tuned desk out-CEs the (conservative)
    GLFT desk; under heavy clustering the ranking must flip - the hand-tuned
    point was tuned for the assumption that just failed."""
    res = experiment_hawkes_clustering(MMParams(flow_imbalance=0.0),
                                       branchings=(0.0, 0.8), n_sims=2500,
                                       seed=10)
    ce = {(r["branching"], r["desk"]): r["ce"] for r in res["rows"]}
    assert ce[(0.0, "hand-tuned")] > ce[(0.0, "GLFT g=0.1")]
    assert ce[(0.8, "GLFT g=0.1")] > ce[(0.8, "hand-tuned")]
