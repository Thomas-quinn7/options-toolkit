"""Tests for the options market-making simulator.

The load-bearing test is ``test_hedging_identity``: it checks the simulated
delta-hedged P&L against the closed-form Black-Scholes gamma-P&L, which is what
makes the simulator's vol P&L trustworthy rather than just plausible.

Run:  python -m pytest tests/test_mm.py -q
"""

import numpy as np
import pytest

from market_making.mm_sim import (
    MMParams,
    bs_delta,
    bs_gamma,
    bs_price,
    experiment_adverse_selection,
    experiment_hedging_validation,
    experiment_mm_vol_sweep,
    _regime_vol_paths,
    experiment_online_toxicity,
    experiment_online_vol_toxicity,
    experiment_toxic_spread,
    experiment_vol_informed_flow,
    experiment_vol_spread_defence,
    pool_vega_markouts,
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


def test_hedging_identity_put():
    """The identity must hold for puts, not just the default call."""
    params = MMParams(n_steps=126, otype="put")
    grid = np.array([0.14, 0.20, 0.26])
    val = experiment_hedging_validation(params, grid, n_sims=3000, seed=11)
    for row in val["rows"]:
        err = abs(row["sim_mean"] - row["theory_mean"])
        assert err < 6 * row["sim_se"] + 1e-3, row


def test_hedging_identity_with_financing():
    """r, q != 0: cash accrues at r, the hedge book earns the dividend yield,
    and the theory accrual carries the e^{r(T-t)} factor - the identity must
    still hold, for both option types."""
    grid = np.array([0.14, 0.26])
    for otype in ("call", "put"):
        params = MMParams(n_steps=126, r=0.05, q=0.02, otype=otype)
        val = experiment_hedging_validation(params, grid, n_sims=3000, seed=13)
        for row in val["rows"]:
            err = abs(row["sim_mean"] - row["theory_mean"])
            assert err < 6 * row["sim_se"] + 1e-3, (otype, row)


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


def test_adverse_selection_put_flow_direction():
    """Informed flow must be toxic for puts too: it buys them before
    down-moves, so the lag-1 residual goes negative just as for calls."""
    base = MMParams(flow_imbalance=0.0, otype="put")
    rows = experiment_adverse_selection(base, [0.0, 0.6], n_sims=3000, seed=5)
    no_tox, toxic = rows[0], rows[1]
    assert abs(toxic["lag0_resid"]) < 0.2
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
# Online toxicity estimation and adaptive quoting                             #
# --------------------------------------------------------------------------- #
def test_toxicity_estimator_converges():
    """The markout EWMA must land near the true informed fraction - and near
    zero when the flow is clean."""
    toxic = simulate_paths(MMParams(toxicity=0.6), 0.2, 2000,
                           np.random.default_rng(8), quoting=True)
    clean = simulate_paths(MMParams(toxicity=0.0), 0.2, 2000,
                           np.random.default_rng(8), quoting=True)
    assert abs(float(toxic["tox_hat_final"].mean()) - 0.6) < 0.15
    assert float(clean["tox_hat_final"].mean()) < 0.15


def test_adaptive_spread_defends_without_overpaying():
    """Under toxic flow the adaptive desk must beat the static one; under
    clean flow it must beat the permanently-wide (oracle-for-toxic) desk,
    because it only widens when its estimator sees toxicity."""
    online = experiment_online_toxicity(MMParams(flow_imbalance=0.0),
                                        tox_hi=0.6, spread_slope=0.25,
                                        n_sims=2500, seed=6)
    t = online["totals"]
    assert t[("toxic", "adaptive")] > t[("toxic", "static")] + 0.3
    assert t[("clean", "adaptive")] > t[("clean", "oracle-wide")] + 0.3
    # when toxicity is time-varying, adapting beats BOTH fixed policies -
    # static is too tight in the toxic half, oracle-wide too wide in the clean
    assert t[("regime switch", "adaptive")] > t[("regime switch", "static")] + 0.2
    assert t[("regime switch", "adaptive")] > t[("regime switch", "oracle-wide")] + 0.2


def test_toxicity_estimator_tracks_regime_switch():
    """After a mid-sim switch from clean to toxic, the running estimate must
    rise materially above its clean-half level."""
    online = experiment_online_toxicity(MMParams(flow_imbalance=0.0),
                                        tox_hi=0.6, spread_slope=0.25,
                                        n_sims=2500, seed=6)
    track = online["tracks"]["regime switch"]
    n = len(track)
    clean_half = float(np.mean(track[n // 4: n // 2]))
    toxic_tail = float(np.mean(track[-n // 5:]))
    assert clean_half < 0.2
    assert toxic_tail > clean_half + 0.15


# --------------------------------------------------------------------------- #
# Online VOL-toxicity estimation (vega-space markout) and adaptive markup     #
# --------------------------------------------------------------------------- #
def test_vega_markout_estimator_discriminates():
    """The vega markout must read high on vol-toxic flow and low on both clean
    and direction-toxic flow - and the two estimators must separate the two
    kinds of toxicity from each other."""
    base = MMParams(flow_imbalance=0.0)
    sig = _regime_vol_paths(base, 0.06, 2500, seed=9)
    vol_toxic = simulate_paths(MMParams(vol_toxicity=0.6), sig, 2500,
                               np.random.default_rng(9), quoting=True)
    clean = simulate_paths(MMParams(), sig, 2500,
                           np.random.default_rng(9), quoting=True)
    dir_toxic = simulate_paths(MMParams(toxicity=0.6), sig, 2500,
                               np.random.default_rng(9), quoting=True)
    vm_vol = float(vol_toxic["volmark_final"].mean())
    vm_clean = float(clean["volmark_final"].mean())
    vm_dir = float(dir_toxic["volmark_final"].mean())
    assert vm_vol > vm_clean + 0.06
    assert abs(vm_dir - vm_clean) < 0.02   # blind to directional toxicity
    # cross-separation: each estimator sees only its own kind
    assert float(dir_toxic["tox_hat_final"].mean()) > 0.4
    assert float(vol_toxic["tox_hat_final"].mean()) < 0.2


def test_per_step_sigma_preserves_hedging_identity():
    """The gamma-P&L identity must hold with a full vol PATH per simulation
    (regimes switching mid-book), not just a per-path constant."""
    params = MMParams(n_steps=126)
    sig = _regime_vol_paths(params, 0.06, 3000, seed=12)
    res = simulate_paths(params, sig, 3000, np.random.default_rng(12),
                         quoting=False, init_position=-1)
    err = abs(res["total_pnl"].mean() - res["vol_theory"].mean())
    se = res["total_pnl"].std(ddof=1) / np.sqrt(len(res["total_pnl"]))
    assert err < 6 * se + 1e-3


def test_adaptive_vol_markup_is_priced_not_free():
    """The adaptive vol markup must (a) nearly match the static desk on clean
    flow - unlike the oracle markup, which taxes clean flow heavily - (b)
    improve on static under vol-toxic flow, and (c) beat the oracle when
    toxicity switches regime. The oracle keeping an edge in stationary toxic
    flow is expected: a single book's vega markout is too noisy to recover
    the full oracle markup, which is the honest asymmetry vs the directional
    case."""
    online = experiment_online_vol_toxicity(MMParams(flow_imbalance=0.0),
                                            n_sims=2500, seed=7)
    t = online["totals"]
    assert t[("clean", "adaptive")] > t[("clean", "oracle-markup")] + 0.2
    assert t[("clean", "adaptive")] > t[("clean", "static")] - 0.15
    assert t[("vol-toxic", "adaptive")] > t[("vol-toxic", "static")] + 0.02
    assert t[("regime switch", "adaptive")] > t[("regime switch", "oracle-markup")] + 0.15
    # the estimator's running mean rises after the vol-toxicity switch
    track = online["tracks"]["regime switch"]
    n = len(track)
    clean_half = float(np.mean(track[n // 4: n // 2]))
    toxic_tail = float(np.mean(track[-n // 5:]))
    assert toxic_tail > clean_half + 0.05


# --------------------------------------------------------------------------- #
# Cross-book pooling of the vega markout                                      #
# --------------------------------------------------------------------------- #
def test_pooling_off_is_identical():
    """pool_books=1 (the default) must be the pre-pooling per-book path
    bit-for-bit - same RNG stream, same arithmetic, same outputs. And with
    the adaptive markup OFF, pooling is a read-only layer: turning it on must
    not change a single P&L number either."""
    base = MMParams(flow_imbalance=0.0)
    sig = _regime_vol_paths(base, 0.06, 800, seed=9)
    kw = dict(flow_imbalance=0.0, vol_toxicity=0.6,
              adaptive_vol_spread=True, vol_spread_slope=0.06)
    a = simulate_paths(MMParams(**kw), sig, 800,
                       np.random.default_rng(3), quoting=True)
    b = simulate_paths(MMParams(**kw, pool_books=1, pool_shrinkage=5.0), sig, 800,
                       np.random.default_rng(3), quoting=True)
    for key in ("total_pnl", "spread_capture", "fills", "volmark_final",
                "volmark_pooled_final", "volmark_track", "tox_hat_final"):
        assert np.array_equal(a[key], b[key]), key
    # pooling off: the acting estimate IS the per-book estimate
    assert np.array_equal(a["volmark_final"], a["volmark_pooled_final"])
    # adaptive markup off: pooling only reads state, so P&L is untouched
    c = simulate_paths(MMParams(flow_imbalance=0.0, vol_toxicity=0.6), sig, 800,
                       np.random.default_rng(3), quoting=True)
    d = simulate_paths(MMParams(flow_imbalance=0.0, vol_toxicity=0.6,
                                pool_books=8), sig, 800,
                       np.random.default_rng(3), quoting=True)
    for key in ("total_pnl", "spread_capture", "fills", "volmark_final"):
        assert np.array_equal(c[key], d[key]), key
    # config validation: path count must group evenly into books
    with pytest.raises(ValueError):
        simulate_paths(MMParams(pool_books=7), 0.2, 10, np.random.default_rng(0))


def test_pooled_estimate_converges_faster_synthetic():
    """On synthetic common-toxicity markouts (every book sees noisy draws of
    the SAME mean - the exchangeable case), the pooled/shrunk estimate must
    converge faster than the per-book EWMA: materially less dispersed at
    every horizon, and closer to the truth once the shared zero-anchor bias
    (the conservative prior, which pooling deliberately keeps) has washed
    out. The update rule mirrors the simulator's."""
    rng = np.random.default_rng(0)
    B, alpha, kappa, mu = 10, 0.08, 20.0, 0.3
    n_desks = 200
    x = np.zeros(n_desks * B)
    nobs = np.zeros(n_desks * B)
    for step in range(1, 31):
        obs = mu + rng.normal(0.0, np.sqrt(2.0 / 5.0), n_desks * B)  # K=5 markout noise
        x += alpha * (obs - x)
        nobs += 1.0
        if step in (10, 30):
            pooled = pool_vega_markouts(x, nobs, B, kappa, alpha)
            assert float(pooled.std()) < 0.7 * float(x.std()), step
    rmse_own = float(np.sqrt(np.mean((x - mu) ** 2)))
    rmse_pool = float(np.sqrt(np.mean((pooled - mu) ** 2)))
    assert rmse_pool < 0.8 * rmse_own


def test_pooled_estimate_less_noisy_in_sim():
    """Inside the simulator, on flow with one COMMON vol-toxicity, the pooled
    acting estimate must be materially less dispersed across books than the
    per-book one at the same horizon, without losing the signal - and on
    clean flow its phantom-toxicity floor (the clipped noise) must drop."""
    base = MMParams(flow_imbalance=0.0)
    sig = _regime_vol_paths(base, 0.06, 2000, seed=9)
    toxic = simulate_paths(MMParams(vol_toxicity=0.6, pool_books=10), sig, 2000,
                           np.random.default_rng(9), quoting=True)
    per_book, pooled = toxic["volmark_final"], toxic["volmark_pooled_final"]
    assert float(pooled.std()) < 0.75 * float(per_book.std())
    assert float(pooled.mean()) > 0.75 * float(per_book.mean())   # signal kept
    clean = simulate_paths(MMParams(pool_books=10), sig, 2000,
                           np.random.default_rng(9), quoting=True)
    assert (float(clean["volmark_pooled_final"].mean())
            < 0.8 * float(clean["volmark_final"].mean()))


def test_pool_shrinkage_borrows_then_releases():
    """A thin book's acting estimate must sit near the pool (borrowing
    strength); as its own markout count grows, the weight returns to its own
    estimate - a genuinely different book separates from the pool instead of
    being averaged away, and the clean books are barely contaminated."""
    B, alpha, kappa = 8, 0.08, 20.0
    own = 0.5                       # one book persistently reads vega-toxic
    counts = [0.0, 5.0, 20.0, 60.0, 300.0]
    ests = []
    for n_own in counts:
        x = np.concatenate([[own], np.zeros(B - 1)])
        n = np.concatenate([[n_own], np.full(B - 1, 60.0)])
        est = pool_vega_markouts(x, n, B, kappa, alpha)
        ests.append(float(est[0]))
        clean_books = est[1:]
        # the outlier's contamination of the clean books stays small
        assert float(np.max(clean_books)) < 0.05
    # no own evidence: act on the pool (its own unsupported EWMA is ignored)
    assert ests[0] < 0.02
    # thin book: still mostly pooled
    assert ests[1] < 0.5 * own
    # evidence accumulates: weight released back to the book's own estimate
    assert np.all(np.diff(ests) > 0)
    assert ests[-1] > 0.9 * own * (300.0 / (300.0 + kappa))
    assert ests[-1] > 0.4


def test_pooled_estimate_is_causal():
    """The acting (pooled) estimate at step t may use only fills resolved
    before t: two runs whose flow differs only from step m onward must show
    identical estimate tracks up to and including m, then diverge."""
    base = MMParams(flow_imbalance=0.0)
    n = base.n_steps
    m = n // 2
    sched_a = np.zeros(n)
    sched_b = np.where(np.arange(n) < m, 0.0, 0.6)
    sig = _regime_vol_paths(base, 0.06, 400, seed=9)
    kw = dict(flow_imbalance=0.0, adaptive_vol_spread=True,
              vol_spread_slope=0.06, volmark_deadband=0.05, pool_books=8)
    ra = simulate_paths(MMParams(**kw, vol_toxicity_schedule=sched_a), sig, 400,
                        np.random.default_rng(21), quoting=True)
    rb = simulate_paths(MMParams(**kw, vol_toxicity_schedule=sched_b), sig, 400,
                        np.random.default_rng(21), quoting=True)
    assert np.array_equal(ra["volmark_track"][: m + 1],
                          rb["volmark_track"][: m + 1])
    assert not np.array_equal(ra["volmark_track"], rb["volmark_track"])


def test_pooled_markup_recovers_more_of_the_oracle_edge():
    """The point of pooling (the README's planned item, now implemented): one
    book's vega markout is too noisy to act on aggressively; ten books'
    pooled markout - with the null threshold recalibrated to the pooled
    noise floor - is not. Same markup rule, same slope and cap."""
    online = experiment_online_vol_toxicity(MMParams(flow_imbalance=0.0),
                                            n_sims=2500, seed=7, pool_books=10)
    t = online["totals"]
    # stationary vol-toxic flow: pooled beats per-book adaptive and static
    assert t[("vol-toxic", "adaptive-pooled")] > t[("vol-toxic", "adaptive")] + 0.015
    assert t[("vol-toxic", "adaptive-pooled")] > t[("vol-toxic", "static")] + 0.05
    # clean flow: the pooled desk's phantom-toxicity tax is SMALLER than the
    # per-book desk's, despite its lower deadband - less noise to rectify
    assert t[("clean", "adaptive-pooled")] > t[("clean", "adaptive")]
    assert (t[("clean", "static")] - t[("clean", "adaptive-pooled")]
            < 0.75 * (t[("clean", "static")] - t[("clean", "adaptive")]))
    # regime switch: pooling keeps the adaptive desk's edge over the oracle
    assert (t[("regime switch", "adaptive-pooled")]
            > t[("regime switch", "oracle-markup")] + 0.15)
    # and the pooled acting estimate still tracks the switch
    track = online["tracks"]["regime switch (pooled)"]
    n = len(track)
    clean_half = float(np.mean(track[n // 4: n // 2]))
    toxic_tail = float(np.mean(track[-n // 5:]))
    assert toxic_tail > clean_half + 0.04


# --------------------------------------------------------------------------- #
# Cross-check the vectorised BS against the repo's autodiff pricer (black.py)  #
# --------------------------------------------------------------------------- #
def test_cross_check_black_py():
    pytest.importorskip("jax")
    from pricing_and_vol_surface import black

    for S, K, tau, r, sigma in [(100, 100, 0.5, 0.02, 0.2), (95, 105, 0.25, 0.0, 0.35)]:
        assert np.isclose(bs_price(S, K, tau, r, sigma), float(black.black_scholes(S, K, tau, r, sigma)), atol=1e-6)
        d, g, *_ = black.greeks(S, K, tau, r, sigma)
        assert np.isclose(bs_delta(S, K, tau, r, sigma), float(d), atol=1e-6)
        assert np.isclose(bs_gamma(S, K, tau, r, sigma), float(g), atol=1e-6)
