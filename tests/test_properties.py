"""Property-based tests of the repo's pricing and no-arbitrage invariants.

Instead of checking hand-picked examples, these tests let hypothesis sample
the parameter spaces and assert the invariants the modules are BUILT on:

* Black-Scholes structure: put-call parity, price bounds, monotonicity in
  spot/strike/vol, delta and gamma bounds - for every valid input, not three.
* IV inversion: price -> implied vol -> price must round-trip.
* SVI: the closed-form derivatives must match finite differences (the
  Durrleman g test is only as good as w' and w'').
* SSVI (Gatheral-Jacquier 2014): parameters satisfying the sufficient
  no-butterfly conditions must produce a non-negative Durrleman density g(k)
  across the whole surface - sampled up to the boundary of the conditions,
  where the guarantee is thinnest. With the additional power-law bound
  eta * (1 + |rho|) <= 2 (Gatheral-Jacquier Theorem 4.3), the surface must
  also be calendar-arbitrage-free.
* GLFT: the closed form's structural claim (constant spread, linear skew) and
  its equivalence to the simulator adapter, for every valid parameter set.

A failure arrives with a minimal counterexample, which is exactly what a
numerical library wants from its test suite.

Run:  python -m pytest tests/test_properties.py -q
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pricing_and_vol_surface import vol_surface as vs
from market_making.glft import asymptotic_quotes, glft_spread_skew, GLFTParams, skew_unit
from market_making.mm_sim import bs_delta, bs_gamma, bs_price

# ---- shared strategies ----------------------------------------------------- #
SPOT = st.floats(50.0, 200.0)
STRIKE = st.floats(50.0, 200.0)
TAU = st.floats(0.02, 2.0)
RATE = st.floats(0.0, 0.08)
DIVY = st.floats(0.0, 0.05)
SIGMA = st.floats(0.05, 1.5)


# --------------------------------------------------------------------------- #
# Black-Scholes structure                                                     #
# --------------------------------------------------------------------------- #
@given(S=SPOT, K=STRIKE, tau=TAU, r=RATE, q=DIVY, sigma=SIGMA)
@settings(max_examples=200, deadline=None)
def test_put_call_parity_everywhere(S, K, tau, r, q, sigma):
    c = float(bs_price(S, K, tau, r, sigma, q, "call"))
    p = float(bs_price(S, K, tau, r, sigma, q, "put"))
    assert abs((c - p) - (S * np.exp(-q * tau) - K * np.exp(-r * tau))) < 1e-6


@given(S=SPOT, K=STRIKE, tau=TAU, r=RATE, q=DIVY, sigma=SIGMA)
@settings(max_examples=200, deadline=None)
def test_call_price_bounds(S, K, tau, r, q, sigma):
    c = float(bs_price(S, K, tau, r, sigma, q, "call"))
    lower = max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)
    assert lower - 1e-9 <= c <= S * np.exp(-q * tau) + 1e-9


@given(S=SPOT, K=STRIKE, tau=TAU, r=RATE, q=DIVY, sigma=SIGMA,
       bump=st.floats(0.01, 1.0))
@settings(max_examples=200, deadline=None)
def test_call_price_increases_in_vol(S, K, tau, r, q, sigma, bump):
    lo = float(bs_price(S, K, tau, r, sigma, q, "call"))
    hi = float(bs_price(S, K, tau, r, sigma + bump, q, "call"))
    assert hi >= lo - 1e-9


@given(S=SPOT, K=STRIKE, tau=TAU, r=RATE, q=DIVY, sigma=SIGMA,
       bump=st.floats(0.5, 50.0))
@settings(max_examples=200, deadline=None)
def test_call_monotone_in_spot_and_strike(S, K, tau, r, q, sigma, bump):
    c = float(bs_price(S, K, tau, r, sigma, q, "call"))
    assert float(bs_price(S + bump, K, tau, r, sigma, q, "call")) >= c - 1e-9
    assert float(bs_price(S, K + bump, tau, r, sigma, q, "call")) <= c + 1e-9


@given(S=SPOT, K=STRIKE, tau=TAU, r=RATE, q=DIVY, sigma=SIGMA)
@settings(max_examples=200, deadline=None)
def test_delta_and_gamma_bounds(S, K, tau, r, q, sigma):
    d_call = float(bs_delta(S, K, tau, r, sigma, q, "call"))
    d_put = float(bs_delta(S, K, tau, r, sigma, q, "put"))
    disc = np.exp(-q * tau)
    assert -1e-9 <= d_call <= disc + 1e-9
    assert -disc - 1e-9 <= d_put <= 1e-9
    # parity in delta: d_call - d_put = e^{-q tau}
    assert abs((d_call - d_put) - disc) < 1e-8
    assert float(bs_gamma(S, K, tau, r, sigma, q)) >= -1e-12


@given(K=STRIKE, tau=TAU, r=RATE, q=DIVY, sigma=SIGMA)
@settings(max_examples=150, deadline=None)
def test_iv_round_trip(K, tau, r, q, sigma):
    """price -> implied vol -> price must round-trip wherever the price
    carries usable vol information (away from the intrinsic-value floor)."""
    S = 100.0
    price = vs.bs_call(S, K, tau, r, sigma, q)
    intrinsic = max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)
    assume(price - intrinsic > 1e-5)          # a vol-less price has no IV to find
    iv = vs.iv_from_price(price, S, K, tau, r, q)
    assert np.isfinite(iv)
    assert abs(iv - sigma) < 1e-5


# --------------------------------------------------------------------------- #
# SVI: closed-form derivatives vs finite differences                          #
# --------------------------------------------------------------------------- #
@given(a=st.floats(1e-4, 0.5), b=st.floats(0.01, 1.0),
       rho=st.floats(-0.95, 0.95), m=st.floats(-0.5, 0.5),
       sig=st.floats(0.05, 1.0), k=st.floats(-1.5, 1.5))
@settings(max_examples=200, deadline=None)
def test_svi_derivatives_match_finite_differences(a, b, rho, m, sig, k):
    p = vs.SVIParams(a=a, b=b, rho=rho, m=m, sigma=sig)
    w, wp, wpp = vs.svi_derivatives(k, p)
    h = 1e-5
    w_hi, w_lo = vs.svi_w(k + h, p), vs.svi_w(k - h, p)
    assert abs(wp - (w_hi - w_lo) / (2 * h)) < 1e-6
    assert abs(wpp - (w_hi - 2 * vs.svi_w(k, p) + w_lo) / h**2) < 1e-4
    # raw SVI is convex in k by construction (b sigma^2 / r^3 >= 0)
    assert wpp >= 0.0
    assert w > 0.0


# --------------------------------------------------------------------------- #
# SSVI: the Gatheral-Jacquier no-arbitrage guarantees                          #
# --------------------------------------------------------------------------- #
THETAS = np.geomspace(0.005, 1.0, 8)          # total-variance term structure
K_GRID = np.linspace(-1.5, 1.5, 301)


def _eta_at_butterfly_boundary(rho: float, gamma: float) -> float:
    """Largest eta such that the G-J sufficient butterfly conditions hold on
    THETAS: both conditions are monotone in eta (linear and quadratic), so the
    bound is closed-form per theta and the min over the grid is exact."""
    phi_unit = vs.ssvi_phi(THETAS, vs.SSVIParams(rho=rho, eta=1.0, gamma=gamma))
    lin = 4.0 / (THETAS * phi_unit * (1.0 + abs(rho)))
    quad = np.sqrt(4.0 / (THETAS * phi_unit**2 * (1.0 + abs(rho))))
    return float(min(lin.min(), quad.min()))


@given(rho=st.floats(-0.9, 0.9), gamma=st.floats(0.05, 0.95),
       frac=st.floats(0.05, 0.999))
@settings(max_examples=100, deadline=None)
def test_ssvi_butterfly_conditions_imply_positive_density(rho, gamma, frac):
    """Sufficient conditions satisfied (sampled all the way up to their
    boundary) => Durrleman g >= 0 across the whole surface. This is
    Gatheral-Jacquier Theorem 4.2 checked computationally."""
    eta = frac * _eta_at_butterfly_boundary(rho, gamma)
    p = vs.SSVIParams(rho=rho, eta=eta, gamma=gamma)
    ok, worst = vs.ssvi_butterfly_conditions(THETAS, p)
    assert ok, f"constructive sampling failed: slack {worst}"
    assert vs.min_butterfly_g_ssvi(THETAS, p, K_GRID) >= -1e-9


@given(rho=st.floats(-0.9, 0.9), eta=st.floats(0.05, 3.0),
       gamma=st.floats(0.05, 0.95))
@settings(max_examples=150, deadline=None)
def test_ssvi_theta_phi_monotone(rho, eta, gamma):
    """theta*phi(theta) non-decreasing - the necessary calendar condition,
    which power-law phi with gamma in [0,1] satisfies identically."""
    p = vs.SSVIParams(rho=rho, eta=eta, gamma=gamma)
    tp = THETAS * vs.ssvi_phi(THETAS, p)
    assert np.all(np.diff(tp) >= -1e-12)


@given(rho=st.floats(-0.9, 0.9), gamma=st.floats(0.05, 0.95),
       frac=st.floats(0.05, 0.999))
@settings(max_examples=100, deadline=None)
def test_ssvi_power_law_surface_statically_arbitrage_free(rho, gamma, frac):
    """Gatheral-Jacquier Theorem 4.3: power-law phi with
    eta (1 + |rho|) <= 2 gives a surface free of BOTH butterfly and calendar
    arbitrage. Sampled up to the tighter of that bound and the butterfly
    boundary."""
    eta = frac * min(_eta_at_butterfly_boundary(rho, gamma),
                     2.0 / (1.0 + abs(rho)))
    p = vs.SSVIParams(rho=rho, eta=eta, gamma=gamma)
    assert vs.min_butterfly_g_ssvi(THETAS, p, K_GRID) >= -1e-9
    assert vs.calendar_min_gap(THETAS, p, K_GRID) >= -1e-9


# --------------------------------------------------------------------------- #
# GLFT: structural invariants for every valid parameter set                   #
# --------------------------------------------------------------------------- #
@given(sigma=st.floats(0.5, 50.0), A=st.floats(10.0, 5000.0),
       k=st.floats(0.5, 50.0), gamma=st.floats(0.005, 2.0))
@settings(max_examples=200, deadline=None)
def test_glft_closed_form_structure(sigma, A, k, gamma):
    p = GLFTParams(sigma=sigma, A=A, k=k, gamma=gamma, Q=10)
    q = np.arange(-10, 11)
    db, da = asymptotic_quotes(p, q)
    c1 = float(skew_unit(sigma, A, k, gamma))
    assert c1 > 0.0
    # constant total spread, linear centre shift
    assert np.allclose(db + da, (db + da)[0], rtol=1e-9)
    assert np.allclose((da - db) / 2.0, -c1 * q, rtol=1e-9, atol=1e-12)
    # the simulator adapter is the same quotes, reparameterised
    half, skew = glft_spread_skew(sigma, A, k, gamma)
    assert np.allclose(half + skew * q, db, rtol=1e-9, atol=1e-12)
    assert np.allclose(half - skew * q, da, rtol=1e-9, atol=1e-12)
    # more risk aversion never tightens the skew
    assert float(skew_unit(sigma, A, k, gamma * 1.5)) >= c1
