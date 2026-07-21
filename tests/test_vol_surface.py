"""Tests for the arbitrage-free vol surface (SVI / SSVI).

The load-bearing tests are the no-arbitrage ones: the fitted SSVI surface has a
non-negative implied density everywhere (Durrleman g >= 0) and total variance
non-decreasing in maturity, and the check correctly flags a surface that
violates them.

Run:  python -m pytest tests/test_vol_surface.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing_and_vol_surface"))

import vol_surface as V  # noqa: E402


# --------------------------------------------------------------------------- #
# SVI geometry                                                                #
# --------------------------------------------------------------------------- #
def test_svi_derivatives_match_finite_difference():
    p = V.SVIParams(a=0.02, b=0.1, rho=-0.4, m=0.0, sigma=0.1)
    k = np.linspace(-0.5, 0.5, 50)
    w, wp, wpp = V.svi_derivatives(k, p)
    h = 1e-5
    fd1 = (V.svi_w(k + h, p) - V.svi_w(k - h, p)) / (2 * h)
    fd2 = (V.svi_w(k + h, p) - 2 * V.svi_w(k, p) + V.svi_w(k - h, p)) / h**2
    assert np.allclose(wp, fd1, atol=1e-4)
    assert np.allclose(wpp, fd2, atol=1e-3)


def test_svi_recovers_known_slice():
    p = V.SVIParams(a=0.02, b=0.15, rho=-0.3, m=0.05, sigma=0.12)
    k = np.linspace(-0.5, 0.4, 25)
    w = V.svi_w(k, p)
    fit = V.fit_svi_slice(k, w)
    assert np.sqrt(np.mean((V.svi_w(k, fit) - w) ** 2)) < 1e-4


# --------------------------------------------------------------------------- #
# SSVI recovery and arbitrage-freeness - the core guarantees                  #
# --------------------------------------------------------------------------- #
def test_ssvi_recovers_and_is_arbitrage_free():
    mats, ks, ivs, true, thetas = V.synthetic_surface(seed=1, noise=0.002)
    ws = [iv**2 * T for iv, T in zip(ivs, mats)]
    theta_fit = np.array([float(np.interp(0.0, k, w)) for k, w in zip(ks, ws)])
    p = V.fit_ssvi(ks, ws, theta_fit)

    # parameter recovery
    assert abs(p.rho - true.rho) < 0.1
    assert abs(p.gamma - true.gamma) < 0.15

    # no butterfly arbitrage: density >= 0 everywhere
    k_grid = np.linspace(-0.8, 0.6, 400)
    assert V.min_butterfly_g_ssvi(theta_fit, p, k_grid) >= 0.0
    # no calendar arbitrage: total variance non-decreasing in maturity
    assert V.calendar_min_gap(theta_fit, p, k_grid) >= 0.0


def test_butterfly_check_flags_bad_surface():
    """An extreme SSVI must be caught by both the parametric and density checks."""
    bad = V.SSVIParams(rho=-0.9, eta=5.0, gamma=0.4)
    thetas = np.array([0.02, 0.04, 0.08])
    k_grid = np.linspace(-0.8, 0.6, 400)
    ok, slack = V.ssvi_butterfly_conditions(thetas, bad)
    assert not ok and slack < 0
    assert V.min_butterfly_g_ssvi(thetas, bad, k_grid) < 0.0


def test_calendar_check_flags_decreasing_variance():
    p = V.SSVIParams(rho=-0.3, eta=1.0, gamma=0.4)
    thetas = np.array([0.08, 0.04, 0.02])  # DEcreasing -> calendar arbitrage
    k_grid = np.linspace(-0.5, 0.5, 200)
    assert V.calendar_min_gap(thetas, p, k_grid) < 0.0


def test_naive_interpolation_admits_arbitrage():
    """The demonstration is real: a cubic spline through noisy quotes has g<0."""
    from scipy.interpolate import CubicSpline

    mats, ks, ivs, true, thetas = V.synthetic_surface(seed=0)
    mid = len(mats) // 2
    k, iv, T = ks[mid], ivs[mid], mats[mid]
    w = iv**2 * T
    o = np.argsort(k)
    sp = CubicSpline(k[o], w[o])
    kk = np.linspace(k.min(), k.max(), 400)
    g = V.durrleman_g_from_w(kk, sp(kk), sp(kk, 1), sp(kk, 2))
    assert g.min() < 0.0  # butterfly arbitrage the SSVI surface would remove


# --------------------------------------------------------------------------- #
# IV inversion                                                                #
# --------------------------------------------------------------------------- #
def test_vega_spread_weights_favour_atm():
    k = np.linspace(-0.6, 0.4, 25)
    iv = 0.2 * np.ones_like(k)
    spread = 0.006 + 0.1 * np.abs(k)   # wider in the wings
    w = V.vega_spread_weights(k, iv, 0.5, spread)
    atm = int(np.argmin(np.abs(k)))
    assert w[atm] > w[0] and w[atm] > w[-1]


def test_weighted_calibration_improves_atm_accuracy():
    """Averaged over draws, vega/spread weighting fits ATM better than OLS."""
    atm_u, atm_w = [], []
    for seed in range(15):
        res = V.fit_weighted_vs_unweighted(V.heteroskedastic_slice(seed=seed))
        atm_u.append(res["atm_err_unw"])
        atm_w.append(res["atm_err_wt"])
    assert np.mean(atm_w) < np.mean(atm_u)


def test_band_fit_respects_the_quotes():
    """Quote noise can put a band entirely on the wrong side of value, so even
    the TRUE smile misses some bands - that rate is the irreducible floor. The
    band fit must sit at that floor (and not above the mid fit)."""
    viol_mid, viol_band, viol_true = [], [], []
    for seed in range(10):
        res = V.fit_band_vs_mid(V.heteroskedastic_slice(seed=seed))
        viol_mid.append(res["viol_mid"])
        viol_band.append(res["viol_band"])
        viol_true.append(res["viol_true"])
    assert np.mean(viol_band) <= np.mean(viol_true) + 0.02  # at the achievable floor
    assert np.mean(viol_band) <= np.mean(viol_mid)


def test_band_fit_improves_liquid_region_accuracy():
    """Averaged over draws, fitting the band beats fitting the point mid where
    it matters - ATM and the liquid region - with no weighting scheme at all."""
    atm_mid, atm_band, liq_mid, liq_band = [], [], [], []
    for seed in range(10):
        res = V.fit_band_vs_mid(V.heteroskedastic_slice(seed=seed))
        atm_mid.append(res["atm_err_mid"])
        atm_band.append(res["atm_err_band"])
        liq_mid.append(res["liq_rms_mid"])
        liq_band.append(res["liq_rms_band"])
    assert np.mean(atm_band) < np.mean(atm_mid)
    assert np.mean(liq_band) < np.mean(liq_mid)


def test_band_fit_can_be_pushed_arbitrage_free():
    """With the butterfly penalty on, the band-fitted slice has g >= 0."""
    s = V.heteroskedastic_slice(seed=3)
    T = s["T"]
    w_bid = np.maximum(s["iv_obs"] - 0.5 * s["spread_iv"], 1e-4) ** 2 * T
    w_ask = (s["iv_obs"] + 0.5 * s["spread_iv"]) ** 2 * T
    p = V.fit_svi_slice_band(s["k"], w_bid, w_ask, butterfly_penalty=10.0)
    k_grid = np.linspace(s["k"].min() - 0.1, s["k"].max() + 0.1, 300)
    w, wp, wpp = V.svi_derivatives(k_grid, p)
    assert V.durrleman_g_from_w(k_grid, w, wp, wpp).min() >= -1e-9


def test_ssvi_band_fit_recovers_and_is_arb_free():
    """The global SSVI band fit must recover the true parameters from banded
    quotes of every maturity and remain butterfly/calendar arbitrage-free."""
    mats, ks, ivs, iv_bids, iv_asks, true, _ = V.synthetic_surface_bands(seed=1)
    w_bids = [iv**2 * T for iv, T in zip(iv_bids, mats)]
    w_asks = [iv**2 * T for iv, T in zip(iv_asks, mats)]
    w_mids = [0.5 * (b + a) for b, a in zip(w_bids, w_asks)]
    thetas = np.array([float(np.interp(0.0, k, w)) for k, w in zip(ks, w_mids)])
    p = V.fit_ssvi_band(ks, w_bids, w_asks, thetas)

    assert abs(p.rho - true.rho) < 0.15
    assert abs(p.gamma - true.gamma) < 0.2
    k_grid = np.linspace(-0.8, 0.6, 400)
    assert V.min_butterfly_g_ssvi(thetas, p, k_grid) >= 0.0
    assert V.calendar_min_gap(thetas, p, k_grid) >= 0.0


def test_ssvi_band_fit_not_worse_than_mid_against_truth():
    """Averaged over draws, the band-fitted global surface should be at least
    as close to the TRUE surface as the mid-fitted one in total-variance RMSE."""
    def true_rmse(p, mats, ks, true, atm_vol=0.20):
        errs = []
        for k, T in zip(ks, mats):
            th = atm_vol**2 * T
            errs.append(V.ssvi_w(k, th, p) - V.ssvi_w(k, th, true))
        return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))

    rmse_mid, rmse_band = [], []
    for seed in range(5):
        mats, ks, ivs, iv_bids, iv_asks, true, _ = V.synthetic_surface_bands(seed=seed)
        w_bids = [iv**2 * T for iv, T in zip(iv_bids, mats)]
        w_asks = [iv**2 * T for iv, T in zip(iv_asks, mats)]
        w_mids = [0.5 * (b + a) for b, a in zip(w_bids, w_asks)]
        thetas = np.array([float(np.interp(0.0, k, w)) for k, w in zip(ks, w_mids)])
        rmse_mid.append(true_rmse(V.fit_ssvi(ks, w_mids, thetas), mats, ks, true))
        rmse_band.append(true_rmse(V.fit_ssvi_band(ks, w_bids, w_asks, thetas), mats, ks, true))
    assert np.mean(rmse_band) <= np.mean(rmse_mid) * 1.05


def test_iv_from_price_roundtrip():
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.02, 0.25
    price = V.bs_call(S, K, T, r, sigma)
    assert abs(V.iv_from_price(price, S, K, T, r) - sigma) < 1e-4
    # put side via parity
    put = price - S + K * np.exp(-r * T)
    assert abs(V.iv_from_price(put, S, K, T, r, otype="put") - sigma) < 1e-4


def test_cross_check_black_py():
    pytest.importorskip("jax")
    import black  # noqa: E402

    for S, K, T, r, sigma in [(100, 100, 0.5, 0.02, 0.2), (95, 110, 0.25, 0.0, 0.35)]:
        assert np.isclose(V.bs_call(S, K, T, r, sigma), float(black.black_scholes(S, K, T, r, sigma)), atol=1e-6)
