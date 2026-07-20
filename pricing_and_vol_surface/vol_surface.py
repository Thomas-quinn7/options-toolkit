"""Arbitrage-free implied-volatility surface (SVI / SSVI).

The repo already plots a *single-snapshot* interpolated IV surface (`black.py:
skew_surface`). That is a picture, not a surface: nothing stops it implying
negative probabilities (butterfly arbitrage) or a total variance that falls with
maturity (calendar arbitrage). This module fits a real, **arbitrage-free**
surface and proves it.

What it does
------------
* **SVI slice** (Gatheral) - fits one smile per expiry in *total implied
  variance* space, w(k) = sigma_BS(k)^2 * T, with analytic first/second
  derivatives.
* **SSVI surface** (Gatheral-Jacquier 2014) - a global surface tied together by
  the ATM total-variance term structure theta(T); arbitrage-free under explicit
  conditions on its parameters.
* **No-arbitrage proof, computed not assumed:**
    - Butterfly: the Durrleman function g(k) >= 0 for all k  <=>  the implied
      risk-neutral density is non-negative. Checked numerically on a grid.
    - Calendar: total variance w(k, T) is non-decreasing in T for every k.
* **The failure it prevents:** a naive interpolation of noisy market IVs
  (a cubic spline through the quotes) produces g(k) < 0 - a butterfly arbitrage -
  which the constrained SSVI surface removes. `main()` demonstrates this.

Everything here is numpy/scipy and needs no market data; `main()` fits to a
synthetic surface with known parameters and verifies recovery + arb-freeness.
The repo's autodiff pricer is `black.py`; `iv_from_price` here is a small
self-contained Brent inverter for turning real mid prices into IVs (so the
surface is built from prices, not from yfinance's own `impliedVolatility`
field), and a test cross-checks it against `black.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, least_squares


# --------------------------------------------------------------------------- #
# SVI slice (raw parameterisation) and its no-arbitrage geometry               #
# --------------------------------------------------------------------------- #
@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_w(k, p: SVIParams):
    """Raw-SVI total variance w(k)."""
    k = np.asarray(k, dtype=float)
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x**2 + p.sigma**2))


def svi_derivatives(k, p: SVIParams):
    """Return (w, w', w'') for raw SVI - all closed form."""
    k = np.asarray(k, dtype=float)
    x = k - p.m
    r = np.sqrt(x**2 + p.sigma**2)
    w = p.a + p.b * (p.rho * x + r)
    wp = p.b * (p.rho + x / r)
    wpp = p.b * p.sigma**2 / r**3
    return w, wp, wpp


def _min_durrleman_g_svi(p: SVIParams, k_grid) -> float:
    w, wp, wpp = svi_derivatives(k_grid, p)
    g = (1.0 - k_grid * wp / (2.0 * w)) ** 2 - (wp**2) / 4.0 * (1.0 / w + 0.25) + wpp / 2.0
    return float(np.min(g))


def durrleman_g_from_w(k_grid, w, wp, wpp):
    """Durrleman g on a grid, given w and its derivatives (any parameterisation)."""
    k_grid = np.asarray(k_grid, dtype=float)
    return (1.0 - k_grid * wp / (2.0 * w)) ** 2 - (wp**2) / 4.0 * (1.0 / w + 0.25) + wpp / 2.0


def fit_svi_slice(k, w_market, weights=None, butterfly_penalty: float = 0.0) -> SVIParams:
    """Least-squares fit of a raw-SVI slice to (k, w_market) points.

    ``weights`` (per point) scale the residuals - pass vega/bid-ask weights to
    let liquid, reliable quotes dominate noisy illiquid wings. If
    butterfly_penalty > 0, negative Durrleman g is penalised so the fit is
    pushed into the no-arbitrage region.
    """
    k = np.asarray(k, dtype=float)
    w_market = np.asarray(w_market, dtype=float)
    weights = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    k_grid = np.linspace(k.min() - 0.1, k.max() + 0.1, 120)

    def resid(theta):
        p = SVIParams(*theta)
        r = weights * (svi_w(k, p) - w_market)
        if butterfly_penalty > 0:
            g = _min_durrleman_g_svi(p, k_grid)
            if g < 0:
                r = np.append(r, butterfly_penalty * (-g))
        return r

    w0 = float(np.min(w_market))
    x0 = [max(w0 * 0.5, 1e-6), 0.1, -0.3, 0.0, 0.1]
    lb = [1e-8, 1e-8, -0.999, -2.0, 1e-4]
    ub = [np.inf, np.inf, 0.999, 2.0, 5.0]
    sol = least_squares(resid, x0, bounds=(lb, ub), max_nfev=5000)
    return SVIParams(*sol.x)


# --------------------------------------------------------------------------- #
# SSVI surface (Gatheral-Jacquier)                                             #
# --------------------------------------------------------------------------- #
@dataclass
class SSVIParams:
    rho: float
    eta: float
    gamma: float


def ssvi_phi(theta, p: SSVIParams):
    """Power-law phi(theta) = eta / (theta^gamma (1+theta)^(1-gamma))."""
    theta = np.asarray(theta, dtype=float)
    return p.eta / (theta**p.gamma * (1.0 + theta) ** (1.0 - p.gamma))


def ssvi_w(k, theta, p: SSVIParams):
    """SSVI total variance w(k, theta)."""
    k = np.asarray(k, dtype=float)
    phi = ssvi_phi(theta, p)
    return 0.5 * theta * (1.0 + p.rho * phi * k + np.sqrt((phi * k + p.rho) ** 2 + (1.0 - p.rho**2)))


def ssvi_derivatives(k, theta, p: SSVIParams):
    """(w, w', w'') of SSVI in k at fixed theta (closed form)."""
    k = np.asarray(k, dtype=float)
    phi = float(ssvi_phi(theta, p))
    rho = p.rho
    root = np.sqrt((phi * k + rho) ** 2 + (1.0 - rho**2))
    w = 0.5 * theta * (1.0 + rho * phi * k + root)
    droot = phi * (phi * k + rho) / root
    wp = 0.5 * theta * (rho * phi + droot)
    d2root = phi**2 * (1.0 - rho**2) / root**3
    wpp = 0.5 * theta * d2root
    return w, wp, wpp


def ssvi_butterfly_conditions(theta_grid, p: SSVIParams) -> Tuple[bool, float]:
    """Gatheral-Jacquier sufficient no-butterfly conditions across a theta grid.

    Requires theta*phi*(1+|rho|) < 4 and theta*phi^2*(1+|rho|) <= 4. Returns
    (ok, worst_slack) where worst_slack < 0 means a condition is violated.
    """
    theta_grid = np.asarray(theta_grid, dtype=float)
    phi = ssvi_phi(theta_grid, p)
    c1 = 4.0 - theta_grid * phi * (1.0 + abs(p.rho))          # must be > 0
    c2 = 4.0 - theta_grid * phi**2 * (1.0 + abs(p.rho))       # must be >= 0
    worst = float(min(c1.min(), c2.min()))
    return worst >= 0.0, worst


def fit_ssvi(ks: Sequence[np.ndarray], ws: Sequence[np.ndarray], thetas: Sequence[float],
             weights: Optional[Sequence[np.ndarray]] = None,
             butterfly_penalty: float = 50.0) -> SSVIParams:
    """Fit global SSVI (rho, eta, gamma) to per-maturity (k, w) data.

    thetas are the ATM total variances per maturity (theta_T = w(0, T)).
    ``weights`` is an optional list of per-slice weight arrays (e.g. vega/bid-ask).
    Negative-density (butterfly) violations are penalised.
    """
    thetas = np.asarray(thetas, dtype=float)
    theta_dense = np.linspace(thetas.min(), thetas.max(), 40)
    if weights is None:
        weights = [np.ones_like(k) for k in ks]

    def resid(x):
        p = SSVIParams(rho=x[0], eta=x[1], gamma=x[2])
        parts = []
        for k, w, th, wt in zip(ks, ws, thetas, weights):
            parts.append(wt * (ssvi_w(k, th, p) - w))
        r = np.concatenate(parts)
        ok, slack = ssvi_butterfly_conditions(theta_dense, p)
        if slack < 0:
            r = np.append(r, butterfly_penalty * (-slack))
        return r

    x0 = [-0.3, 1.0, 0.4]
    lb = [-0.999, 1e-3, 1e-3]
    ub = [0.999, 10.0, 0.999]
    sol = least_squares(resid, x0, bounds=(lb, ub), max_nfev=8000)
    return SSVIParams(rho=sol.x[0], eta=sol.x[1], gamma=sol.x[2])


# --------------------------------------------------------------------------- #
# Surface-level no-arbitrage checks (numerical, authoritative)                 #
# --------------------------------------------------------------------------- #
def min_butterfly_g_ssvi(thetas, p: SSVIParams, k_grid) -> float:
    """Minimum Durrleman g over the whole SSVI surface (>=0 => no butterfly arb)."""
    gmin = np.inf
    for th in thetas:
        w, wp, wpp = ssvi_derivatives(k_grid, th, p)
        g = durrleman_g_from_w(k_grid, w, wp, wpp)
        gmin = min(gmin, float(np.min(g)))
    return gmin


def calendar_min_gap(thetas, p: SSVIParams, k_grid) -> float:
    """Min over k and adjacent maturities of w(k,T_{i+1}) - w(k,T_i).

    ``thetas`` must be given in maturity (time) order. The result is >= 0
    everywhere  <=>  no calendar-spread arbitrage (total variance rises with
    maturity). The thetas are NOT sorted here: a theta that falls with maturity
    is itself the arbitrage and must be flagged, not reordered away.
    """
    thetas = np.asarray(thetas, dtype=float)
    worst = np.inf
    for i in range(len(thetas) - 1):
        w_lo = ssvi_w(k_grid, thetas[i], p)
        w_hi = ssvi_w(k_grid, thetas[i + 1], p)
        worst = min(worst, float(np.min(w_hi - w_lo)))
    return worst


# --------------------------------------------------------------------------- #
# Black-Scholes IV inversion (compact, for building surfaces from real prices) #
# --------------------------------------------------------------------------- #
from scipy.stats import norm  # noqa: E402


def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Black-Scholes vega (per unit vol). Largest at the money."""
    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def vega_spread_weights(k, iv, T, spread_iv, S=100.0, r=0.0, q=0.0):
    """Calibration weights = vega / bid-ask-spread, normalised to mean 1.

    Reliable quotes get more weight two ways: vega (an ATM quote carries far more
    information about the smile than a deep-wing one) and 1/spread (a tight
    market is a more trustworthy price than a wide, illiquid one).
    """
    k = np.asarray(k, dtype=float)
    K = S * np.exp(k)
    vega = bs_vega(S, K, T, r, np.asarray(iv, dtype=float), q)
    w = vega / np.maximum(np.asarray(spread_iv, dtype=float), 1e-6)
    return w / w.mean()


def iv_from_price(price, S, K, T, r, q=0.0, otype="call"):
    """Brent-invert a European price to Black-Scholes implied vol.

    Puts are converted to the equivalent call price via put-call parity so a
    single call inverter is used. Returns nan if the price is outside no-arb
    bounds.
    """
    if otype == "put":
        price = price + S * np.exp(-q * T) - K * np.exp(-r * T)  # -> call price
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper = S * np.exp(-q * T)
    if not (intrinsic - 1e-10 <= price <= upper + 1e-10):
        return float("nan")
    try:
        return float(brentq(lambda s: bs_call(S, K, T, r, s, q) - price, 1e-4, 5.0, maxiter=200))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Synthetic ground-truth surface (for the self-contained demo/tests)           #
# --------------------------------------------------------------------------- #
def synthetic_surface(seed=0, noise=0.006, n_strikes=25):
    """A known arbitrage-free SSVI surface, sampled with IV noise.

    ``noise`` is a per-quote IV error (~0.6 vol points by default) and
    ``n_strikes`` a realistic chain density; together they make a naive
    interpolation of the quotes admit butterfly arbitrage while the fitted SSVI
    does not. Returns (maturities, ks_per_mat, iv_market_per_mat, true, thetas).
    """
    rng = np.random.default_rng(seed)
    true = SSVIParams(rho=-0.4, eta=1.0, gamma=0.4)
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    atm_vol = 0.20
    thetas = (atm_vol**2) * maturities  # increasing -> calendar arb-free
    ks, ivs = [], []
    for T, th in zip(maturities, thetas):
        k = np.linspace(-0.6, 0.4, n_strikes)
        w = ssvi_w(k, th, true)
        iv = np.sqrt(w / T) + rng.normal(0.0, noise, size=k.shape)
        ks.append(k)
        ivs.append(iv)
    return maturities, ks, ivs, true, thetas


def heteroskedastic_slice(seed=0, T=0.5, n=25, atm_vol=0.20):
    """One smile with realistic quote quality: tight, reliable near the money;
    noisy and wide (illiquid) in the wings. Returns the truth plus the quotes.
    """
    rng = np.random.default_rng(seed)
    true = SSVIParams(rho=-0.4, eta=1.0, gamma=0.4)
    theta = atm_vol**2 * T
    k = np.linspace(-0.6, 0.4, n)
    w_true = ssvi_w(k, theta, true)
    iv_true = np.sqrt(w_true / T)
    sd = 0.003 + 0.05 * np.abs(k)     # per-quote IV error grows into the wings
    iv_obs = iv_true + rng.normal(0.0, sd)
    spread_iv = 2.0 * sd              # bid-ask (in vol) wider in the wings
    return {"k": k, "T": T, "theta": theta, "w_true": w_true, "iv_true": iv_true,
            "iv_obs": iv_obs, "spread_iv": spread_iv}


def fit_weighted_vs_unweighted(slice_data):
    """Fit an SVI slice to noisy quotes with and without vega/spread weights.

    Weighting sacrifices the noisy wings to nail the liquid, high-vega region -
    which is where you actually price and hedge - so it wins on ATM and
    liquid-region accuracy while (by design) giving up equal-weighted wing fit.
    """
    s = slice_data
    w_obs = s["iv_obs"] ** 2 * s["T"]
    wts = vega_spread_weights(s["k"], s["iv_obs"], s["T"], s["spread_iv"])
    p_unw = fit_svi_slice(s["k"], w_obs)
    p_wt = fit_svi_slice(s["k"], w_obs, weights=wts)

    def rms(x):
        return float(np.sqrt(np.mean(x**2)))

    liq = np.abs(s["k"]) <= 0.2
    return {
        "p_unw": p_unw, "p_wt": p_wt, "weights": wts,
        "atm_err_unw": abs(float(svi_w(0.0, p_unw)) - s["theta"]),
        "atm_err_wt": abs(float(svi_w(0.0, p_wt)) - s["theta"]),
        "liq_rms_unw": rms(svi_w(s["k"][liq], p_unw) - s["w_true"][liq]),
        "liq_rms_wt": rms(svi_w(s["k"][liq], p_wt) - s["w_true"][liq]),
    }


# --------------------------------------------------------------------------- #
# Plotting (guarded) and CLI demo                                              #
# --------------------------------------------------------------------------- #
def _try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def main():
    import os

    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    mats, ks, ivs, true, thetas = synthetic_surface()
    ws = [iv**2 * T for iv, T in zip(ivs, mats)]
    theta_fit = np.array([float(np.interp(0.0, k, w)) for k, w in zip(ks, ws)])

    print("Arbitrage-free vol surface (SVI / SSVI)")
    print(f"  {len(mats)} maturities, {len(ks[0])} strikes each; true SSVI "
          f"rho={true.rho}, eta={true.eta}, gamma={true.gamma}\n")

    # Per-slice SVI
    svi_ps = [fit_svi_slice(k, w, butterfly_penalty=10.0) for k, w in zip(ks, ws)]
    svi_rmse = np.sqrt(np.mean(np.concatenate(
        [(svi_w(k, p) - w) ** 2 for k, w, p in zip(ks, ws, svi_ps)])))

    # Global SSVI
    ssvi_p = fit_ssvi(ks, ws, theta_fit)
    ssvi_rmse = np.sqrt(np.mean(np.concatenate(
        [(ssvi_w(k, th, ssvi_p) - w) ** 2 for k, w, th in zip(ks, ws, theta_fit)])))

    print(f"  per-slice SVI  fit RMSE (total var) = {svi_rmse:.2e}")
    print(f"  global   SSVI  fit RMSE (total var) = {ssvi_rmse:.2e}")
    print(f"  recovered SSVI: rho={ssvi_p.rho:+.3f}, eta={ssvi_p.eta:.3f}, gamma={ssvi_p.gamma:.3f}")

    # No-arbitrage proof
    k_grid = np.linspace(-0.8, 0.6, 400)
    gmin = min_butterfly_g_ssvi(theta_fit, ssvi_p, k_grid)
    cal = calendar_min_gap(theta_fit, ssvi_p, k_grid)
    cond_ok, slack = ssvi_butterfly_conditions(np.linspace(theta_fit.min(), theta_fit.max(), 40), ssvi_p)
    print("\n  No-arbitrage checks on the fitted SSVI surface:")
    print(f"    butterfly: min Durrleman g = {gmin:+.4f}   (>=0 required)  -> {'PASS' if gmin >= 0 else 'FAIL'}")
    print(f"    calendar : min d(total var)/dT gap = {cal:+.4f}   (>=0 required)  -> {'PASS' if cal >= 0 else 'FAIL'}")
    print(f"    GJ sufficient conditions: {'satisfied' if cond_ok else 'violated'} (slack {slack:+.3f})")

    # The failure SSVI prevents: naive interpolation of noisy quotes
    from scipy.interpolate import CubicSpline
    mid = len(mats) // 2
    k_s, w_s = ks[mid], ws[mid]
    order = np.argsort(k_s)
    spline = CubicSpline(k_s[order], w_s[order])
    kk = np.linspace(k_s.min(), k_s.max(), 400)
    w_sp = spline(kk)
    wp_sp = spline(kk, 1)
    wpp_sp = spline(kk, 2)
    g_naive = durrleman_g_from_w(kk, w_sp, wp_sp, wpp_sp)
    print(f"\n  Naive cubic-spline interpolation of the noisy quotes (T={mats[mid]}):")
    print(f"    min Durrleman g = {g_naive.min():+.4f}  -> {'butterfly ARBITRAGE (g<0)' if g_naive.min() < 0 else 'no violation this draw'}")

    # Vega/liquidity-weighted calibration under heteroskedastic quote noise
    sl = heteroskedastic_slice(seed=0)
    wcal = fit_weighted_vs_unweighted(sl)
    print("\n  Vega/liquidity-weighted calibration (noisy wide-spread wings, tight ATM):")
    print(f"    ATM total-variance error : unweighted={wcal['atm_err_unw']:.2e}  weighted={wcal['atm_err_wt']:.2e}")
    print(f"    liquid-region |k|<=0.2   : unweighted={wcal['liq_rms_unw']:.2e}  weighted={wcal['liq_rms_wt']:.2e}")
    print("    weighting nails the liquid/high-vega region (where you price and hedge)")
    print("    by not chasing the noisy wings.")

    plt = _try_plt()
    if plt is None:
        print("\n[matplotlib not available - numeric results above are the proof]")
        return

    # Figure 1: fitted arb-free surface
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    K_grid, T_grid = np.meshgrid(np.linspace(-0.6, 0.4, 40), mats)
    IV = np.zeros_like(K_grid)
    for i, (T, th) in enumerate(zip(mats, theta_fit)):
        IV[i, :] = np.sqrt(ssvi_w(K_grid[i, :], th, ssvi_p) / T)
    ax.plot_surface(K_grid, T_grid, IV, cmap="viridis", alpha=0.9, edgecolor="none")
    for k, iv, T in zip(ks, ivs, mats):
        ax.scatter(k, np.full_like(k, T), iv, color="crimson", s=8)
    ax.set_xlabel("log-moneyness k"); ax.set_ylabel("maturity T"); ax.set_zlabel("implied vol")
    ax.set_title("Arbitrage-free SSVI surface (points = market quotes)")
    fig.tight_layout(); fig.savefig(f"{outdir}/ssvi_surface.png", dpi=130); plt.close(fig)

    # Figure 2: one slice - SVI vs SSVI fit
    fig, ax = plt.subplots(figsize=(7.5, 5))
    k_plot = np.linspace(k_s.min(), k_s.max(), 200)
    ax.scatter(k_s, np.sqrt(np.asarray(w_s) / mats[mid]), color="crimson", s=25, label="market IV", zorder=3)
    ax.plot(k_plot, np.sqrt(svi_w(k_plot, svi_ps[mid]) / mats[mid]), label="SVI slice fit")
    ax.plot(k_plot, np.sqrt(ssvi_w(k_plot, theta_fit[mid], ssvi_p) / mats[mid]), "--", label="SSVI fit")
    ax.set_xlabel("log-moneyness k"); ax.set_ylabel("implied vol")
    ax.set_title(f"Smile fit at T={mats[mid]}"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/slice_fit.png", dpi=130); plt.close(fig)

    # Figure 3: the density check - SSVI g>=0 vs naive spline g<0
    fig, ax = plt.subplots(figsize=(7.5, 5))
    w, wp, wpp = ssvi_derivatives(kk, theta_fit[mid], ssvi_p)
    ax.plot(kk, durrleman_g_from_w(kk, w, wp, wpp), label="SSVI (arb-free)", lw=2)
    ax.plot(kk, g_naive, label="naive cubic spline", lw=1.5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.fill_between(kk, g_naive, 0, where=(g_naive < 0), color="red", alpha=0.25, label="butterfly arbitrage (g<0)")
    ax.set_xlabel("log-moneyness k"); ax.set_ylabel("Durrleman g  (density >= 0 iff g >= 0)")
    ax.set_title("No-arbitrage density check")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/density_check.png", dpi=130); plt.close(fig)

    # Figure 4: vega/liquidity-weighted vs unweighted calibration
    fig, ax = plt.subplots(figsize=(7.5, 5))
    kk2 = np.linspace(sl["k"].min(), sl["k"].max(), 200)
    ax.errorbar(sl["k"], sl["iv_obs"], yerr=sl["spread_iv"], fmt="o", ms=4, color="crimson",
                ecolor="grey", elinewidth=1, capsize=2, alpha=0.8, label="market quotes (bid-ask)")
    ax.plot(kk2, np.sqrt(ssvi_w(kk2, sl["theta"], SSVIParams(-0.4, 1.0, 0.4)) / sl["T"]),
            "k-", lw=1.5, label="true smile")
    ax.plot(kk2, np.sqrt(svi_w(kk2, wcal["p_unw"]) / sl["T"]), "--", color="C0", label="unweighted fit")
    ax.plot(kk2, np.sqrt(svi_w(kk2, wcal["p_wt"]) / sl["T"]), "-", color="C2", label="vega/spread-weighted fit")
    ax.axvspan(-0.2, 0.2, color="green", alpha=0.07, label="liquid region")
    ax.set_xlabel("log-moneyness k"); ax.set_ylabel("implied vol")
    ax.set_title("Weighted calibration hugs the reliable ATM quotes")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/weighted_calibration.png", dpi=130); plt.close(fig)

    print(f"\nFigures written to {os.path.normpath(outdir)}/:")
    print("  ssvi_surface.png, slice_fit.png, density_check.png, weighted_calibration.png")


if __name__ == "__main__":
    main()
