"""GLFT optimal market-making quotes (Gueant-Lehalle-Fernandez-Tapia).

The simulator in ``mm_sim.py`` quotes with two hand-picked knobs: a fixed
``half_spread`` and a linear inventory skew ``skew_coef``. This module replaces
the hand-picking with the *optimal* spread and skew for exactly the fill model
the simulator already uses (arrival intensity ``lambda = A * exp(-k * delta)``),
by solving the Avellaneda-Stoikov problem in the form of
Gueant, Lehalle & Fernandez-Tapia, "Dealing with the inventory risk: a solution
to the market making problem" (arXiv:1105.3115, Math. Fin. Econ. 2013).

The model
---------
Mid-price ``dS = sigma dW`` (arithmetic), the MM posts a bid at ``S - delta_b``
and an ask at ``S + delta_a``, each filled with intensity ``A exp(-k delta)``
for one unit; inventory q is capped at ``|q| <= Q``; the desk maximises CARA
expected utility ``E[-exp(-gamma * W_T)]``, with terminal inventory optionally
liquidated at a per-unit penalty ``b``. GLFT's contribution is that the HJB
equation *linearises*: with

    alpha = k * gamma * sigma**2 / 2
    eta   = A * (1 + gamma/k) ** -(1 + k/gamma)

the value-function transform ``v(t)`` (one entry per inventory level) solves the
LINEAR ODE system  ``v'_q = alpha q^2 v_q - eta (v_{q+1} + v_{q-1})`` with
``v_q(T) = exp(-k b |q|)`` - i.e. ``v(t) = expm(-M (T-t)) v(T)`` for the
symmetric tridiagonal matrix M - and the optimal quotes are pure functions of
neighbouring ratios of v:

    delta_b(t, q) = (1/k) ln( v_q / v_{q+1} ) + (1/gamma) ln(1 + gamma/k)
    delta_a(t, q) = (1/k) ln( v_q / v_{q-1} ) + (1/gamma) ln(1 + gamma/k)

Far from the terminal time the quotes converge to the stationary quotes built
from the ground eigenvector of M (``ergodic_quotes``), for which the paper gives
the closed-form approximation practitioners actually implement
(``asymptotic_quotes``):

    delta_b(q) ~ c0 + (2q+1)/2 * c1        c0 = (1/gamma) ln(1 + gamma/k)
    delta_a(q) ~ c0 - (2q-1)/2 * c1        c1 = sqrt( sigma^2 gamma / (2 k A)
                                                  * (1 + gamma/k)^(1 + k/gamma) )

i.e. a *constant* total spread ``2 c0 + c1`` and a reservation price shifted
linearly by inventory, ``S - q c1``. That (half-spread, skew) pair is what
``glft_spread_skew`` hands to the simulator.

Honest caveats (state these, don't hide them)
---------------------------------------------
* Tractability needs CARA utility and exponential fill intensities - both are
  modelling choices, not facts about markets.
* The closed-form quotes are an asymptotic approximation: exact far from
  expiry, far from the inventory bound, in the continuum-inventory limit.
  ``ergodic_quotes`` is the exact stationary answer; the demo and tests
  measure the gap instead of assuming it away.
* The model quotes a *single asset with Gaussian mid*. ``mm_sim.py`` maps an
  option onto it through the option's instantaneous dollar vol
  ``|delta| * sigma * S`` - the risk of the raw (unhedged) option inventory.
  A delta-hedged desk carries less risk than that, so the mapping is
  conservative; gamma is the honest knob, the derived *structure* (spread and
  skew from A, k, sigma, gamma) is the point.

Run ``python market_making/glft.py`` to print a quote table and write
``charts/market_making/glft_quotes.png`` (exact finite-horizon quotes converging to the
ergodic ones, and the closed form's accuracy against both).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal, expm

# Keep `python market_making/glft.py` working from a plain clone: repo root on
# the path so `import plotstyle` resolves without `pip install -e .`.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@dataclass(frozen=True)
class GLFTParams:
    sigma: float            # arithmetic (dollar) vol of the mid, per sqrt(year)
    A: float                # arrival intensity at zero quote distance (fills/year/side)
    k: float                # intensity decay per unit quote distance (1/price)
    gamma: float            # CARA risk aversion (1/price)
    Q: int = 25             # hard inventory bound |q| <= Q
    liq_penalty: float = 0.0  # terminal liquidation cost b per unit: l(q) = b|q|

    def __post_init__(self):
        for name in ("sigma", "A", "k", "gamma"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.Q < 1:
            raise ValueError("Q must be >= 1")
        if self.liq_penalty < 0:
            raise ValueError("liq_penalty must be >= 0")


# --------------------------------------------------------------------------- #
# Closed-form constants                                                       #
# --------------------------------------------------------------------------- #
def base_half_spread(gamma: float, k: float) -> float:
    """c0 = (1/gamma) ln(1 + gamma/k): the risk-neutral-inventory half quote."""
    return np.log1p(gamma / k) / gamma


def skew_unit(sigma, A: float, k: float, gamma: float):
    """c1: reservation-price shift per unit of inventory (vectorised in sigma)."""
    sigma = np.asarray(sigma, dtype=float)
    return np.sqrt(sigma**2 * gamma / (2.0 * k * A) * (1.0 + gamma / k) ** (1.0 + k / gamma))


def glft_spread_skew(sigma, A: float, k: float, gamma: float):
    """The (half_spread, skew) pair the simulator consumes, vectorised in sigma.

    Quotes reconstruct as  bid = fair - skew*q - half_spread  and
    ask = fair - skew*q + half_spread, which is identical to the paper's
    delta_b(q) = c0 + (2q+1)/2 c1, delta_a(q) = c0 - (2q-1)/2 c1.
    """
    c1 = skew_unit(sigma, A, k, gamma)
    return base_half_spread(gamma, k) + 0.5 * c1, c1


def asymptotic_quotes(p: GLFTParams, q):
    """Closed-form stationary quote distances (delta_b, delta_a) at inventory q.

    The approximation knows nothing about the bound Q; callers mask q = +/-Q
    themselves. Distances can go negative at large |q| - quoting through fair
    to shed inventory is genuine model behaviour, not an error.
    """
    q = np.asarray(q, dtype=float)
    c0 = base_half_spread(p.gamma, p.k)
    c1 = float(skew_unit(p.sigma, p.A, p.k, p.gamma))
    return c0 + (2.0 * q + 1.0) / 2.0 * c1, c0 - (2.0 * q - 1.0) / 2.0 * c1


# --------------------------------------------------------------------------- #
# Exact solution: v(t) = expm(-M tau) v(T) on the inventory grid              #
# --------------------------------------------------------------------------- #
def inventory_grid(p: GLFTParams) -> np.ndarray:
    return np.arange(-p.Q, p.Q + 1)


def _coefficients(p: GLFTParams) -> tuple[float, float]:
    alpha = 0.5 * p.k * p.gamma * p.sigma**2
    eta = p.A * (1.0 + p.gamma / p.k) ** -(1.0 + p.k / p.gamma)
    return alpha, eta


def _generator(p: GLFTParams) -> np.ndarray:
    """The (2Q+1)x(2Q+1) symmetric tridiagonal M with v(t) = expm(-M tau) v(T)."""
    q = inventory_grid(p)
    alpha, eta = _coefficients(p)
    M = np.diag(alpha * q.astype(float) ** 2)
    off = np.arange(2 * p.Q)
    M[off, off + 1] = -eta
    M[off + 1, off] = -eta
    return M


def value_function(p: GLFTParams, tau: float) -> np.ndarray:
    """v(T - tau) over the inventory grid, up to a positive scale factor.

    Computed by full eigendecomposition of the symmetric tridiagonal M (exact,
    and stable for any tau: the spectrum is shifted by its smallest eigenvalue
    before exponentiating, which only rescales v - the quotes depend on ratios
    of neighbouring entries, so the scale is irrelevant).
    """
    if tau < 0:
        raise ValueError("tau must be >= 0")
    qs = inventory_grid(p)
    alpha, eta = _coefficients(p)
    diag = alpha * qs.astype(float) ** 2
    lam, U = eigh_tridiagonal(diag, np.full(2 * p.Q, -eta))
    vT = np.exp(-p.k * p.liq_penalty * np.abs(qs))
    v = U @ (np.exp(-(lam - lam[0]) * tau) * (U.T @ vT))
    return v / v.max()


# Entries of v this far (relatively) below the peak are beyond double-precision
# resolution - the true values decay like a Gaussian in q and can reach 1e-100
# while the eigendecomposition's roundoff noise sits near 1e-14. Quotes at such
# inventory levels are reported as NaN rather than as noise; those states carry
# e^-25-ish occupation probability, so nothing practical is lost.
_RESOLUTION = 1e-12


def _quotes_from_profile(v: np.ndarray, k: float, c0: float):
    """Neighbour-ratio quotes from a value profile, NaN where v is unresolved."""
    ok = v > _RESOLUTION * v.max()
    n = v.size
    delta_b = np.full(n, np.nan)
    delta_a = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pair = ok[:-1] & ok[1:]
        ratio = np.where(pair, v[:-1] / np.where(pair, v[1:], 1.0), np.nan)
        delta_b[:-1] = np.where(pair, np.log(ratio) / k + c0, np.nan)
        delta_a[1:] = np.where(pair, -np.log(ratio) / k + c0, np.nan)
    return delta_b, delta_a


def exact_quotes(p: GLFTParams, tau: float):
    """Exact quote distances at time-to-terminal tau.

    Returns (q_grid, delta_b, delta_a); delta_b is NaN at q = Q (no bid quoted
    at the long bound), delta_a is NaN at q = -Q, and both are NaN at levels
    whose value-function weight is numerically unresolvable (see _RESOLUTION).
    """
    v = value_function(p, tau)
    delta_b, delta_a = _quotes_from_profile(v, p.k, base_half_spread(p.gamma, p.k))
    return inventory_grid(p), delta_b, delta_a


def ergodic_quotes(p: GLFTParams):
    """Exact stationary (tau -> infinity) quotes, from the ground eigenvector of M.

    For large tau, v(t) aligns with the eigenvector of the smallest eigenvalue
    (positive by Perron-Frobenius: expm(-M tau) has positive entries), so the
    stationary quotes are the neighbour-ratio formula on that eigenvector. This
    is what ``asymptotic_quotes`` approximates in closed form.
    """
    qs = inventory_grid(p)
    alpha, eta = _coefficients(p)
    diag = alpha * qs.astype(float) ** 2
    lam, vecs = eigh_tridiagonal(diag, np.full(2 * p.Q, -eta),
                                 select="i", select_range=(0, 0))
    f = vecs[:, 0]
    f = f * np.sign(f[p.Q])            # Perron eigenvector, fixed positive
    if f.max() <= 0 or np.any(f < -_RESOLUTION * f.max()):
        raise RuntimeError("ground eigenvector not positive - numerical failure")
    delta_b, delta_a = _quotes_from_profile(f, p.k, base_half_spread(p.gamma, p.k))
    return qs, delta_b, delta_a


def value_function_expm(p: GLFTParams, tau: float) -> np.ndarray:
    """v via scipy's dense expm - the paper's literal formula, kept as a
    cross-check for the eigendecomposition route (tests compare the two).
    Overflows for large eta*tau, which is exactly why ``value_function``
    shifts the spectrum first."""
    qs = inventory_grid(p)
    vT = np.exp(-p.k * p.liq_penalty * np.abs(qs))
    v = expm(-_generator(p) * tau) @ vT
    return v / v.max()


# --------------------------------------------------------------------------- #
# Demo                                                                        #
# --------------------------------------------------------------------------- #
def main():
    import os
    outdir = os.path.join(os.path.dirname(__file__), "..", "charts", "market_making")
    os.makedirs(outdir, exist_ok=True)

    # The simulator's own fill model, mapped to its ATM option: an ATM call at
    # S=100, sigma_impl=0.20 has |delta| ~ 0.55, so dollar vol ~ 11 $/sqrt(y).
    p = GLFTParams(sigma=11.0, A=250.0, k=8.0, gamma=0.1, Q=25)
    c0 = base_half_spread(p.gamma, p.k)
    c1 = float(skew_unit(p.sigma, p.A, p.k, p.gamma))
    half, skew = glft_spread_skew(p.sigma, p.A, p.k, p.gamma)

    print("GLFT optimal quotes (arXiv:1105.3115) on the simulator's fill model")
    print(f"  sigma={p.sigma} $/sqrt(y)  A={p.A}/y  k={p.k}/$  gamma={p.gamma}/$  Q={p.Q}")
    print(f"  c0 (base half quote)      = {c0:.4f} $")
    print(f"  c1 (skew per contract)    = {c1:.4f} $")
    print(f"  -> half_spread = c0 + c1/2 = {float(half):.4f} $   skew = c1 = {float(skew):.4f} $")
    print(f"  -> total spread 2*c0 + c1  = {2 * c0 + c1:.4f} $  (inventory-independent)")

    qs, db_erg, da_erg = ergodic_quotes(p)
    db_cf, da_cf = asymptotic_quotes(p, qs)
    print("\n  q     delta_b(erg)  delta_b(closed)   delta_a(erg)  delta_a(closed)")
    for qi in (-6, -3, -1, 0, 1, 3, 6):
        i = qi + p.Q
        print(f"  {qi:+3d}   {db_erg[i]:12.4f}  {db_cf[i]:15.4f}   "
              f"{da_erg[i]:12.4f}  {da_cf[i]:15.4f}")
    print("  (closed form vs exact stationary: tight near flat inventory, drifting")
    print("   apart at larger |q|; at this parameter set the stationary book lives")
    print("   within a couple of contracts of flat, so the closed form is the one")
    print("   the desk actually quotes with)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import plotstyle as ps
        ps.apply_style()
    except Exception:
        print("\n[matplotlib not available - skipped the figure]")
        return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: finite-horizon bid quotes converging to the ergodic ones. The
    # relaxation rate is ~2*sqrt(alpha*eta) (~130/yr here), so the skew fades
    # within the last few trading days and is stationary before that.
    taus = [0.0005, 0.002, 0.008, 0.04]
    show = slice(p.Q - 8, p.Q + 8)        # inventories with resolved quotes
    for i, tau in enumerate(taus):
        _, db, _ = exact_quotes(p, tau)
        axA.plot(qs[show], db[show], "-o", ms=3, color=ps.series_color(i),
                 label=f"exact, tau = {tau * 252:.2g} trading days")
    axA.plot(qs[show], db_erg[show], "--", color=ps.INK, lw=2, label="ergodic (exact, tau -> inf)")
    axA.axhline(c0, color=ps.MUTED, ls=":", lw=1, label="c0 (terminal limit)")
    axA.set_xlabel("inventory q (contracts)")
    axA.set_ylabel("optimal bid distance delta_b ($)")
    axA.set_title("Near the terminal time the skew fades to c0;\n"
                  "far from it the quotes are stationary")
    axA.legend(fontsize=8)

    # Panel B: the closed form vs the exact stationary quotes, spread + centre.
    axB.plot(qs[show], (db_erg + da_erg)[show], "-o", ms=3, color=ps.series_color(0),
             label="total spread, exact ergodic")
    axB.axhline(2 * c0 + c1, color=ps.series_color(0), ls="--", lw=1.2,
                label="total spread, closed form (constant)")
    axB.plot(qs[show], ((da_erg - db_erg) / 2)[show], "-o", ms=3, color=ps.series_color(1),
             label="quote-centre shift, exact ergodic")
    axB.plot(qs[show], -c1 * qs[show].astype(float), "--", color=ps.series_color(1), lw=1.2,
             label="centre shift, closed form (-c1 q)")
    axB.axhline(0, color=ps.BASELINE, lw=0.8)
    axB.set_xlabel("inventory q (contracts)")
    axB.set_ylabel("$")
    axB.set_title("The optimum is a constant spread plus a linear\n"
                  "inventory skew - the closed form's whole claim")
    axB.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/glft_quotes.png", dpi=130)
    plt.close(fig)
    print(f"\nFigure written to {outdir}/glft_quotes.png")


if __name__ == "__main__":
    main()
