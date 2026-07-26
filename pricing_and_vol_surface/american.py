"""American option pricing: Cox-Ross-Rubinstein binomial tree.

Everything yfinance serves for single names and ETFs is American-exercise,
while this repo's inverter (`vol_surface.iv_from_price`) and the arb scanner's
settlement logic are European. This module closes that gap honestly:

* ``crr_price`` - CRR binomial price, American or European exercise, with a
  continuous dividend yield. Vectorised over strike and vol (broadcast), so a
  whole chain prices in one call. The European mode converges to the
  closed-form Black-Scholes price, which is exactly what the tests pin.
* ``early_exercise_premium`` - American minus European price *on the same
  tree*, so discretisation error cancels and the premium is clean. This is
  the number that turns "we apply European relationships to American options"
  from an unquantified caveat into a measured one: for the short-dated OTM
  quotes the skew scanner consumes, the premium is far inside the bid-ask
  spread (``tests/test_american.py`` asserts it).
* ``american_iv`` / ``american_iv_vector`` - implied vol by inverting the
  binomial price (Brent for one quote, vectorised bisection for a chain).
  ``IV_skew.py --american`` uses the vectorised path.

Model notes, stated not hidden: continuous dividend *yield* only (no discrete
dividend schedule, so early exercise of calls just before an ex-date is
approximated, not modelled); CRR converges O(1/n) with the usual odd/even
oscillation - ``n_steps`` defaults are chosen so tree error is far below
retail bid-ask spreads.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import brentq

ArrayLike = Union[float, np.ndarray]


def _payoff(S, K, otype: str):
    if otype == "call":
        return np.maximum(S - K, 0.0)
    if otype == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("otype must be 'call' or 'put'")


def crr_price(S: float, K: ArrayLike, T: float, r: float, sigma: ArrayLike,
              q: float = 0.0, otype: str = "call", style: str = "american",
              n_steps: int = 400) -> ArrayLike:
    """CRR binomial price, vectorised over strike and vol (broadcast together).

    ``style="european"`` disables the early-exercise comparison and converges
    to Black-Scholes - the tests use that as the correctness anchor.
    """
    if style not in ("american", "european"):
        raise ValueError("style must be 'american' or 'european'")
    K_in, sig_in = np.asarray(K, dtype=float), np.asarray(sigma, dtype=float)
    scalar = (K_in.ndim == 0) and (sig_in.ndim == 0)
    K_b, sig_b = np.broadcast_arrays(np.atleast_1d(K_in), np.atleast_1d(sig_in))
    K_b = K_b.astype(float)
    sig_b = np.maximum(sig_b.astype(float), 1e-8)

    if T <= 0:
        out = _payoff(float(S), K_b, otype)
        return float(out[0]) if scalar else out

    dt = T / n_steps
    u = np.exp(sig_b * np.sqrt(dt))                      # (m,)
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # Tiny vol against a large drift can push p out of [0,1]; clipping keeps
    # the tree a valid lattice (the price limit is then just discounted
    # intrinsic along the drift, which is the correct sigma->0 limit).
    p = np.clip(p, 0.0, 1.0)[:, None]

    j = np.arange(n_steps + 1)
    ST = S * u[:, None] ** j * d[:, None] ** (n_steps - j)
    V = _payoff(ST, K_b[:, None], otype)

    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:, 1:] + (1.0 - p) * V[:, :-1])
        if style == "american":
            Si = S * u[:, None] ** j[: i + 1] * d[:, None] ** (i - j[: i + 1])
            V = np.maximum(V, _payoff(Si, K_b[:, None], otype))

    out = V[:, 0]
    return float(out[0]) if scalar else out


def early_exercise_premium(S: float, K: ArrayLike, T: float, r: float,
                           sigma: ArrayLike, q: float = 0.0,
                           otype: str = "call", n_steps: int = 400) -> ArrayLike:
    """American minus European price on the SAME tree (discretisation cancels)."""
    amer = crr_price(S, K, T, r, sigma, q, otype, "american", n_steps)
    euro = crr_price(S, K, T, r, sigma, q, otype, "european", n_steps)
    return np.maximum(amer - euro, 0.0) if np.ndim(amer) else max(amer - euro, 0.0)


def american_iv(price: float, S: float, K: float, T: float, r: float,
                q: float = 0.0, otype: str = "call",
                n_steps: int = 200) -> float:
    """Implied vol from an American price (Brent on the binomial pricer).

    Returns nan when the price sits outside the attainable range (below the
    sigma->0 limit or above the sigma=5 price) - the same free no-arb gate
    ``iv_from_price`` gives for European quotes.
    """
    lo, hi = 1e-4, 5.0
    p_lo = crr_price(S, K, T, r, lo, q, otype, "american", n_steps)
    p_hi = crr_price(S, K, T, r, hi, q, otype, "american", n_steps)
    if not (p_lo - 1e-10 <= price <= p_hi + 1e-10):
        return float("nan")
    try:
        return float(brentq(
            lambda s: crr_price(S, K, T, r, s, q, otype, "american", n_steps) - price,
            lo, hi, maxiter=100))
    except Exception:
        return float("nan")


def american_iv_vector(prices: np.ndarray, S: float, Ks: np.ndarray, T: float,
                       r: float, q: float = 0.0, otype: str = "call",
                       n_steps: int = 200, n_iter: int = 50) -> np.ndarray:
    """Implied vols for a whole chain side at once (vectorised bisection).

    The binomial price is monotone in sigma, so bisection is exact and every
    quote shares each tree evaluation. Unattainable prices come back nan.
    """
    prices = np.asarray(prices, dtype=float)
    Ks = np.asarray(Ks, dtype=float)
    lo = np.full_like(prices, 1e-4)
    hi = np.full_like(prices, 5.0)
    p_lo = crr_price(S, Ks, T, r, lo, q, otype, "american", n_steps)
    p_hi = crr_price(S, Ks, T, r, hi, q, otype, "american", n_steps)
    ok = np.isfinite(prices) & (prices >= p_lo - 1e-10) & (prices <= p_hi + 1e-10)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        p_mid = crr_price(S, Ks, T, r, mid, q, otype, "american", n_steps)
        below = p_mid < prices
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)

    out = 0.5 * (lo + hi)
    out[~ok] = np.nan
    return out
