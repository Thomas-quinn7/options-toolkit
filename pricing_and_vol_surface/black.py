"""Pure Black-Scholes pricer on JAX: prices, the full closed-form Greeks,
and a Newton-Raphson implied-vol solver (derivative via autodiff).

This module imports nothing beyond JAX - no network, no plotting - so any
consumer can price without dragging in the scraping stack. Live-input
helpers (spot, ^IRX rate, chain scraping) and the plotting demos
(``price_heatmap``, ``skew_surface``) live in ``black_data.py``.

Note: importing this module enables JAX float64 process-wide (see below) -
a pricing library silently wrong in the wings would be worse.
"""
import jax

# Full float64 precision: without this JAX defaults to float32, which caps
# implied-vol accuracy around 1e-4 (worse in low-vega wings) and adds ~1e-6
# noise to prices - unacceptable for a pricing library, negligible cost here.
jax.config.update("jax_enable_x64", True)

from jax.scipy.stats import norm as jnorm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import grad, jit  # noqa: E402


# S = Stock Price; K = Strike Price; T = Time Period; r = Risk-free Return;
# Sigma = Implied Volatility; q = Dividends Yield


@jit
def _black_scholes_call(S, K, T, r, sigma, q):
    """JIT-compiled call option pricing."""
    d1 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return jnorm.cdf(d1) * S * jnp.exp(-q * T) - jnorm.cdf(d2) * K * jnp.exp(-r * T)


@jit
def _black_scholes_put(S, K, T, r, sigma, q):
    """JIT-compiled put option pricing."""
    d1 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return K * jnp.exp(-r * T) * jnorm.cdf(-d2) - S * jnorm.cdf(-d1) * jnp.exp(-q * T)


def black_scholes(S, K, T, r, sigma, q=0, otype="call"):
    """Black-Scholes option pricing with JIT optimization."""
    if otype == "call":
        return _black_scholes_call(S, K, T, r, sigma, q)
    elif otype == "put":
        return _black_scholes_put(S, K, T, r, sigma, q)
    else:
        raise ValueError("otype must be 'call' or 'put'")


def diff_function(S, K, T, r, sigma_est, price, q=0, otype="call"):
    """Difference function for implied volatility (not JIT due to string argument)."""
    theoretical = black_scholes(S, K, T, r, sigma_est, q, otype)
    return theoretical - price


def implied_volatility(
    S, K, sigma_est, price, r=0.045, T=1, q=0, otype="call", E=0.01, max_iter=40
):
    """Newton-Raphson implied vol solver.

    E is an *absolute* price tolerance: tighten it (e.g. 1e-6) for low-premium
    OTM options, otherwise sigma_est may be accepted unchanged. If Newton stalls
    from the caller's guess (vega ~ 0 far from the money) it is retried from a
    few standard starts; returns nan when no start converges (e.g. price
    outside no-arb bounds) rather than a stale or diverged guess.

    Spot (S) and the risk-free rate (r) are parameters, not fetches: use
    black_data.stock_data() / black_data.get_riskfree_rate() at the call
    site for live inputs.
    """
    loss_grad = grad(diff_function, argnums=4)

    def newton(sigma):
        for _ in range(max_iter):
            diff = diff_function(S, K, T, r, sigma, price, q, otype)
            if not jnp.isfinite(diff):
                return float("nan")
            if abs(diff) <= E:
                return sigma
            diff_grad = loss_grad(S, K, T, r, sigma, price, q, otype)
            if not jnp.isfinite(diff_grad) or diff_grad == 0:
                return float("nan")
            # Keep the iterate in a sane vol range so one bad Newton step
            # (tiny vega) cannot fling it negative or to infinity.
            sigma = jnp.clip(sigma - diff / diff_grad, 1e-4, 10.0)
        diff = diff_function(S, K, T, r, sigma, price, q, otype)
        return sigma if abs(diff) <= E else float("nan")

    for start in (sigma_est, 0.2, 0.5, 1.0, 2.0):
        result = newton(start)
        if jnp.isfinite(result):
            return result
    return float("nan")


@jit
def _delta_call(S, K, T, r, sigma, q):
    d1 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * jnp.sqrt(T))
    return jnorm.cdf(d1) * jnp.exp(-q * T)


@jit
def _delta_put(S, K, T, r, sigma, q):
    d1 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * jnp.sqrt(T))
    return (jnorm.cdf(d1) - 1) * jnp.exp(-q * T)


@jit
def _gamma(S, K, T, r, sigma, q):
    d1 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * jnp.sqrt(T))
    return jnorm.pdf(d1) * jnp.exp(-q * T) / (S * sigma * jnp.sqrt(T))


@jit
def _theta_call(S, K, T, r, sigma, q):
    d1 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    term1 = -S * jnorm.pdf(d1) * sigma * jnp.exp(-q * T) / (2 * jnp.sqrt(T))
    term2 = q * S * jnorm.cdf(d1) * jnp.exp(-q * T)
    term3 = -r * K * jnp.exp(-r * T) * jnorm.cdf(d2)
    return (term1 + term2 + term3) / 365


@jit
def _theta_put(S, K, T, r, sigma, q):
    d1 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    term1 = -S * jnorm.pdf(d1) * sigma * jnp.exp(-q * T) / (2 * jnp.sqrt(T))
    term2 = -q * S * jnorm.cdf(-d1) * jnp.exp(-q * T)
    term3 = r * K * jnp.exp(-r * T) * jnorm.cdf(-d2)
    return (term1 + term2 + term3) / 365


@jit
def _vega(S, K, T, r, sigma, q):
    d1 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (sigma * jnp.sqrt(T))
    return S * jnorm.pdf(d1) * jnp.sqrt(T) * jnp.exp(-q * T) / 100


@jit
def _rho_call(S, K, T, r, sigma, q):
    d2 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (
        sigma * jnp.sqrt(T)
    ) - sigma * jnp.sqrt(T)
    return K * T * jnp.exp(-r * T) * jnorm.cdf(d2) / 100


@jit
def _rho_put(S, K, T, r, sigma, q):
    d2 = (jnp.log(S / K) + (r - q + (sigma**2) / 2) * T) / (
        sigma * jnp.sqrt(T)
    ) - sigma * jnp.sqrt(T)
    return -K * T * jnp.exp(-r * T) * jnorm.cdf(-d2) / 100


def greeks(S, K, T, r, sigma, q=0, otype="call"):
    """Optimized Greeks calculation using pre-compiled JIT functions."""
    if otype == "call":
        delta = _delta_call(S, K, T, r, sigma, q)
        theta = _theta_call(S, K, T, r, sigma, q)
        rho = _rho_call(S, K, T, r, sigma, q)
    elif otype == "put":
        delta = _delta_put(S, K, T, r, sigma, q)
        theta = _theta_put(S, K, T, r, sigma, q)
        rho = _rho_put(S, K, T, r, sigma, q)
    else:
        raise ValueError("otype must be 'call' or 'put'")

    gamma = _gamma(S, K, T, r, sigma, q)
    vega = _vega(S, K, T, r, sigma, q)

    return delta, gamma, theta, vega, rho
