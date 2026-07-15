from jax.scipy.stats import norm as jnorm
import yfinance as yf
import jax.numpy as jnp
from jax import grad, jit, vmap
import seaborn as sn
from matplotlib import pyplot as plt
from datetime import datetime, date
import pandas as pd
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata


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


black_scholes_vectorized = vmap(black_scholes, in_axes=(0, 0, 0, 0, 0, 0, None))
black_scholes_batch_strikes = vmap(
    black_scholes, in_axes=(None, 0, None, None, None, None, None)
)
black_scholes_batch_volatilities = vmap(
    black_scholes, in_axes=(None, None, None, None, 0, None, None)
)


def stock_data(stock):
    stock_data = yf.Ticker(stock).history(period="max")
    S = stock_data["Close"].iloc[-1]
    return S


def get_riskfree_rate():
    r_df = yf.Ticker("^IRX").history(period="5d")
    if r_df.empty:
        raise ValueError(
            "No data returned for ^IRX. Check internet connection or ticker."
        )
    y = r_df["Close"].iloc[-1] / 100
    T = 13 / 52
    r = -jnp.log(1 - T * y) / T
    return float(r)


def diff_function(S, K, T, r, sigma_est, price, q=0, otype="call"):
    """Difference function for implied volatility (not JIT due to string argument)."""
    theoretical = black_scholes(S, K, T, r, sigma_est, q, otype)
    return theoretical - price


def implied_volatility(
    stock, K, sigma_est, price, T=1, q=0, otype="call", E=0.01, iter=40
):
    S = stock_data(stock)
    r = get_riskfree_rate()
    iterations = 0
    diff = diff_function(S, K, T, r, sigma_est, price, q, otype)
    loss_grad = grad(diff_function, argnums=4)
    while abs(diff) > E and iterations < iter:
        diff = diff_function(S, K, T, r, sigma_est, price, q, otype)
        diff_grad = loss_grad(S, K, T, r, sigma_est, price, q, otype)
        if diff_grad == 0:
            print("Gradient is zero or invalid; stopping.")
            break

        iterations += 1
        if abs(diff) < E:
            break
        sigma_est = sigma_est - diff / diff_grad
    return sigma_est


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


greeks_vectorized = vmap(greeks, in_axes=(0, 0, 0, 0, 0, 0, None))
greeks_batch_strikes = vmap(greeks, in_axes=(None, 0, None, None, None, None, None))
greeks_batch_volatilities = vmap(
    greeks, in_axes=(None, None, None, None, 0, None, None)
)


def price_heatmap(
    S, K, T=1, r=0.03, sigma=0.1, q=0, otype="call", Pur_Price=0, grid=9, diff=0.2
):
    """Optimized price heatmap using vectorized operations."""
    interval = (2 * diff) / (grid - 1)
    step_p = interval * S
    step_vol = interval * sigma

    min_p = (1 - diff) * S
    min_vol = (1 - diff) * sigma
    diff_prices = jnp.linspace(min_p, min_p + (grid - 1) * step_p, grid)
    diff_vols = jnp.linspace(min_vol, min_vol + (grid - 1) * step_vol, grid)

    P_grid, V_grid = jnp.meshgrid(diff_prices, diff_vols)

    @jit
    def compute_price_matrix(prices, vols):
        return vmap(vmap(lambda p, v: black_scholes(p, K, T, r, v, q, otype)))(
            prices, vols
        )

    price_matrix = compute_price_matrix(P_grid, V_grid) - Pur_Price

    plt.figure(figsize=(8, 6), dpi=200)
    sn.heatmap(
        jnp.array(price_matrix),
        xticklabels=[round(float(p), 2) for p in diff_prices],
        yticklabels=[round(float(v), 3) for v in diff_vols],
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",
    )
    plt.xlabel("Stock Price")
    plt.ylabel("Implied Volatility (σ)")
    if Pur_Price == 0:
        plt.title(f"Option Price Heatmap (K={K}, T={T}, {otype.capitalize()})")
    else:
        plt.title(
            f"Option Profit Heatmap (K={K}, T={T}, {otype.capitalize()}, Purchase Price={Pur_Price})"
        )
    plt.show()


def options(ticker):
    stock = yf.Ticker(ticker)
    Today = date.today()
    all_options = []
    for expiration in stock.options:
        expiry_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        T = (expiry_date - Today).days / 365
        chain = stock.option_chain(expiration)

        calls_df = chain.calls.copy()
        calls_df["T"] = T
        calls_df["Expiration"] = expiration
        calls_df["otype"] = "call"
        all_options.append(calls_df)

        puts_df = chain.puts.copy()
        puts_df["T"] = T
        puts_df["Expiration"] = expiration
        puts_df["otype"] = "put"
        all_options.append(puts_df)

    options_df = pd.concat(all_options, ignore_index=False)
    return options_df


def skew_surface(ticker, otype="call"):
    options_df = options(ticker)
    stock_price = stock_data(ticker)
    filtered = options_df[options_df["otype"] == otype].copy()
    filtered = filtered.dropna(subset=["impliedVolatility"])
    K = jnp.asarray(filtered["strike"] / stock_price)
    T = jnp.asarray(filtered["T"])
    IV = jnp.asarray(filtered["impliedVolatility"])
    T_grid, K_grid = jnp.meshgrid(
        jnp.linspace(min(T), max(T), 50), jnp.linspace(min(K), max(K), 50)
    )
    IV_grid = griddata(points=(T, K), values=IV, xi=(T_grid, K_grid), method="linear")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        T_grid, K_grid, IV_grid, cmap="viridis", edgecolor="none", alpha=0.9
    )
    ax.set_xlabel("Time to expiry (Years)")
    ax.set_ylabel("Moneyness (K/S)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"{ticker.upper()} {otype.capitalize()} Volatility Skew Surface")
    ax.view_init(elev=15,azim=27,roll=0)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()
    return T, K

d=skew_surface("AAPL")