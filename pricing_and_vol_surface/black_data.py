"""Live-input and plotting helpers for the pure JAX pricer (black.py).

Everything network-facing or GUI-facing that used to live in black.py lives
here, so importing the pricer itself pulls in nothing beyond JAX. Importing
THIS module also has no side effects; the live demos run only under
``__main__``.
"""

import yfinance as yf
from datetime import datetime, date

import jax.numpy as jnp
from jax import jit, vmap
import seaborn as sn
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d  # noqa: F401  (registers the 3d projection)
from scipy.interpolate import griddata

from pricing_and_vol_surface.black import black_scholes


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


if __name__ == "__main__":
    # Demonstration only - fires a live download and a matplotlib window.
    # Kept behind the guard so importing this module has no side effects.
    skew_surface("AAPL")
