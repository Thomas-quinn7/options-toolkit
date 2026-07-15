from jax.scipy.stats import norm as jnorm
import yfinance as yf
import jax.numpy as jnp
from jax import grad
import seaborn as sn
from matplotlib import pyplot as plt
from datetime import datetime,date
import pandas as pd
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata

# S = Stock Price; K = Strike Price; T = Time Period; r = Risk-free Return;
# Sigma = Implied Volatility; q = Dividends Yield

def black_scholes(S,K,T,r,sigma,q=0,otype="call"):
    d1= (jnp.log(S/K)+(r-q+(sigma**2)/2)*T)/(sigma*jnp.sqrt(T))
    d2= d1 - sigma*jnp.sqrt(T)
    if otype == "call":
        call = jnorm.cdf(d1)*S*jnp.exp(-q*T) - jnorm.cdf(d2)*K*jnp.exp(-r*T)
        return call
    elif otype == "put":
        put = K*jnp.exp(-r*T)*jnorm.cdf(-d2)-S*jnorm.cdf(-d1)*jnp.exp(-q*T)
        return put
    else:
        raise ValueError("otype must be 'call' or 'put'")

def stock_data(stock):
    stock_data = yf.Ticker(stock).history(period="max")
    S=stock_data["Close"].iloc[-1]
    return S

def get_riskfree_rate():
    r_data = yf.Ticker("^IRX")
    r_df = r_data.history(period="5d")
    if r_df.empty:
        raise ValueError("No data returned for ^IRX. Check internet connection or ticker.")
    else:
        r = r_df['Close'].iloc[-1] / 100
        return r


def diff_function(S,K,T,r,sigma_est,price,q=0,otype="call"):
    theoretical = black_scholes(S,K,T,r,sigma_est,q)
    return theoretical - price

def implied_volatility(stock,K,sigma_est,price,T=1,q=0,otype="call",E=0.01,iter=40):
    S=stock_data(stock)
    r=get_riskfree_rate()
    iterations=0
    diff = diff_function(S, K, T, r, sigma_est, price, q, otype)
    loss_grad = grad(diff_function,argnums=4)
    while abs(diff) > E and iterations < iter:
        diff = diff_function(S,K,T,r,sigma_est,price,q,otype)
        diff_grad = loss_grad(S,K,T,r,sigma_est,price,q,otype)
        if diff_grad == 0:
            print("Gradient is zero or invalid; stopping.")
            break

        iterations += 1
        if abs(diff) < E:
            break
        sigma_est = sigma_est - diff / diff_grad
    return sigma_est

def greeks(S,K,T,r,sigma,q=0,otype="call"):
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    q = float(q)
    Delta_func = grad(black_scholes,argnums=0)
    Gamma_func = grad(Delta_func,argnums=0)
    Theta_func = grad(black_scholes,argnums=2)
    Vega_func = grad(black_scholes,argnums=4)
    Rho_func = grad(black_scholes,argnums=3)
    Delta = Delta_func(S,K,T,r,sigma,q,otype)
    Gamma = Gamma_func(S,K,T,r,sigma,q,otype)
    Theta = Theta_func(S,K,T,r,sigma,q,otype)
    Vega = Vega_func(S,K,T,r,sigma,q,otype)
    Rho = Rho_func(S,K,T,r,sigma,q,otype)
    return Delta, Gamma, Theta, Vega, Rho

def price_heatmap(S,K,T=1,r=0.03,sigma=0.1,q=0,otype="call",Pur_Price=0,grid=9,diff=0.2):
    interval = (2*diff) / (grid-1)
    step_p = interval*S
    step_vol = interval*sigma
    diff_prices = list()
    diff_vols = list()
    min_p= (1-diff)*S
    min_vol= (1-diff)*sigma
    iter=0
    while iter != grid:
        diff_prices.append(min_p+(iter)*step_p)
        diff_vols.append(min_vol+(iter)*step_vol)
        iter+=1
    plt.figure(figsize = (8,6), dpi = 200)
    price_matrix = []
    for vol in diff_vols:
        row = []
        for price in diff_prices:
            price = float(black_scholes(price,K,T,r,vol,q,otype)) - Pur_Price
            row.append(price)
        price_matrix.append(row)
    price_matrix = jnp.array(price_matrix)
    sn.heatmap(price_matrix, xticklabels=[round(float(p), 2) for p in diff_prices],
                yticklabels=[round(float(v), 3) for v in diff_vols],
                cmap='RdYlGn', annot=True, fmt=".2f")
    plt.xlabel("Stock Price")
    plt.ylabel("Implied Volatility (σ)")
    if Pur_Price == 0:
        plt.title(f"Option Price Heatmap (K={K}, T={T}, {otype.capitalize()})")
    else:
        plt.title(f"Option Profit Heatmap (K={K}, T={T}, {otype.capitalize()}, Purchase Price={Pur_Price})")
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
        calls_df['T'] = T
        calls_df['Expiration'] = expiration
        calls_df['otype'] = "call"
        all_options.append(calls_df)

        puts_df = chain.puts.copy()
        puts_df['T'] = T
        puts_df['Expiration'] = expiration
        puts_df['otype'] = "put"
        all_options.append(puts_df)

    options_df = pd.concat(all_options, ignore_index=False)
    return options_df

def skew_surface(ticker,otype="call"):
    options_df = options(ticker)
    stock_price = stock_data(ticker)
    filtered = options_df[options_df['otype'] == otype].copy()
    filtered = filtered.dropna(subset=['impliedVolatility'])
    K = filtered['strike'].values/stock_price
    T = filtered['T'].values
    IV = filtered['impliedVolatility'].values
    T_grid, K_grid = jnp.meshgrid(
        jnp.linspace(min(T), max(T), 50),
        jnp.linspace(min(K), max(K), 50)
    )
    IV_grid = griddata(
        points=(T, K),
        values=IV,
        xi=(T_grid, K_grid),
        method='linear'
    )
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        T_grid, K_grid, IV_grid,
        cmap='viridis', edgecolor='none', alpha=0.9
    )
    ax.set_xlabel("Time to expiry (Years)")
    ax.set_ylabel("Moneyness (K/S)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"{ticker.upper()} {otype.capitalize()} Volatility Skew Surface")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()
    return T, K

d=skew_surface("AAPL")