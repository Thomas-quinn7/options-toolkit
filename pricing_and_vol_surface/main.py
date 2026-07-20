"""Smoke driver for black.py.

Exercises Black-Scholes pricing, the full Greeks, and an implied-volatility
round-trip on known inputs (no network calls). Run from this folder:

    python main.py
"""

import jax.numpy as jnp

from black import (
    black_scholes,
    greeks,
    diff_function,
    implied_volatility,
)


def demo_pricing():
    print("=== Black-Scholes pricing ===")
    # S=spot, K=strike, T=years, r=rate, sigma=vol
    S, K, T, r, sigma = 100.0, 110.0, 0.25, 0.05, 0.2

    call = black_scholes(S, K, T, r, sigma, otype="call")
    put = black_scholes(S, K, T, r, sigma, otype="put")
    print(f"Call price: {call:.4f}")
    print(f"Put price:  {put:.4f}")

    # Put-call parity (q=0): C - P == S - K*e^(-rT)
    parity_lhs = float(call - put)
    parity_rhs = float(S - K * jnp.exp(-r * T))
    print(f"C - P:              {parity_lhs:.6f}")
    print(f"S - K*e^(-rT):      {parity_rhs:.6f}")
    print(f"Put-call parity err: {abs(parity_lhs - parity_rhs):.2e}\n")


def demo_greeks():
    print("=== Greeks ===")
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    delta, gamma, theta, vega, rho = greeks(S, K, T, r, sigma, otype="call")
    print(f"Call:  delta={float(delta):+.4f}  gamma={float(gamma):.4f}  "
          f"theta={float(theta):+.4f}  vega={float(vega):.4f}  rho={float(rho):+.4f}")

    delta_put, _, _, _, _ = greeks(S, K, T, r, sigma, otype="put")
    # For q=0: delta_call - delta_put == 1
    print(f"Put delta: {float(delta_put):+.4f}")
    print(f"delta_call - delta_put (should be ~1): {float(delta - delta_put):.4f}\n")


def demo_implied_vol_roundtrip():
    print("=== Implied-volatility round-trip ===")
    S, K, T, r, true_sigma = 100.0, 105.0, 0.25, 0.05, 0.2

    # Price at a known vol, then recover that vol from the price.
    price = black_scholes(S, K, T, r, true_sigma, otype="call")
    recovered = implied_volatility(
        S, K, sigma_est=0.5, price=price, r=r, T=T, otype="call"
    )
    residual = diff_function(S, K, T, r, recovered, price, 0, "call")
    print(f"True sigma:      {true_sigma:.4f}")
    print(f"Recovered sigma: {float(recovered):.4f}")
    print(f"Pricing residual at recovered vol: {float(residual):.2e}\n")


def main():
    print("Black-Scholes / Greeks smoke driver\n")
    demo_pricing()
    demo_greeks()
    demo_implied_vol_roundtrip()
    print("Done.")


if __name__ == "__main__":
    main()
