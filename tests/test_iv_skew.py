"""Offline tests for the IV-skew scanner's own-IV pipeline.

The scanner now inverts implied vols from bid-ask mid prices with the repo's
Brent inverter instead of trusting yfinance's ``impliedVolatility`` field.
These tests build synthetic chains whose PRICES come from a known smile while
the vendor IV column is filled with garbage - the pipeline must recover the
true smile from the prices alone. No network access anywhere.

Run:  python -m pytest tests/test_iv_skew.py -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skew_bubble_indicator"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing_and_vol_surface"))

import IV_skew as X  # noqa: E402
from vol_surface import bs_call  # noqa: E402

S, T, R = 100.0, 30.0 / 365.0, 0.03


def true_smile(K: float) -> float:
    """A put-skewed smile: vol rises as strikes fall below spot."""
    return 0.20 + 0.8 * max(0.0, (0.95 * S - K) / S)


def make_chain(otype: str, strikes=None) -> pd.DataFrame:
    """A synthetic chain priced off the true smile; vendor IV is GARBAGE.

    Half-spreads are proportional to price (~6% total spread) as in a real
    chain - a fixed absolute spread would trip the spread-percentage gate on
    every cheap OTM quote and silently empty the wings.
    """
    if strikes is None:
        strikes = np.arange(80.0, 121.0, 2.5)
    rows = []
    for K in strikes:
        sigma = true_smile(K)
        call = bs_call(S, K, T, R, sigma)
        price = call if otype == "call" else call - S + K * np.exp(-R * T)
        half = 0.03 * price
        rows.append({
            "strike": K,
            "bid": price - half,
            "ask": price + half,
            "volume": 500,
            "openInterest": 1000,
            "lastPrice": price,
            "impliedVolatility": 9.99,   # vendor value is nonsense on purpose
        })
    return pd.DataFrame(rows)


def test_add_own_iv_recovers_true_smile_from_prices():
    for otype in ("put", "call"):
        df = X.add_own_iv(make_chain(otype), S, T, R, otype)
        assert (df["yf_iv"] == 9.99).all()   # vendor column preserved as diagnostic
        err = np.abs(df["impliedVolatility"].values
                     - np.array([true_smile(k) for k in df["strike"]]))
        assert np.nanmax(err) < 0.01         # recovered from prices, not vendor IV


def test_no_arb_violating_mid_inverts_to_nan():
    df = make_chain("call", strikes=np.array([90.0]))
    df.loc[0, "bid"] = 5.0                   # deep below intrinsic (S-K=10)
    df.loc[0, "ask"] = 5.2
    out = X.add_own_iv(df, S, T, R, "call")
    assert np.isnan(out["impliedVolatility"].iloc[0])


def test_missing_two_sided_market_inverts_to_nan():
    df = make_chain("call", strikes=np.array([100.0]))
    df.loc[0, "bid"] = 0.0                   # no bid -> no usable mid
    out = X.add_own_iv(df, S, T, R, "call")
    assert np.isnan(out["impliedVolatility"].iloc[0])


def test_validation_pipeline_runs_on_own_iv():
    """validate_option_data must keep good rows (own IV in the sane band)
    even though the vendor IV column is out of range, and drop rows whose
    price cannot be inverted."""
    df = make_chain("put")
    # A put quoted ABOVE its no-arbitrage upper bound (K e^-rT ~ 89.8): the
    # tight spread passes the spread gate, but the mid cannot be inverted.
    bad = pd.DataFrame([{
        "strike": 90.0, "bid": 92.0, "ask": 93.0,
        "volume": 500, "openInterest": 1000, "lastPrice": 92.5,
        "impliedVolatility": 9.99,
    }])
    bad_iv = X.add_own_iv(bad, S, T, R, "put")["impliedVolatility"].iloc[0]
    assert not np.isfinite(bad_iv)
    out = X.validate_option_data(pd.concat([df, bad], ignore_index=True),
                                 S, T, R, "put")
    # good rows survive on OWN IV despite garbage vendor IV...
    assert len(out) >= len(df) - 1
    assert out["impliedVolatility"].between(X.MIN_IV, X.MAX_IV).all()
    # ...and the uninvertible quote is gone
    assert not (out["bid"] >= 92.0).any()


def test_skew_metric_reads_the_price_implied_skew():
    """End-to-end without network: put IV at the OTM put strike must exceed
    call IV at the OTM call strike, matching the smile the PRICES encode."""
    puts = X.validate_option_data(make_chain("put"), S, T, R, "put")
    calls = X.validate_option_data(make_chain("call"), S, T, R, "call")
    put_iv = X.find_mean_iv_at_strike(puts, 0.90 * S)
    call_iv = X.find_mean_iv_at_strike(calls, 1.10 * S)
    assert np.isfinite(put_iv) and np.isfinite(call_iv)
    assert put_iv - call_iv > 0.02           # the built-in put skew
    assert abs(put_iv - true_smile(0.90 * S)) < 0.02
