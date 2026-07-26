"""Tests for the static no-arbitrage scanner (synthetic chains, no network).

The load-bearing tests are the executable-price parity ones: quotes whose MIDS
violate European parity but whose bid/ask-crossed prices sit inside the
American no-arbitrage band must NOT be flagged (the old mid-price false
positive), while genuine breaches of either band edge must be flagged with the
profit computed at the prices you would actually trade.

Run:  python -m pytest tests/test_arb_scan.py -q
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from arbitrage.arb_scan import ArbitrageDetector, Option


EXPIRY = datetime.now() + timedelta(days=30)
FAR_EXPIRY = datetime.now() + timedelta(days=60)


def opt(strike, otype, bid, ask, expiry=EXPIRY):
    return Option(strike=strike, expiry=expiry, type=otype, bid=bid, ask=ask)


def detector(q=0.0, r=0.05):
    return ArbitrageDetector(spot_price=100.0, risk_free_rate=r, div_yield=q)


# --------------------------------------------------------------------------- #
# Put-call parity: American band on executable prices                         #
# --------------------------------------------------------------------------- #
def test_parity_mid_violation_inside_band_not_flagged():
    # Mids: C-P = 0.20 vs European theoretical ~0.42 -> the old mid-price
    # check flagged this. Executable prices sit inside [lower, upper]:
    # C_bid - P_ask = 0.10, C_ask - P_bid = 0.30.
    d = detector()
    call = opt(100, "call", 2.50, 2.60)
    put = opt(100, "put", 2.30, 2.40)
    assert d.check_put_call_parity(call, put) is None


def test_parity_conversion_breach_flagged_at_executable_prices():
    d = detector()
    call = opt(100, "call", 3.50, 3.60)
    put = opt(100, "put", 2.90, 3.00)
    result = d.check_put_call_parity(call, put)
    assert result is not None
    assert "Conversion" in result["type"]
    T = d.time_to_expiry(EXPIRY)
    upper = 100.0 - 100.0 * np.exp(-0.05 * T)
    # Profit is sell-call-at-bid / buy-put-at-ask edge beyond the upper bound.
    assert result["estimated_profit"] == pytest.approx((3.50 - 3.00) - upper)


def test_parity_reversal_breach_flagged_without_dividend():
    d = detector(q=0.0)
    call = opt(100, "call", 1.90, 2.00)
    put = opt(100, "put", 2.10, 2.20)
    result = d.check_put_call_parity(call, put)
    assert result is not None
    assert "Reversal" in result["type"]
    # lower bound is 0 at q=0; executable diff C_ask - P_bid = -0.10.
    assert result["estimated_profit"] == pytest.approx(0.10)


def test_parity_reversal_gap_explained_by_dividend_not_flagged():
    # Same quotes as above, but a 2% trailing dividend yield lowers the
    # American lower bound below the executable diff -> no arbitrage.
    d = detector(q=0.02)
    call = opt(100, "call", 1.90, 2.00)
    put = opt(100, "put", 2.10, 2.20)
    assert d.check_put_call_parity(call, put) is None


def test_parity_requires_matching_strike_and_expiry():
    d = detector()
    assert d.check_put_call_parity(opt(100, "call", 5, 5.1),
                                   opt(105, "put", 5, 5.1)) is None
    assert d.check_put_call_parity(opt(100, "call", 5, 5.1),
                                   opt(100, "put", 5, 5.1, expiry=FAR_EXPIRY)) is None


# --------------------------------------------------------------------------- #
# Time to expiry: fractional days                                             #
# --------------------------------------------------------------------------- #
def test_time_to_expiry_fractional_and_monotonic():
    d = detector()
    t2 = d.time_to_expiry(datetime.now() + timedelta(days=2))
    t3 = d.time_to_expiry(datetime.now() + timedelta(days=3))
    assert 0 < t2 < t3
    # Two calendar days out, expiring at the 16:00 close: between ~1.5 and
    # ~2.7 days of year-time regardless of the wall-clock hour the test runs.
    assert 1.5 / 365 < t2 < 2.7 / 365


def test_time_to_expiry_never_zero():
    d = detector()
    assert d.time_to_expiry(datetime.now() - timedelta(days=1)) > 0


# --------------------------------------------------------------------------- #
# Box spread: each direction priced at its own executable side                #
# --------------------------------------------------------------------------- #
def test_box_sell_side_not_flagged_from_buy_side_prices():
    # Wide spreads: buy-box cost 10.36 (rich), sell-box proceeds 9.56 (cheap),
    # theoretical ~9.96 sits between them -> no executable arbitrage. The old
    # check compared theoretical to buy-box cost for BOTH directions and
    # flagged a phantom sell-box here.
    d = detector()
    result = d.check_box_spread(
        opt(95, "call", 5.90, 6.10), opt(95, "put", 0.50, 0.70),
        opt(105, "call", 0.70, 0.90), opt(105, "put", 5.26, 5.46),
    )
    assert result is None


def test_box_buy_side_breach_flagged():
    d = detector()
    result = d.check_box_spread(
        opt(95, "call", 5.00, 5.10), opt(95, "put", 0.50, 0.60),
        opt(105, "call", 0.80, 0.90), opt(105, "put", 5.00, 5.10),
    )
    # Buy-box cost = (5.10 - 0.80) + (5.10 - 0.50) = 8.90 < ~9.96 theoretical.
    assert result is not None
    assert "Buy Box" in result["strategy"]
    T = d.time_to_expiry(EXPIRY)
    assert result["estimated_profit"] == pytest.approx(
        10 * np.exp(-0.05 * T) - 8.90)


def test_box_sell_side_breach_flagged_at_bid():
    d = detector()
    result = d.check_box_spread(
        opt(95, "call", 7.00, 7.10), opt(95, "put", 0.50, 0.60),
        opt(105, "call", 0.80, 0.90), opt(105, "put", 5.00, 5.10),
    )
    # Sell-box proceeds = (7.00 - 0.90) + (5.00 - 0.60) = 10.50 > theoretical.
    assert result is not None
    assert "Sell Box" in result["strategy"]
    assert result["market_proceeds"] == pytest.approx(10.50)


# --------------------------------------------------------------------------- #
# Full-chain scan: a consistent synthetic chain must be silent                #
# --------------------------------------------------------------------------- #
def _consistent_chain():
    return [
        opt(95, "call", 5.90, 6.10), opt(95, "put", 0.50, 0.70),
        opt(100, "call", 2.45, 2.65), opt(100, "put", 1.93, 2.13),
        opt(105, "call", 0.70, 0.90), opt(105, "put", 5.26, 5.46),
        opt(100, "call", 3.60, 3.70, expiry=FAR_EXPIRY),
        opt(100, "put", 2.90, 3.00, expiry=FAR_EXPIRY),
    ]


def test_consistent_chain_no_opportunities_institutional():
    d = detector()
    assert d.find_all_arbitrage(_consistent_chain(), allow_short=True) == []


def test_consistent_chain_no_opportunities_retail():
    d = detector()
    assert d.find_all_arbitrage(_consistent_chain(), allow_short=False) == []


def test_negative_cost_butterfly_still_flagged():
    d = detector()
    result = d.check_butterfly_arbitrage([
        opt(95, "call", 6.90, 7.00),
        opt(100, "call", 5.00, 5.10),
        opt(105, "call", 2.40, 2.50),
    ])
    # Cost = 7.00 - 2*5.00 + 2.50 = -0.50.
    assert result is not None
    assert result["estimated_profit"] == pytest.approx(0.50)
