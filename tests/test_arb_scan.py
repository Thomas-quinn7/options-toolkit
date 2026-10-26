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
import pytz

from arbitrage.arb_scan import (
    ArbitrageDetector,
    Option,
    quote_is_tradeable,
    within_moneyness,
)


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


def test_time_to_expiry_is_anchored_to_the_16_et_close():
    """ET, not the local wall clock: on a European box the naive comparison
    was off by 5-6 hours - material at 0-3 DTE. The result must equal the
    true America/New_York distance to within a second."""
    d = detector()
    expiry = datetime.now() + timedelta(days=2)
    et = pytz.timezone("America/New_York")
    expect = (et.localize(expiry.replace(hour=16, minute=0, second=0,
                                         microsecond=0))
              - datetime.now(et)).total_seconds() / (365.0 * 24 * 3600)
    assert abs(d.time_to_expiry(expiry) - expect) < 1.0 / (365.0 * 24 * 3600)


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
    assert len(result) == 1
    assert result[0]["estimated_profit"] == pytest.approx(0.50)


# --------------------------------------------------------------------------- #
# Liquidity gate: stale marks on dead strikes are the false-positive source    #
# --------------------------------------------------------------------------- #
def test_liquidity_gate_blocks_dead_strikes():
    """A stale mark on a dead strike (nothing traded all session, no open
    interest) is the scanner's historical false-positive fire hose -
    measured 2026-10-26: 23,836 'opportunities' with $80+ phantom box
    profits on AAPL+MSFT before this gate. Two-sided quotes with any real
    life behind them still pass."""
    assert quote_is_tradeable(1.0, 1.1, volume=0, open_interest=500)
    assert quote_is_tradeable(1.0, 1.1, volume=3, open_interest=0)
    assert not quote_is_tradeable(1.0, 1.1, volume=0, open_interest=0)
    assert not quote_is_tradeable(1.0, 1.1, volume=0, open_interest=50)
    assert not quote_is_tradeable(0.0, 1.1, volume=5, open_interest=500)   # one-sided
    assert not quote_is_tradeable(1.2, 1.1, volume=5, open_interest=500)   # crossed


def test_moneyness_window_drops_the_wings():
    """The wings are where stale marks live even past the liquidity gate
    (open interest remembers quotes that stopped trading weeks ago) -
    measured on MSFT: 0 flags within +-10% of spot, 15,462 on the full
    chain. The scan stops at the window edge."""
    assert within_moneyness(100.0, 100.0)
    assert within_moneyness(114.0, 100.0)
    assert within_moneyness(86.0, 100.0)
    assert not within_moneyness(116.0, 100.0)
    assert not within_moneyness(84.0, 100.0)
    assert within_moneyness(129.0, 100.0, max_moneyness=0.30)   # override widens


def test_default_min_edge_is_a_nickel():
    """On delayed quotes a penny edge is stale-mark noise: the default floor
    is a nickel (an explicit min_edge can still hunt smaller)."""
    assert ArbitrageDetector(spot_price=100.0).min_edge == 0.05


def test_subnickel_parity_edge_not_flagged_by_default():
    d = detector()
    call = opt(100, "call", 1.90, 2.00)
    put = opt(100, "put", 2.01, 2.11)          # reversal edge = 0.01
    assert d.check_put_call_parity(call, put) is None
    tight = ArbitrageDetector(spot_price=100.0, risk_free_rate=0.05,
                              min_edge=0.005)
    assert tight.check_put_call_parity(call, put) is not None


# --------------------------------------------------------------------------- #
# Vertical / retail box / calendar checks (previously untested)               #
# --------------------------------------------------------------------------- #
def test_vertical_call_monotonicity_flagged():
    """A lower-strike call whose ASK is under the higher strike's BID is a
    free vertical - buy the cheap rich leg, sell the expensive poor one."""
    d = detector()
    found = d.check_vertical_spread_arbitrage([
        opt(95, "call", 5.00, 5.10),
        opt(100, "call", 5.30, 5.40),
    ])
    assert len(found) == 1
    assert "Call" in found[0]["type"]
    assert found[0]["estimated_profit"] == pytest.approx(5.30 - 5.10)


def test_vertical_put_monotonicity_flagged():
    d = detector()
    found = d.check_vertical_spread_arbitrage([
        opt(95, "put", 5.30, 5.40),            # lower strike, bid rich
        opt(100, "put", 5.00, 5.10),           # higher strike, ask cheap
    ])
    assert len(found) == 1
    assert "Put" in found[0]["type"]
    assert found[0]["estimated_profit"] == pytest.approx(5.30 - 5.10)


def test_vertical_monotone_chain_not_flagged():
    d = detector()
    assert d.check_vertical_spread_arbitrage([
        opt(95, "call", 5.90, 6.10),
        opt(100, "call", 2.45, 2.65),
        opt(105, "call", 0.70, 0.90),
    ]) == []


def test_retail_box_buy_side_underpriced_flagged():
    """The retail box only flags the BUY side (no shorting): a box bought
    below its discounted width locks in value at expiry."""
    d = detector()
    result = d.check_box_spread_retail(
        opt(95, "call", 5.00, 5.10), opt(95, "put", 0.50, 0.60),
        opt(105, "call", 0.80, 0.90), opt(105, "put", 5.00, 5.10),
    )
    assert result is not None
    assert "Retail Buy" in result["type"]
    T = d.time_to_expiry(EXPIRY)
    assert result["estimated_profit"] == pytest.approx(
        10 * np.exp(-0.05 * T) - 8.90)          # buy cost (5.10-0.80)+(5.10-0.50)


def test_institutional_calendar_monotonicity_flagged():
    """A longer-dated option is worth at least as much as a shorter one at
    the same strike; near bid above far ask is a locked-in credit (the check
    is model-free even for American options)."""
    d = detector()
    near = opt(100, "call", 3.20, 3.30, expiry=EXPIRY)
    far = opt(100, "call", 2.90, 3.00, expiry=FAR_EXPIRY)
    result = d.check_calendar_spread(near, far)
    assert result is not None
    assert result["estimated_profit"] == pytest.approx(3.20 - 3.00)
    assert d.check_calendar_spread(far, near) is None          # order matters
    assert d.check_calendar_spread(
        opt(100, "call", 3.00, 3.10, expiry=EXPIRY), far) is None


def test_retail_calendar_flagged():
    """Retail variant: buy the far-dated below the near-dated bid - a
    long-biased edge a no-short account can still enter."""
    d = detector()
    near = opt(100, "put", 3.20, 3.30, expiry=EXPIRY)
    far = opt(100, "put", 2.80, 2.90, expiry=FAR_EXPIRY)
    result = d.check_calendar_spread_retail(near, far)
    assert result is not None
    assert result["estimated_profit"] == pytest.approx(3.20 - 2.90)
    borderline = opt(100, "put", 3.16, 3.20, expiry=FAR_EXPIRY)
    assert d.check_calendar_spread_retail(
        near, borderline) is None                # inside the 5c retail tolerance
