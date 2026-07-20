"""Static no-arbitrage scanner for listed options (yfinance data).

One scanner, two modes controlled by `allow_short` (CLI: `--retail`):

  allow_short=True  (default)  - institutional relationships that assume you can
                                 short the underlying / sell options freely:
                                 put-call parity, both-direction box spreads,
                                 negative-cost butterflies, calendar monotonicity.

  allow_short=False (--retail) - checks a retail account with NO short selling
                                 can still act on: vertical monotonicity,
                                 buy-side underpriced boxes, negative-cost
                                 butterflies, and calendar mispricings you can
                                 enter long.

Caveat: these are European parity/box relationships applied to American
yfinance options with no dividend term, so flagged trades can be spurious.
See README "Known limitations".
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import yfinance as yf


@dataclass
class Option:
    """Represents an option contract."""
    strike: float
    expiry: datetime
    type: str  # 'call' or 'put'
    bid: float
    ask: float

    @property
    def mid(self):
        return (self.bid + self.ask) / 2


class OptionsDataFetcher:
    """Fetches options data from Yahoo Finance."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = None
        self.spot_price = None

    def fetch_options(self) -> "tuple[Optional[float], List[Option]]":
        """Fetch all available options for the ticker.

        Returns:
            (spot_price, list of Option objects). On failure returns (None, []).
        """
        try:
            self.stock = yf.Ticker(self.ticker)

            hist = self.stock.history(period="1d")
            if hist.empty:
                raise ValueError(f"Could not fetch price data for {self.ticker}")

            self.spot_price = hist["Close"].iloc[-1]

            expirations = self.stock.options
            if not expirations:
                raise ValueError(f"No options data available for {self.ticker}")

            print(f"\n{self.ticker} - Current Price: ${self.spot_price:.2f}")
            print(f"Available expirations: {len(expirations)}")

            all_options = []

            for expiry_str in expirations:
                try:
                    opt_chain = self.stock.option_chain(expiry_str)
                    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")

                    for _, row in opt_chain.calls.iterrows():
                        if row["bid"] > 0 and row["ask"] > 0 and row["ask"] > row["bid"]:
                            all_options.append(Option(
                                strike=float(row["strike"]),
                                expiry=expiry_date,
                                type="call",
                                bid=float(row["bid"]),
                                ask=float(row["ask"]),
                            ))

                    for _, row in opt_chain.puts.iterrows():
                        if row["bid"] > 0 and row["ask"] > 0 and row["ask"] > row["bid"]:
                            all_options.append(Option(
                                strike=float(row["strike"]),
                                expiry=expiry_date,
                                type="put",
                                bid=float(row["bid"]),
                                ask=float(row["ask"]),
                            ))

                except (KeyError, ValueError) as e:
                    print(f"  Warning: Could not parse options for {expiry_str}: {e}")
                    continue

            print(f"Successfully fetched {len(all_options)} option contracts")
            return self.spot_price, all_options

        except (ValueError, KeyError, ConnectionError) as e:
            print(f"Error fetching options for {self.ticker}: {e}")
            return None, []


class ArbitrageDetector:
    """Static no-arbitrage checks over a set of Option objects.

    Individual check_* methods hold the no-arb logic; find_all_arbitrage()
    wires them together and picks the institutional vs retail set via
    `allow_short`.
    """

    def __init__(self, spot_price: float, risk_free_rate: float = 0.05):
        self.spot = spot_price
        self.r = risk_free_rate
        self.opportunities = []

    def time_to_expiry(self, expiry: datetime) -> float:
        """Time to expiry in years (floored to avoid division by zero)."""
        days = (expiry - datetime.now()).days
        return max(days / 365.0, 0.001)

    # ---------- institutional checks (assume short selling) ----------

    def check_put_call_parity(self, call: Option, put: Option) -> Optional[Dict]:
        """Put-call parity: C - P = S - K*e^(-rT). Violations imply arbitrage.

        Requires the ability to short the underlying, so institutional only.
        """
        if call.strike != put.strike or call.expiry != put.expiry:
            return None

        K = call.strike
        T = self.time_to_expiry(call.expiry)

        pv_strike = K * np.exp(-self.r * T)
        theoretical_diff = self.spot - pv_strike
        actual_diff = call.mid - put.mid

        tolerance = 0.01
        discrepancy = abs(actual_diff - theoretical_diff)

        if discrepancy > tolerance:
            if actual_diff > theoretical_diff:
                strategy = "Sell Call, Buy Put, Buy Stock, Borrow PV(K)"
                profit = actual_diff - theoretical_diff
            else:
                strategy = "Buy Call, Sell Put, Short Stock, Lend PV(K)"
                profit = theoretical_diff - actual_diff

            return {
                "type": "Put-Call Parity Violation",
                "strike": K,
                "expiry": call.expiry,
                "theoretical_diff": theoretical_diff,
                "actual_diff": actual_diff,
                "discrepancy": discrepancy,
                "strategy": strategy,
                "estimated_profit": profit,
            }

        return None

    def check_box_spread(self, call_low: Option, put_low: Option,
                         call_high: Option, put_high: Option) -> Optional[Dict]:
        """Box spread: theoretical value = (K2 - K1)*e^(-rT).

        Flags both buy-box (cheap) and sell-box (rich) mispricings.
        """
        if not (call_low.expiry == put_low.expiry == call_high.expiry == put_high.expiry):
            return None

        K1 = min(call_low.strike, call_high.strike)
        K2 = max(call_low.strike, call_high.strike)
        if K1 >= K2:
            return None

        T = self.time_to_expiry(call_low.expiry)
        theoretical_value = (K2 - K1) * np.exp(-self.r * T)
        market_cost = (call_low.ask - call_high.bid) + (put_high.ask - put_low.bid)

        tolerance = 0.01
        discrepancy = abs(market_cost - theoretical_value)

        if discrepancy > tolerance:
            if market_cost < theoretical_value:
                return {
                    "type": "Box Spread Arbitrage",
                    "strikes": f"{K1}/{K2}",
                    "expiry": call_low.expiry,
                    "theoretical_value": theoretical_value,
                    "market_cost": market_cost,
                    "strategy": "Buy Box (Long Call Spread + Long Put Spread)",
                    "estimated_profit": theoretical_value - market_cost,
                }
            return {
                "type": "Box Spread Arbitrage",
                "strikes": f"{K1}/{K2}",
                "expiry": call_low.expiry,
                "theoretical_value": theoretical_value,
                "market_cost": market_cost,
                "strategy": "Sell Box (Short Call Spread + Short Put Spread)",
                "estimated_profit": market_cost - theoretical_value,
            }

        return None

    def check_butterfly_arbitrage(self, options: List[Option],
                                  tolerance: float = 0.01,
                                  spacing_tol: float = 0.01) -> Optional[Dict]:
        """Butterfly convexity: buy 1 low, sell 2 mid, buy 1 high must cost >= 0.

        A negative entry cost is a free-money arbitrage. Tolerances default to the
        strict (institutional) values; the retail path passes looser ones.
        """
        options = sorted(options, key=lambda x: x.strike)
        if len(options) < 3:
            return None

        for i in range(len(options) - 2):
            low, mid, high = options[i], options[i + 1], options[i + 2]

            if low.expiry != mid.expiry or mid.expiry != high.expiry:
                continue
            if low.type != mid.type or mid.type != high.type:
                continue

            wing1 = mid.strike - low.strike
            wing2 = high.strike - mid.strike
            if abs(wing1 - wing2) > spacing_tol:
                continue

            cost = low.ask - 2 * mid.bid + high.ask

            if cost < -tolerance:
                return {
                    "type": "Butterfly Arbitrage (Negative Cost)",
                    "strikes": f"{low.strike}/{mid.strike}/{high.strike}",
                    "option_type": low.type,
                    "expiry": low.expiry,
                    "strategy": f"Buy ${low.strike}, Sell 2x ${mid.strike}, Buy ${high.strike}",
                    "entry_cost": cost,
                    "estimated_profit": -cost,
                    "max_additional_profit": min(wing1, wing2),
                }

        return None

    def check_calendar_spread(self, near: Option, far: Option,
                              tolerance: float = 0.01) -> Optional[Dict]:
        """Calendar monotonicity: a longer-dated option is worth at least as much
        as a shorter-dated one at the same strike/type. If the near bid exceeds
        the far ask you can sell near / buy far for a locked-in credit.
        """
        if near.strike != far.strike or near.type != far.type:
            return None
        if near.expiry >= far.expiry:
            return None

        if near.bid > far.ask + tolerance:
            return {
                "type": "Calendar Spread Arbitrage",
                "strike": near.strike,
                "option_type": near.type,
                "near_expiry": near.expiry,
                "far_expiry": far.expiry,
                "strategy": f"Sell near-dated @ ${near.bid:.2f}, Buy far-dated @ ${far.ask:.2f}",
                "estimated_profit": near.bid - far.ask,
            }

        return None

    # ---------- retail checks (no short selling) ----------

    def check_vertical_spread_arbitrage(self, options: List[Option]) -> Optional[Dict]:
        """Vertical monotonicity (no shorting needed).

        Calls: a lower strike must be worth more than a higher strike.
        Puts:  a higher strike must be worth more than a lower strike.
        If the more valuable option can be BOUGHT below the other's bid, arbitrage.
        """
        options = sorted(options, key=lambda x: x.strike)
        if len(options) < 2:
            return None

        for i in range(len(options) - 1):
            opt1, opt2 = options[i], options[i + 1]

            if opt1.expiry != opt2.expiry or opt1.type != opt2.type:
                continue

            if opt1.type == "call":
                if opt1.ask < opt2.bid - 0.02:
                    return {
                        "type": "Vertical Spread Arbitrage (Call)",
                        "strikes": f"{opt1.strike}/{opt2.strike}",
                        "expiry": opt1.expiry,
                        "strategy": f"Buy ${opt1.strike} call @ ${opt1.ask:.2f}, "
                                    f"Sell ${opt2.strike} call @ ${opt2.bid:.2f}",
                        "estimated_profit": opt2.bid - opt1.ask,
                        "capital_required": opt1.ask,
                        "risk": "Limited to premium paid",
                    }
            elif opt1.type == "put":
                if opt2.ask < opt1.bid - 0.02:
                    return {
                        "type": "Vertical Spread Arbitrage (Put)",
                        "strikes": f"{opt1.strike}/{opt2.strike}",
                        "expiry": opt1.expiry,
                        "strategy": f"Buy ${opt2.strike} put @ ${opt2.ask:.2f}, "
                                    f"Sell ${opt1.strike} put @ ${opt1.bid:.2f}",
                        "estimated_profit": opt1.bid - opt2.ask,
                        "capital_required": opt2.ask,
                        "risk": "Limited to premium paid",
                    }

        return None

    def check_box_spread_retail(self, call_low: Option, put_low: Option,
                                call_high: Option, put_high: Option) -> Optional[Dict]:
        """Retail box: only flag when the box can be BOUGHT below its theoretical
        value (K2-K1)*e^(-rT). No short selling required.
        """
        if not (call_low.expiry == put_low.expiry == call_high.expiry == put_high.expiry):
            return None

        K1 = min(call_low.strike, call_high.strike)
        K2 = max(call_low.strike, call_high.strike)
        if K1 >= K2 or (K2 - K1) < 1:
            return None

        T = self.time_to_expiry(call_low.expiry)
        theoretical_value = (K2 - K1) * np.exp(-self.r * T)
        buy_box_cost = (call_low.ask - call_high.bid) + (put_high.ask - put_low.bid)

        tolerance = 0.05  # retail: wider for spreads/commissions
        if buy_box_cost < theoretical_value - tolerance:
            profit = theoretical_value - buy_box_cost
            roi = (profit / buy_box_cost) * 100 if buy_box_cost > 0 else 0
            return {
                "type": "Box Spread Arbitrage (Retail Buy)",
                "strikes": f"{K1}/{K2}",
                "expiry": call_low.expiry,
                "theoretical_value": theoretical_value,
                "market_cost": buy_box_cost,
                "strategy": f"Buy Box: Buy ${K1} call, Sell ${K2} call, "
                            f"Buy ${K2} put, Sell ${K1} put",
                "estimated_profit": profit,
                "roi_percent": roi,
                "capital_required": buy_box_cost,
                "risk": "Nearly risk-free (box locks in value at expiry)",
            }

        return None

    def check_calendar_spread_retail(self, near: Option, far: Option) -> Optional[Dict]:
        """Retail calendar: if the far-dated option (more time value) can be bought
        below the near-dated bid, buy far / sell near for a long-biased edge.
        """
        if near.strike != far.strike or near.type != far.type:
            return None
        if near.expiry >= far.expiry:
            return None

        if far.ask < near.bid - 0.05:
            return {
                "type": "Calendar Spread Arbitrage (Retail)",
                "strike": near.strike,
                "option_type": near.type,
                "near_expiry": near.expiry,
                "far_expiry": far.expiry,
                "strategy": f"Buy far-dated @ ${far.ask:.2f}, Sell near-dated @ ${near.bid:.2f}",
                "estimated_profit": near.bid - far.ask,
                "capital_required": far.ask,
                "risk": "Limited - you own the longer-dated option",
            }

        return None

    # ---------- driver ----------

    def find_all_arbitrage(self, options: List[Option],
                           allow_short: bool = True) -> List[Dict]:
        """Scan all options for arbitrage.

        allow_short=True  -> institutional set (parity, box both ways, butterfly,
                             calendar monotonicity).
        allow_short=False -> retail set (vertical monotonicity, buy-side box,
                             negative-cost butterfly, retail calendar).
        """
        self.opportunities = []

        by_expiry: Dict[datetime, Dict[str, List[Option]]] = {}
        for opt in options:
            slot = by_expiry.setdefault(opt.expiry, {"calls": [], "puts": []})
            if opt.type == "call":
                slot["calls"].append(opt)
            else:
                slot["puts"].append(opt)

        for expiry, opts in by_expiry.items():
            calls = sorted(opts["calls"], key=lambda x: x.strike)
            puts = sorted(opts["puts"], key=lambda x: x.strike)

            if allow_short:
                # Put-call parity per matched strike.
                for call in calls:
                    for put in [p for p in puts if p.strike == call.strike]:
                        result = self.check_put_call_parity(call, put)
                        if result:
                            self.opportunities.append(result)
            else:
                # Vertical monotonicity (calls and puts).
                for result in (self.check_vertical_spread_arbitrage(calls),
                               self.check_vertical_spread_arbitrage(puts)):
                    if result:
                        self.opportunities.append(result)

            # Box spreads (both modes, different admissible direction).
            for i, c1 in enumerate(calls):
                for c2 in calls[i + 1:]:
                    p1_matches = [p for p in puts if p.strike == c1.strike]
                    p2_matches = [p for p in puts if p.strike == c2.strike]
                    for p1 in p1_matches:
                        for p2 in p2_matches:
                            if allow_short:
                                result = self.check_box_spread(c1, p1, c2, p2)
                            else:
                                result = self.check_box_spread_retail(c1, p1, c2, p2)
                            if result:
                                self.opportunities.append(result)

            # Negative-cost butterfly (both modes; retail uses looser tolerances).
            if allow_short:
                bfly_calls = self.check_butterfly_arbitrage(calls)
                bfly_puts = self.check_butterfly_arbitrage(puts)
            else:
                bfly_calls = self.check_butterfly_arbitrage(
                    calls, tolerance=0.05, spacing_tol=0.5)
                bfly_puts = self.check_butterfly_arbitrage(
                    puts, tolerance=0.05, spacing_tol=0.5)
            for result in (bfly_calls, bfly_puts):
                if result:
                    self.opportunities.append(result)

        # Calendar spreads across expiries.
        all_expiries = sorted(by_expiry.keys())
        for i, exp1 in enumerate(all_expiries):
            for exp2 in all_expiries[i + 1:]:
                for otype in ("calls", "puts"):
                    for o1 in by_expiry[exp1][otype]:
                        for o2 in [o for o in by_expiry[exp2][otype]
                                   if o.strike == o1.strike]:
                            if allow_short:
                                result = self.check_calendar_spread(o1, o2)
                            else:
                                result = self.check_calendar_spread_retail(o1, o2)
                            if result:
                                self.opportunities.append(result)

        return self.opportunities


def scan_tickers(tickers: List[str], risk_free_rate: float = 0.05,
                 allow_short: bool = True) -> Dict[str, List[Dict]]:
    """Scan multiple tickers for arbitrage opportunities."""
    all_results = {}

    mode = "INSTITUTIONAL (short selling allowed)" if allow_short else "RETAIL (no short selling)"
    print("=" * 80)
    print(f"OPTIONS ARBITRAGE SCANNER - {mode}")
    print("=" * 80)

    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(f"Scanning {ticker}...")
        print("=" * 80)

        try:
            fetcher = OptionsDataFetcher(ticker)
            spot_price, options = fetcher.fetch_options()

            if not options:
                print(f"No valid options data for {ticker}")
                all_results[ticker] = []
                continue

            detector = ArbitrageDetector(spot_price=spot_price, risk_free_rate=risk_free_rate)
            opportunities = detector.find_all_arbitrage(options, allow_short=allow_short)

            if opportunities:
                all_results[ticker] = opportunities
                print(f"\nFound {len(opportunities)} candidate opportunities for {ticker}:")
                print("-" * 80)
                for i, opp in enumerate(opportunities, 1):
                    print(f"\nOpportunity #{i}: {opp['type']}")
                    print("-" * 40)
                    for key, value in opp.items():
                        if key != "type":
                            if isinstance(value, float):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
            else:
                print(f"\nNo arbitrage opportunities found for {ticker}")
                all_results[ticker] = []

        except (ValueError, KeyError, ConnectionError) as e:
            print(f"\nError processing {ticker}: {e}")
            all_results[ticker] = []
            continue

    print("\n" + "=" * 80)
    print("SCAN SUMMARY")
    print("=" * 80)
    total = sum(len(o) for o in all_results.values())
    print(f"Total tickers scanned: {len(tickers)}")
    print(f"Total opportunities found: {total}")
    if total > 0:
        print("\nOpportunities by ticker:")
        for ticker, opps in all_results.items():
            if opps:
                print(f"  {ticker}: {len(opps)} opportunities")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static options no-arbitrage scanner")
    parser.add_argument("tickers", nargs="*",
                        default=["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"],
                        help="Ticker symbols to scan")
    parser.add_argument("--rate", type=float, default=0.045,
                        help="Annual risk-free rate")
    parser.add_argument("--retail", action="store_true",
                        help="Retail mode: no short selling (default assumes shorting is allowed)")
    args = parser.parse_args()

    scan_tickers(args.tickers, risk_free_rate=args.rate, allow_short=not args.retail)

    print("\n" + "=" * 80)
    print("Scan complete.")
    print("=" * 80)
