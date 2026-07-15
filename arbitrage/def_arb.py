import numpy as np
import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Option:
    """Represents an option contract"""
    strike: float
    expiry: datetime
    type: str  # 'call' or 'put'
    bid: float
    ask: float
    
    @property
    def mid(self):
        return (self.bid + self.ask) / 2

class OptionsDataFetcher:
    """Fetches options data from Yahoo Finance"""
    
    def __init__(self, ticker: str):
        """
        Initialize data fetcher for a ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        """
        self.ticker = ticker
        self.stock = None
        self.spot_price = None
        
    def fetch_options(self) -> tuple[float, List[Option]]:
        """
        Fetch all available options for the ticker
        
        Returns:
            tuple: (spot_price, list of Option objects)
        """
        try:
            self.stock = yf.Ticker(self.ticker)
            
            # Get current stock price
            hist = self.stock.history(period='1d')
            if hist.empty:
                raise ValueError(f"Could not fetch price data for {self.ticker}")
            
            self.spot_price = hist['Close'].iloc[-1]
            
            # Get all expiration dates
            expirations = self.stock.options
            
            if not expirations:
                raise ValueError(f"No options data available for {self.ticker}")
            
            print(f"\n{self.ticker} - Current Price: ${self.spot_price:.2f}")
            print(f"Available expirations: {len(expirations)}")
            
            all_options = []
            
            # Fetch options for each expiration
            for expiry_str in expirations:
                try:
                    opt_chain = self.stock.option_chain(expiry_str)
                    
                    # Convert expiry string to datetime
                    expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                    
                    # Process calls
                    calls = opt_chain.calls
                    for _, row in calls.iterrows():
                        # Only include options with valid bid/ask
                        if row['bid'] > 0 and row['ask'] > 0 and row['ask'] > row['bid']:
                            option = Option(
                                strike=float(row['strike']),
                                expiry=expiry_date,
                                type='call',
                                bid=float(row['bid']),
                                ask=float(row['ask'])
                            )
                            all_options.append(option)
                    
                    # Process puts
                    puts = opt_chain.puts
                    for _, row in puts.iterrows():
                        # Only include options with valid bid/ask
                        if row['bid'] > 0 and row['ask'] > 0 and row['ask'] > row['bid']:
                            option = Option(
                                strike=float(row['strike']),
                                expiry=expiry_date,
                                type='put',
                                bid=float(row['bid']),
                                ask=float(row['ask'])
                            )
                            all_options.append(option)
                    
                except Exception as e:
                    print(f"  Warning: Could not fetch options for {expiry_str}: {e}")
                    continue
            
            print(f"Successfully fetched {len(all_options)} option contracts")
            
            return self.spot_price, all_options
            
        except Exception as e:
            print(f"Error fetching options for {self.ticker}: {e}")
            return None, []


class ArbitrageDetector:
    def __init__(self, spot_price: float, risk_free_rate: float = 0.05):
        """
        Initialize arbitrage detector
        
        Args:
            spot_price: Current price of underlying asset
            risk_free_rate: Annual risk-free interest rate
        """
        self.spot = spot_price
        self.r = risk_free_rate
        self.opportunities = []
    
    def time_to_expiry(self, expiry: datetime) -> float:
        """Calculate time to expiry in years"""
        now = datetime.now()
        days = (expiry - now).days
        return max(days / 365.0, 0.001)  # Avoid division by zero
    
    def check_put_call_parity(self, call: Option, put: Option) -> Optional[Dict]:
        """
        Check for put-call parity violations
        
        Put-Call Parity: C - P = S - K*e^(-r*T)
        If violated, arbitrage exists
        
        For retail (no short selling):
        - Only flag if we can BUY underpriced side
        """
        if call.strike != put.strike or call.expiry != put.expiry:
            return None
        
        K = call.strike
        T = self.time_to_expiry(call.expiry)
        
        # Theoretical relationship
        pv_strike = K * np.exp(-self.r * T)
        theoretical_diff = self.spot - pv_strike
        
        # Actual market prices (use ask for buying, bid for selling)
        # For buying: C(ask) - P(ask) vs theoretical
        # For conversion: Buy call, sell put, short stock (NOT AVAILABLE)
        
        # Check if synthetic long (buy call, sell put) is cheaper than buying stock
        synthetic_long_cost = call.ask - put.bid
        
        # Can't short stock, so skip traditional put-call parity arbitrage
        # Most put-call parity requires short selling
        
        return None
    
    def check_vertical_spread_arbitrage(self, options: List[Option]) -> Optional[Dict]:
        """
        Check for vertical spread arbitrage (no short selling needed)
        
        For calls: Lower strike should always be worth MORE than higher strike
        For puts: Higher strike should always be worth MORE than lower strike
        
        Arbitrage: If we can buy a more valuable option for less money
        """
        options = sorted(options, key=lambda x: x.strike)
        
        if len(options) < 2:
            return None
        
        opportunities = []
        
        for i in range(len(options) - 1):
            opt1 = options[i]
            opt2 = options[i + 1]
            
            # Must be same expiry and type
            if opt1.expiry != opt2.expiry or opt1.type != opt2.type:
                continue
            
            if opt1.type == 'call':
                # Lower strike call should be worth more
                # If we can BUY lower strike call cheaper than higher strike, arbitrage!
                if opt1.ask < opt2.bid - 0.02:  # With tolerance
                    profit = opt2.bid - opt1.ask
                    return {
                        'type': 'Vertical Spread Arbitrage (Call)',
                        'strikes': f'{opt1.strike}/{opt2.strike}',
                        'expiry': opt1.expiry,
                        'strategy': f'Buy ${opt1.strike} call @ ${opt1.ask:.2f}, Sell ${opt2.strike} call @ ${opt2.bid:.2f}',
                        'estimated_profit': profit,
                        'capital_required': opt1.ask,
                        'risk': 'Limited to premium paid',
                        'execution': '1. Buy lower strike call\n2. Immediately sell higher strike call\n3. Pocket the difference'
                    }
            
            elif opt1.type == 'put':
                # Higher strike put should be worth more
                # If we can BUY higher strike put cheaper than lower strike, arbitrage!
                if opt2.ask < opt1.bid - 0.02:  # With tolerance
                    profit = opt1.bid - opt2.ask
                    return {
                        'type': 'Vertical Spread Arbitrage (Put)',
                        'strikes': f'{opt1.strike}/{opt2.strike}',
                        'expiry': opt1.expiry,
                        'strategy': f'Buy ${opt2.strike} put @ ${opt2.ask:.2f}, Sell ${opt1.strike} put @ ${opt1.bid:.2f}',
                        'estimated_profit': profit,
                        'capital_required': opt2.ask,
                        'risk': 'Limited to premium paid',
                        'execution': '1. Buy higher strike put\n2. Immediately sell lower strike put\n3. Pocket the difference'
                    }
        
        return None
    
    def check_box_spread_retail(self, call_low: Option, put_low: Option, 
                                call_high: Option, put_high: Option) -> Optional[Dict]:
        """
        Check for box spread arbitrage (retail-friendly version)
        
        A box spread is risk-free if priced correctly.
        Retail can BUY a box if market price < theoretical value
        
        No short selling required - just buying/selling options
        """
        # Verify all options have same expiry
        if not (call_low.expiry == put_low.expiry == call_high.expiry == put_high.expiry):
            return None
        
        K1 = min(call_low.strike, call_high.strike)
        K2 = max(call_low.strike, call_high.strike)
        
        if K1 >= K2 or (K2 - K1) < 1:  # Need meaningful spread
            return None
        
        T = self.time_to_expiry(call_low.expiry)
        
        # Theoretical value of box spread (what it SHOULD be worth at expiry)
        theoretical_value = (K2 - K1) * np.exp(-self.r * T)
        
        # Cost to BUY box spread (using ASK prices for buying, BID for selling)
        # Buy call at low strike, sell call at high strike
        # Buy put at high strike, sell put at low strike
        buy_box_cost = (call_low.ask - call_high.bid) + (put_high.ask - put_low.bid)
        
        # RETAIL ARBITRAGE: Can only profit if we can BUY box cheaper than theoretical value
        tolerance = 0.05  # Higher tolerance for retail (spreads, commissions)
        
        if buy_box_cost < theoretical_value - tolerance:
            profit = theoretical_value - buy_box_cost
            
            # Calculate return on investment
            roi = (profit / buy_box_cost) * 100 if buy_box_cost > 0 else 0
            
            return {
                'type': 'Box Spread Arbitrage (Retail Buy)',
                'strikes': f'{K1}/{K2}',
                'expiry': call_low.expiry,
                'theoretical_value': theoretical_value,
                'market_cost': buy_box_cost,
                'strategy': f'Buy Box Spread: Buy ${K1} call, Sell ${K2} call, Buy ${K2} put, Sell ${K1} put',
                'estimated_profit': profit,
                'roi_percent': roi,
                'capital_required': buy_box_cost,
                'risk': 'Nearly risk-free (box locks in profit)',
                'execution': '1. Simultaneously execute all 4 legs\n2. Hold until expiration\n3. Box guaranteed to be worth (K2-K1)'
            }
        
        return None
    
    def check_butterfly_arbitrage(self, options: List[Option]) -> Optional[Dict]:
        """
        Check if butterfly spread violates no-arbitrage bounds
        Butterfly should always have non-negative cost
        
        RETAIL: If we can SELL a butterfly for more than it can possibly be worth, arbitrage!
        But we need to BUY it for less than minimum value (which should be 0)
        
        Actually, for retail: if butterfly has NEGATIVE cost (we get paid to enter), that's arbitrage!
        """
        # Need 3 strikes with same expiry and type
        options = sorted(options, key=lambda x: x.strike)
        
        if len(options) < 3:
            return None
        
        for i in range(len(options) - 2):
            low = options[i]
            mid = options[i + 1]
            high = options[i + 2]
            
            if low.expiry != mid.expiry or mid.expiry != high.expiry:
                continue
            if low.type != mid.type or mid.type != high.type:
                continue
            
            # Check if strikes are evenly spaced (or close to it)
            wing1 = mid.strike - low.strike
            wing2 = high.strike - mid.strike
            if abs(wing1 - wing2) > 0.5:  # Allow some flexibility
                continue
            
            # Cost to BUY butterfly: buy 1 low, sell 2 mid, buy 1 high
            # Using real bid/ask: buy at ASK, sell at BID
            buy_cost = low.ask - 2 * mid.bid + high.ask
            
            # Butterfly value is always between 0 and wing width
            # If cost is NEGATIVE, we get paid to enter - that's arbitrage!
            if buy_cost < -0.05:  # We get paid to enter!
                max_profit = min(wing1, wing2)
                return {
                    'type': 'Butterfly Arbitrage (Negative Cost)',
                    'strikes': f'{low.strike}/{mid.strike}/{high.strike}',
                    'option_type': low.type,
                    'expiry': low.expiry,
                    'strategy': f'Buy ${low.strike}, Sell 2x ${mid.strike}, Buy ${high.strike}',
                    'entry_cost': buy_cost,
                    'estimated_profit': -buy_cost,  # We receive this to enter
                    'max_additional_profit': max_profit,
                    'capital_required': max(0, buy_cost),  # Might receive credit
                    'risk': 'Limited to wing width minus credit received',
                    'execution': '1. Execute all 3 legs simultaneously\n2. Get paid to enter position\n3. Max profit if stock lands at middle strike'
                }
        
        return None
    
    def check_calendar_spread(self, near: Option, far: Option) -> Optional[Dict]:
        """
        Check for calendar spread violations (retail-friendly)
        
        Far-dated option must be worth at least as much as near-dated option
        If near-dated costs MORE to buy, we can't arbitrage without short selling
        
        But if far-dated COSTS LESS, we can buy it and sell near-term!
        """
        if near.strike != far.strike or near.type != far.type:
            return None
        
        if near.expiry >= far.expiry:
            return None
        
        # RETAIL OPPORTUNITY: If far option (more time value) costs LESS than near
        # We can BUY far, SELL near, and profit when near expires
        if far.ask < near.bid - 0.05:  # Tolerance for spreads
            profit = near.bid - far.ask
            
            return {
                'type': 'Calendar Spread Arbitrage (Retail)',
                'strike': near.strike,
                'option_type': near.type,
                'near_expiry': near.expiry,
                'far_expiry': far.expiry,
                'strategy': f'Buy far-dated @ ${far.ask:.2f}, Sell near-dated @ ${near.bid:.2f}',
                'estimated_profit': profit,
                'capital_required': far.ask,
                'risk': 'Limited risk - you own the longer-dated option',
                'execution': '1. Buy far-dated option\n2. Sell near-dated option\n3. After near expires, you own far option'
            }
        
        return None
    
    def find_all_arbitrage(self, options: List[Option]) -> List[Dict]:
        """
        Scan all options for arbitrage opportunities (RETAIL-FRIENDLY - NO SHORT SELLING)
        
        Args:
            options: List of Option objects
            
        Returns:
            List of arbitrage opportunities found
        """
        self.opportunities = []
        
        # Group options by expiry and strike
        by_expiry = {}
        for opt in options:
            key = opt.expiry
            if key not in by_expiry:
                by_expiry[key] = {'calls': [], 'puts': []}
            
            if opt.type == 'call':
                by_expiry[key]['calls'].append(opt)
            else:
                by_expiry[key]['puts'].append(opt)
        
        # Check each expiry
        for expiry, opts in by_expiry.items():
            calls = sorted(opts['calls'], key=lambda x: x.strike)
            puts = sorted(opts['puts'], key=lambda x: x.strike)
            
            # 1. Check VERTICAL SPREAD arbitrage (most common for retail)
            result = self.check_vertical_spread_arbitrage(calls)
            if result:
                self.opportunities.append(result)
            
            result = self.check_vertical_spread_arbitrage(puts)
            if result:
                self.opportunities.append(result)
            
            # 2. Check BOX SPREADS (retail can BUY underpriced boxes)
            for i, c1 in enumerate(calls):
                for c2 in calls[i+1:]:
                    p1_matches = [p for p in puts if p.strike == c1.strike]
                    p2_matches = [p for p in puts if p.strike == c2.strike]
                    
                    for p1 in p1_matches:
                        for p2 in p2_matches:
                            result = self.check_box_spread_retail(c1, p1, c2, p2)
                            if result:
                                self.opportunities.append(result)
            
            # 3. Check BUTTERFLY spreads (negative cost = free money)
            result = self.check_butterfly_arbitrage(calls)
            if result:
                self.opportunities.append(result)
            
            result = self.check_butterfly_arbitrage(puts)
            if result:
                self.opportunities.append(result)
        
        # 4. Check CALENDAR spreads (can buy/sell without shorting)
        all_expiries = sorted(by_expiry.keys())
        for i, exp1 in enumerate(all_expiries):
            for exp2 in all_expiries[i+1:]:
                # Compare calls
                for c1 in by_expiry[exp1]['calls']:
                    matching = [c for c in by_expiry[exp2]['calls'] 
                               if c.strike == c1.strike]
                    for c2 in matching:
                        result = self.check_calendar_spread(c1, c2)
                        if result:
                            self.opportunities.append(result)
                
                # Compare puts
                for p1 in by_expiry[exp1]['puts']:
                    matching = [p for p in by_expiry[exp2]['puts'] 
                               if p.strike == p1.strike]
                    for p2 in matching:
                        result = self.check_calendar_spread(p1, p2)
                        if result:
                            self.opportunities.append(result)
        
        return self.opportunities


def scan_tickers(tickers: List[str], risk_free_rate: float = 0.05) -> Dict[str, List[Dict]]:
    """
    Scan multiple tickers for arbitrage opportunities
    
    Args:
        tickers: List of ticker symbols to scan
        risk_free_rate: Annual risk-free interest rate
        
    Returns:
        Dictionary mapping ticker to list of arbitrage opportunities
    """
    all_results = {}
    
    print("=" * 80)
    print("OPTIONS ARBITRAGE SCANNER")
    print("=" * 80)
    
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Scanning {ticker}...")
        print('='*80)
        
        try:
            # Fetch options data
            fetcher = OptionsDataFetcher(ticker)
            spot_price, options = fetcher.fetch_options()
            
            if not options:
                print(f"No valid options data for {ticker}")
                continue
            
            # Initialize detector
            detector = ArbitrageDetector(spot_price=spot_price, risk_free_rate=risk_free_rate)
            
            # Find arbitrage opportunities
            opportunities = detector.find_all_arbitrage(options)
            
            if opportunities:
                all_results[ticker] = opportunities
                print(f"\n🎯 FOUND {len(opportunities)} ARBITRAGE OPPORTUNITIES for {ticker}:")
                print("-" * 80)
                
                for i, opp in enumerate(opportunities, 1):
                    print(f"\nOpportunity #{i}: {opp['type']}")
                    print("-" * 40)
                    for key, value in opp.items():
                        if key != 'type':
                            if isinstance(value, float):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
            else:
                print(f"\n✓ No arbitrage opportunities found for {ticker}")
                all_results[ticker] = []
                
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            all_results[ticker] = []
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("SCAN SUMMARY")
    print("=" * 80)
    total_opportunities = sum(len(opps) for opps in all_results.values())
    print(f"Total tickers scanned: {len(tickers)}")
    print(f"Total opportunities found: {total_opportunities}")
    
    if total_opportunities > 0:
        print("\nOpportunities by ticker:")
        for ticker, opps in all_results.items():
            if opps:
                print(f"  {ticker}: {len(opps)} opportunities")
    
    return all_results


# Example usage
if __name__ == "__main__":
    # List of tickers to scan
    tickers_to_scan = [
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'GOOGL', # Google
        'TSLA',  # Tesla
        'SPY',   # S&P 500 ETF
    ]
    
    # Scan all tickers
    results = scan_tickers(tickers_to_scan, risk_free_rate=0.045)
    
    # You can also scan a single ticker
    # results = scan_tickers(['AAPL'])
    
    print("\n" + "=" * 80)
    print("Scan complete! Check results above.")
    print("=" * 80)