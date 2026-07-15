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
        """
        if call.strike != put.strike or call.expiry != put.expiry:
            return None
        
        K = call.strike
        T = self.time_to_expiry(call.expiry)
        
        # Theoretical relationship
        pv_strike = K * np.exp(-self.r * T)
        theoretical_diff = self.spot - pv_strike
        
        # Actual market prices (use mid prices for simplicity)
        actual_diff = call.mid - put.mid
        
        # Check for violation (with small tolerance for transaction costs)
        tolerance = 0.01
        discrepancy = abs(actual_diff - theoretical_diff)
        
        if discrepancy > tolerance:
            # Determine arbitrage direction
            if actual_diff > theoretical_diff:
                # Call overpriced or Put underpriced
                strategy = "Sell Call, Buy Put, Buy Stock, Borrow PV(K)"
                profit = actual_diff - theoretical_diff
            else:
                # Call underpriced or Put overpriced
                strategy = "Buy Call, Sell Put, Short Stock, Lend PV(K)"
                profit = theoretical_diff - actual_diff
            
            return {
                'type': 'Put-Call Parity Violation',
                'strike': K,
                'expiry': call.expiry,
                'theoretical_diff': theoretical_diff,
                'actual_diff': actual_diff,
                'discrepancy': discrepancy,
                'strategy': strategy,
                'estimated_profit': profit
            }
        
        return None
    
    def check_box_spread(self, call_low: Option, put_low: Option, 
                        call_high: Option, put_high: Option) -> Optional[Dict]:
        """
        Check for box spread arbitrage
        
        A box spread combines a bull call spread and a bear put spread
        Theoretical value = (K2 - K1) * e^(-r*T)
        """
        # Verify all options have same expiry
        if not (call_low.expiry == put_low.expiry == call_high.expiry == put_high.expiry):
            return None
        
        K1 = min(call_low.strike, call_high.strike)
        K2 = max(call_low.strike, call_high.strike)
        
        if K1 >= K2:
            return None
        
        T = self.time_to_expiry(call_low.expiry)
        
        # Theoretical value of box spread
        theoretical_value = (K2 - K1) * np.exp(-self.r * T)
        
        # Cost to establish box spread (buy low call, sell high call, buy high put, sell low put)
        market_cost = (call_low.ask - call_high.bid) + (put_high.ask - put_low.bid)
        
        # Check for arbitrage
        tolerance = 0.01
        discrepancy = abs(market_cost - theoretical_value)
        
        if discrepancy > tolerance:
            if market_cost < theoretical_value:
                profit = theoretical_value - market_cost
                return {
                    'type': 'Box Spread Arbitrage',
                    'strikes': f'{K1}/{K2}',
                    'expiry': call_low.expiry,
                    'theoretical_value': theoretical_value,
                    'market_cost': market_cost,
                    'strategy': 'Buy Box (Long Call Spread + Long Put Spread)',
                    'estimated_profit': profit
                }
            else:
                profit = market_cost - theoretical_value
                return {
                    'type': 'Box Spread Arbitrage',
                    'strikes': f'{K1}/{K2}',
                    'expiry': call_low.expiry,
                    'theoretical_value': theoretical_value,
                    'market_cost': market_cost,
                    'strategy': 'Sell Box (Short Call Spread + Short Put Spread)',
                    'estimated_profit': profit
                }
        
        return None
    
    def check_butterfly_arbitrage(self, options: List[Option]) -> Optional[Dict]:
        """
        Check if butterfly spread violates no-arbitrage bounds
        Butterfly should always have non-negative value
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
            
            # Check if strikes are evenly spaced
            if abs((mid.strike - low.strike) - (high.strike - mid.strike)) > 0.01:
                continue
            
            # Cost of butterfly: buy 1 low, sell 2 mid, buy 1 high
            cost = low.ask - 2 * mid.bid + high.ask
            
            # Butterfly should never have negative cost (arbitrage)
            if cost < -0.01:  # Small tolerance
                return {
                    'type': 'Butterfly Arbitrage',
                    'strikes': f'{low.strike}/{mid.strike}/{high.strike}',
                    'option_type': low.type,
                    'expiry': low.expiry,
                    'strategy': 'Buy low strike, Sell 2x mid strike, Buy high strike',
                    'estimated_profit': -cost
                }
        
        return None
    
    def check_calendar_spread(self, near: Option, far: Option) -> Optional[Dict]:
        """
        Check for calendar spread violations
        Far-dated option must be worth at least as much as near-dated option
        """
        if near.strike != far.strike or near.type != far.type:
            return None
        
        if near.expiry >= far.expiry:
            return None
        
        # Far option should be worth more (more time value)
        if near.bid > far.ask + 0.01:  # Tolerance for transaction costs
            profit = near.bid - far.ask
            return {
                'type': 'Calendar Spread Arbitrage',
                'strike': near.strike,
                'option_type': near.type,
                'near_expiry': near.expiry,
                'far_expiry': far.expiry,
                'strategy': 'Sell near-dated, Buy far-dated',
                'estimated_profit': profit
            }
        
        return None
    
    def find_all_arbitrage(self, options: List[Option]) -> List[Dict]:
        """
        Scan all options for arbitrage opportunities
        
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
        
        # Check put-call parity for each expiry
        for expiry, opts in by_expiry.items():
            calls = sorted(opts['calls'], key=lambda x: x.strike)
            puts = sorted(opts['puts'], key=lambda x: x.strike)
            
            # Match calls and puts by strike
            for call in calls:
                matching_puts = [p for p in puts if p.strike == call.strike]
                for put in matching_puts:
                    result = self.check_put_call_parity(call, put)
                    if result:
                        self.opportunities.append(result)
            
            # Check box spreads
            for i, c1 in enumerate(calls):
                for c2 in calls[i+1:]:
                    p1_matches = [p for p in puts if p.strike == c1.strike]
                    p2_matches = [p for p in puts if p.strike == c2.strike]
                    
                    for p1 in p1_matches:
                        for p2 in p2_matches:
                            result = self.check_box_spread(c1, p1, c2, p2)
                            if result:
                                self.opportunities.append(result)
            
            # Check butterfly spreads
            result = self.check_butterfly_arbitrage(calls)
            if result:
                self.opportunities.append(result)
            
            result = self.check_butterfly_arbitrage(puts)
            if result:
                self.opportunities.append(result)
        
        # Check calendar spreads
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