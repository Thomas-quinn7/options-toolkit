import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from collections import namedtuple

warnings.filterwarnings("ignore", category=RuntimeWarning)

GLOBAL_INDICES = {
    "US_SP500_MEGA": {
        "name": "S&P 500 Mega Caps",
        "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", 
                   "BRK-B", "LLY", "AVGO", "JPM", "V", "XOM", "UNH", "MA",
                   "WMT", "JNJ", "PG", "ORCL", "HD"]
    },
    "US_SP500_LARGE": {
        "name": "S&P 500 Large Caps",
        "tickers": ["COST", "NFLX", "CRM", "BAC", "ABBV", "CVX", "MRK", "AMD",
                   "KO", "PEP", "ADBE", "TMO", "ACN", "CSCO", "MCD", "LIN",
                   "ABT", "INTC", "DHR", "CMCSA"]
    },
    "US_TECH": {
        "name": "US Technology",
        "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
                   "AVGO", "ORCL", "NFLX", "AMD", "CRM", "INTC", "CSCO",
                   "ADBE", "QCOM", "TXN", "INTU", "IBM", "NOW"]
    },
    "US_FINANCE": {
        "name": "US Financials",
        "tickers": ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BX", "C",
                   "SPGI", "AXP", "BLK", "CB", "PGR", "MMC", "CME", "USB",
                   "PNC", "TFC", "COF"]
    },
    "US_HEALTH": {
        "name": "US Healthcare",
        "tickers": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR",
                   "BMY", "AMGN", "PFE", "CVS", "ISRG", "GILD", "VRTX", "CI",
                   "REGN", "ELV", "MCK", "ZTS"]
    },
    "US_CONSUMER": {
        "name": "US Consumer",
        "tickers": ["AMZN", "TSLA", "WMT", "HD", "COST", "MCD", "NKE", "SBUX",
                   "TGT", "LOW", "DIS", "BKNG", "ABNB", "GM", "F", "MAR",
                   "CMG", "YUM", "ROST", "ORLY"]
    },
    "US_ENERGY": {
        "name": "US Energy & Materials",
        "tickers": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX",
                   "VLO", "OXY", "HAL", "WMB", "KMI", "LNG", "HES", "FANG",
                   "DVN", "BKR", "TRGP", "MRO"]
    },
    "US_INDUSTRIAL": {
        "name": "US Industrials",
        "tickers": ["BA", "CAT", "GE", "UPS", "RTX", "HON", "UNP", "LMT", "DE",
                   "ADP", "GD", "NOC", "ETN", "ITW", "MMM", "CSX", "EMR",
                   "TT", "WM", "NSC"]
    },
    "US_SMALL_MID": {
        "name": "US Small/Mid Caps",
        "tickers": ["MRVL", "PANW", "PLTR", "CRWD", "SNOW", "FTNT", "DDOG",
                   "NET", "TEAM", "ZS", "UBER", "LYFT", "DASH", "COIN",
                   "RBLX", "U", "AFRM", "SQ", "RIVN", "LCID"]
    },
    "US_MEME": {
        "name": "US Retail Favorites",
        "tickers": ["GME", "AMC", "BBBY", "BB", "NOK", "WISH", "CLOV", "SPCE",
                   "SOFI", "HOOD", "PLTR", "NIO", "RIVN", "LCID", "AFRM",
                   "COIN", "RBLX", "DKNG", "SKLZ", "OPEN"]
    },
    "ETF_MAJOR": {
        "name": "Major ETFs",
        "tickers": ["SPY", "QQQ", "IWM", "DIA", "EEM", "EFA", "GLD", "SLV",
                   "TLT", "HYG", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
                   "XLY", "XLU", "XLRE", "XLC"]
    }
}

STRIKE_NEIGHBOR_COUNT = 3
MAX_STRIKE_WINDOW_PCT = 10.0
INITIAL_WINDOW_PCT = 2.0
WINDOW_EXPANSION_FACTOR = 2.0

OTM_PUT_DELTA_TARGET = 0.25 
OTM_CALL_DELTA_TARGET = 0.25

OTM_PUT_BAND = 0.90  
OTM_CALL_BAND = 1.10 

CRITICAL_INVERSION_THRESHOLD = 60
WARNING_INVERSION_THRESHOLD = 40

DAILY_SNAPSHOT_CSV = "daily_IV_skew_snapshot.csv"
BUBBLE_SUMMARY_CSV = "bubble_summary.csv"

# Data quality constants
MIN_VOLUME = 10
MIN_OPEN_INTEREST = 50
MAX_BID_ASK_SPREAD_PCT = 20
MIN_IV = 0.05
MAX_IV = 3.0
MIN_DTE = 7
MAX_DTE = 60

# ============== DATA VALIDATION FUNCTIONS ==============

def validate_option_data(options_df: pd.DataFrame) -> pd.DataFrame:
    """Filter options data to ensure quality and reliability."""
    if options_df.empty:
        return options_df
    
    df = options_df.copy()
    
    # Filter 1: Volume and Open Interest
    df = df[
        (df["volume"] >= MIN_VOLUME) & 
        (df["openInterest"] >= MIN_OPEN_INTEREST)
    ]
    
    # Filter 2: Bid-Ask Spread (if available)
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread_pct"] = ((df["ask"] - df["bid"]) / df["mid"]) * 100
        df = df[df["spread_pct"] <= MAX_BID_ASK_SPREAD_PCT]
    
    # Filter 3: IV Sanity Checks
    df = df[
        (df["impliedVolatility"] >= MIN_IV) & 
        (df["impliedVolatility"] <= MAX_IV) &
        (df["impliedVolatility"].notna())
    ]
    
    # Filter 4: Remove zero or negative strikes
    df = df[df["strike"] > 0]
    
    return df


def validate_expiry_date(expiry_str: str) -> Tuple[bool, int]:
    """Check if expiry date is within acceptable range."""
    try:
        expiry_date = pd.to_datetime(expiry_str)
        today = pd.Timestamp.now()
        dte = (expiry_date - today).days
        
        is_valid = MIN_DTE <= dte <= MAX_DTE
        return is_valid, dte
    except:
        return False, 0


def get_best_expiry(expiry_dates: list) -> Optional[str]:
    """Select the best expiry date for analysis. Prefer ~30 DTE."""
    valid_expiries = []
    
    for exp in expiry_dates:
        is_valid, dte = validate_expiry_date(exp)
        if is_valid:
            valid_expiries.append((exp, dte))
    
    if not valid_expiries:
        return None
    
    # Prefer expiry closest to 30 days
    valid_expiries.sort(key=lambda x: abs(x[1] - 30))
    return valid_expiries[0][0]


def compute_data_quality_score(
    puts_df: pd.DataFrame, 
    calls_df: pd.DataFrame,
    spot_price: float
) -> Dict[str, float]:
    """Calculate data quality metrics to flag unreliable results."""
    try:
        otm_puts = puts_df[puts_df["strike"] < spot_price * 0.95]
        otm_calls = calls_df[calls_df["strike"] > spot_price * 1.05]
        
        quality_metrics = {
            "otm_put_count": len(otm_puts),
            "otm_call_count": len(otm_calls),
            "otm_put_avg_volume": otm_puts["volume"].mean() if len(otm_puts) > 0 else 0,
            "otm_call_avg_volume": otm_calls["volume"].mean() if len(otm_calls) > 0 else 0,
            "otm_put_avg_oi": otm_puts["openInterest"].mean() if len(otm_puts) > 0 else 0,
            "otm_call_avg_oi": otm_calls["openInterest"].mean() if len(otm_calls) > 0 else 0,
        }
        
        # Overall quality score (0-1)
        min_options = 5
        min_volume = 50
        
        quality_score = 0
        if quality_metrics["otm_put_count"] >= min_options:
            quality_score += 0.25
        if quality_metrics["otm_call_count"] >= min_options:
            quality_score += 0.25
        if quality_metrics["otm_put_avg_volume"] >= min_volume:
            quality_score += 0.25
        if quality_metrics["otm_call_avg_volume"] >= min_volume:
            quality_score += 0.25
        
        quality_metrics["quality_score"] = quality_score
        
        return quality_metrics
    except:
        return {"quality_score": 0.0}


# ============== HELPER FUNCTIONS ==============

def get_nearest_strike_indices(strikes: np.ndarray, target: float, n: int) -> np.ndarray:
    """Return indices of the n nearest strikes to the target price."""
    if len(strikes) == 0:
        return np.array([], dtype=int)
    distances = np.abs(strikes - target)
    return np.argsort(distances)[:n]


def compute_robust_mean_iv(iv_values: np.ndarray) -> float:
    """Calculate mean of finite IV values, filtering out NaN and inf."""
    iv_clean = np.array(iv_values, dtype=float)
    iv_clean = iv_clean[np.isfinite(iv_clean)]
    return np.mean(iv_clean) if len(iv_clean) > 0 else np.nan


def get_current_spot_price(ticker: str) -> float:
    """Retrieve current market price for ticker."""
    try:
        tk = yf.Ticker(ticker)
        price = tk.info.get("regularMarketPrice") or tk.info.get("currentPrice")
        return float(price) if price is not None else np.nan
    except Exception:
        return np.nan


# ============== OPTION CHAIN FETCHING ==============

def fetch_option_chain(ticker: str, max_retries: int = 3) -> Tuple[str, Optional[str], Optional[object], Dict]:
    """Fetch option chain with quality validation."""
    for attempt in range(max_retries):
        try:
            tk = yf.Ticker(ticker)
            expiry_dates = tk.options
            
            if not expiry_dates:
                return ticker, None, None, {}
            
            # Use best expiry instead of nearest
            best_expiry = get_best_expiry(expiry_dates)
            if not best_expiry:
                return ticker, None, None, {}
            
            chain = tk.option_chain(best_expiry)
            
            # Validate option data quality
            spot_price = get_current_spot_price(ticker)
            if np.isnan(spot_price):
                return ticker, None, None, {}
            
            # Apply data quality filters
            validated_puts = validate_option_data(chain.puts)
            validated_calls = validate_option_data(chain.calls)
            
            # Check if we have enough data
            if len(validated_puts) < 3 or len(validated_calls) < 3:
                return ticker, None, None, {}
            
            # Create new chain with validated data
            ValidatedChain = namedtuple('ValidatedChain', ['puts', 'calls'])
            validated_chain = ValidatedChain(puts=validated_puts, calls=validated_calls)
            
            # Calculate quality metrics
            quality_metrics = compute_data_quality_score(validated_puts, validated_calls, spot_price)
            
            return ticker, best_expiry, validated_chain, quality_metrics
            
        except Exception:
            if attempt == max_retries - 1:
                pass
            continue
    
    return ticker, None, None, {}


# ============== IV CALCULATION FUNCTIONS ==============

def find_mean_iv_at_strike(
    options_df: pd.DataFrame, 
    target_strike: float,
    initial_window_pct: float = INITIAL_WINDOW_PCT,
    max_window_pct: float = MAX_STRIKE_WINDOW_PCT,
    max_attempts: int = 4
) -> float:
    """Find mean IV for options near a target strike with adaptive windowing."""
    window_pct = initial_window_pct
    attempts = 0
    
    while attempts < max_attempts:
        try:
            lower_bound = target_strike * (1 - window_pct / 100.0)
            upper_bound = target_strike * (1 + window_pct / 100.0)
            
            windowed_options = options_df[
                (options_df["strike"] >= lower_bound) & 
                (options_df["strike"] <= upper_bound)
            ]
            
            if not windowed_options.empty:
                nearest_indices = get_nearest_strike_indices(
                    windowed_options["strike"].values,
                    target_strike,
                    STRIKE_NEIGHBOR_COUNT
                )
                
                if len(nearest_indices) > 0:
                    iv_values = windowed_options.iloc[nearest_indices]["impliedVolatility"]
                    iv_values = iv_values.astype(float).replace([np.inf, -np.inf], np.nan)
                    return compute_robust_mean_iv(iv_values.values)
            
            if window_pct >= max_window_pct:
                if len(options_df) > 0:
                    nearest_indices = get_nearest_strike_indices(
                        options_df["strike"].values,
                        target_strike,
                        STRIKE_NEIGHBOR_COUNT
                    )
                    if len(nearest_indices) > 0:
                        iv_values = options_df.iloc[nearest_indices]["impliedVolatility"]
                        iv_values = iv_values.astype(float).replace([np.inf, -np.inf], np.nan)
                        return compute_robust_mean_iv(iv_values.values)
                return np.nan
            
            window_pct *= WINDOW_EXPANSION_FACTOR
            attempts += 1
            
        except Exception:
            attempts += 1
            if attempts >= max_attempts:
                return np.nan
            window_pct *= WINDOW_EXPANSION_FACTOR
            continue
    
    return np.nan


def get_otm_strike_targets(spot_price: float, chain: object) -> Tuple[float, float]:
    """Determine optimal OTM strike targets based on available strikes."""
    try:
        puts_df = chain.puts
        calls_df = chain.calls

        put_strikes = puts_df["strike"].values
        call_strikes = calls_df["strike"].values

        put_target = spot_price * OTM_PUT_BAND
        call_target = spot_price * OTM_CALL_BAND

        if len(put_strikes) > 0:
            put_diffs = np.abs(put_strikes - put_target)
            closest_put = put_strikes[np.argmin(put_diffs)]
            if closest_put < spot_price * 0.98:
                put_target = closest_put
        
        if len(call_strikes) > 0:
            call_diffs = np.abs(call_strikes - call_target)
            closest_call = call_strikes[np.argmin(call_diffs)]
            if closest_call > spot_price * 1.02:
                call_target = closest_call
        
        return put_target, call_target
        
    except Exception:
        return spot_price * OTM_PUT_BAND, spot_price * OTM_CALL_BAND


def compute_volume_weighted_iv_skew(
    puts_df: pd.DataFrame, 
    calls_df: pd.DataFrame, 
    spot_price: float
) -> Tuple[float, float]:
    """Fallback: Compute volume-weighted average IV for OTM options."""
    try:
        otm_puts = puts_df[
            (puts_df["strike"] < spot_price * 0.98) &  
            (puts_df["strike"] > spot_price * 0.80) &  
            (puts_df["volume"] > 0) &             
            (puts_df["impliedVolatility"] > 0)   
        ].copy()
        
        otm_calls = calls_df[
            (calls_df["strike"] > spot_price * 1.02) & 
            (calls_df["strike"] < spot_price * 1.20) & 
            (calls_df["volume"] > 0) &
            (calls_df["impliedVolatility"] > 0)
        ].copy()
        
        put_iv = np.nan
        if len(otm_puts) > 0:
            otm_puts["weight"] = otm_puts["volume"] / otm_puts["volume"].sum()
            put_iv = (otm_puts["impliedVolatility"] * otm_puts["weight"]).sum()
        
        call_iv = np.nan
        if len(otm_calls) > 0:
            otm_calls["weight"] = otm_calls["volume"] / otm_calls["volume"].sum()
            call_iv = (otm_calls["impliedVolatility"] * otm_calls["weight"]).sum()
        
        return put_iv, call_iv
        
    except Exception:
        return np.nan, np.nan


def compute_iv_skew_metric(ticker: str, expiry: str, chain: object, quality_metrics: Dict) -> Tuple[str, str, float, float, float, float]:
    """Calculate IV skew metric with quality score."""
    try:
        calls_df = chain.calls
        puts_df = chain.puts
        
        spot_price = get_current_spot_price(ticker)
        
        if np.isnan(spot_price):
            return ticker, expiry, np.nan, np.nan, np.nan, 0.0
        
        otm_put_strike, otm_call_strike = get_otm_strike_targets(spot_price, chain)
        
        mean_otm_put_iv = find_mean_iv_at_strike(
            puts_df, 
            otm_put_strike,
            initial_window_pct=3.0, 
            max_window_pct=15.0     
        )
        
        mean_otm_call_iv = find_mean_iv_at_strike(
            calls_df, 
            otm_call_strike,
            initial_window_pct=3.0,
            max_window_pct=15.0
        )
        
        if np.isnan(mean_otm_put_iv) or np.isnan(mean_otm_call_iv):
            mean_otm_put_iv, mean_otm_call_iv = compute_volume_weighted_iv_skew(
                puts_df, calls_df, spot_price
            )
        
        iv_skew = mean_otm_put_iv - mean_otm_call_iv
        quality_score = quality_metrics.get("quality_score", 0.0)
        
        return ticker, expiry, mean_otm_put_iv, mean_otm_call_iv, iv_skew, quality_score
        
    except Exception:
        return ticker, expiry, np.nan, np.nan, np.nan, 0.0


# ============== INDEX ANALYSIS ==============

def analyze_index(index_key: str, index_data: Dict, n_workers: int = 5) -> pd.DataFrame:
    """Analyze IV skew with quality tracking."""
    tickers = index_data["tickers"]
    results = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_option_chain, ticker): ticker 
            for ticker in tickers
        }
        
        for future in as_completed(future_to_ticker):
            ticker, expiry, chain, quality_metrics = future.result()
            
            if chain is None:
                results.append((ticker, expiry, np.nan, np.nan, np.nan, 0.0))
                continue
            
            iv_skew_result = compute_iv_skew_metric(ticker, expiry, chain, quality_metrics)
            results.append(iv_skew_result)
    
    df = pd.DataFrame(
        results,
        columns=["ticker", "expiry", "otm_put_iv", "otm_call_iv", "iv_skew", "quality_score"]
    )
    
    # Flag reliable data
    df["is_reliable"] = df["quality_score"] >= 0.5
    
    df["index"] = index_key
    df["index_name"] = index_data["name"]
    
    return df


def print_index_summary(index_key: str, index_name: str, df: pd.DataFrame) -> None:
    """Print formatted summary for an index."""
    valid_df = df.dropna(subset=["iv_skew"])
    
    if len(valid_df) == 0:
        print(f"  ⚠️  No valid data")
        return
    
    # Show data quality
    reliable_count = (valid_df["is_reliable"] == True).sum()
    
    inverted_count = (valid_df["iv_skew"] < 0).sum()
    inversion_pct = (inverted_count / len(valid_df)) * 100
    mean_skew = valid_df["iv_skew"].mean()
    median_skew = valid_df["iv_skew"].median()
    
    if inversion_pct > 70:
        status = "🔴🚨" 
    elif inversion_pct > CRITICAL_INVERSION_THRESHOLD:
        status = "🚨" 
    elif inversion_pct > WARNING_INVERSION_THRESHOLD:
        status = "🟡" 
    elif inversion_pct > 25:
        status = "🟠"
    else:
        status = "🟢"
    
    print(f"  {status} Inverted: {inverted_count}/{len(valid_df)} ({inversion_pct:.1f}%) | Mean: {mean_skew:.4f} | Median: {median_skew:.4f} | Quality: {reliable_count}/{len(valid_df)}")


def print_detailed_skews(df: pd.DataFrame) -> None:
    """Print individual ticker skews in a formatted table."""
    print("\n" + "="*110)
    print("DETAILED IV SKEW BY TICKER")
    print("="*110)
    
    for index_key in df["index"].unique():
        index_df = df[df["index"] == index_key].copy()
        index_name = index_df["index_name"].iloc[0]
        
        print(f"\n📊 {index_name} ({index_key})")
        print("-" * 110)
        
        index_df = index_df.sort_values("iv_skew")
        
        for _, row in index_df.iterrows():
            ticker = row["ticker"]
            skew = row["iv_skew"]
            put_iv = row["otm_put_iv"]
            call_iv = row["otm_call_iv"]
            quality = row["quality_score"]
            reliable = "✓" if row["is_reliable"] else "✗"
            
            if pd.isna(skew):
                status = "❌"
                skew_str = "NO DATA"
            elif skew < 0:
                status = "🔴"
                skew_str = f"{skew:+.4f}"
            else:
                status = "🟢"
                skew_str = f"{skew:+.4f}"
            
            put_str = f"{put_iv:.4f}" if not pd.isna(put_iv) else "N/A"
            call_str = f"{call_iv:.4f}" if not pd.isna(call_iv) else "N/A"
            qual_str = f"{quality:.2f}" if not pd.isna(quality) else "N/A"
            
            print(f"  {status} {ticker:12s} | Skew: {skew_str:>10s} | Put: {put_str:>8s} | Call: {call_str:>8s} | Q: {qual_str:>4s} {reliable}")


# ============== MAIN PIPELINE ==============

def run_global_iv_skew_analysis(n_workers: int = 5, show_plot: bool = False) -> pd.DataFrame:
    """Run IV skew analysis across all global indices."""
    print("="*100)
    print("GLOBAL IMPLIED VOLATILITY SKEW BUBBLE DETECTOR (DATA VALIDATED)")
    print("="*100)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analyzing {len(GLOBAL_INDICES)} market segments with {sum(len(v['tickers']) for v in GLOBAL_INDICES.values())} tickers")
    print("Note: Options data availability varies by market. US markets have the most complete data.")
    print(f"Data Quality Filters: Vol≥{MIN_VOLUME}, OI≥{MIN_OPEN_INTEREST}, {MIN_DTE}≤DTE≤{MAX_DTE}, {MIN_IV*100}%≤IV≤{MAX_IV*100}%\n")
    
    all_results = []
    
    for index_key, index_data in GLOBAL_INDICES.items():
        print(f"Analyzing {index_data['name']}...")
        df_index = analyze_index(index_key, index_data, n_workers)
        all_results.append(df_index)
        print_index_summary(index_key, index_data["name"], df_index)
    
    df_combined = pd.concat(all_results, ignore_index=True)
    
    print_detailed_skews(df_combined)
    
    print("\n" + "="*100)
    print("GLOBAL MARKET SUMMARY")
    print("="*100)
    
    valid_df = df_combined.dropna(subset=["iv_skew"])
    total_attempted = len(df_combined)
    data_success_rate = (len(valid_df) / total_attempted * 100) if total_attempted > 0 else 0
    reliable_count = (valid_df["is_reliable"] == True).sum()
    
    print(f"Data Retrieval: {len(valid_df)}/{total_attempted} successful ({data_success_rate:.1f}%)")
    print(f"High Quality Data: {reliable_count}/{len(valid_df)} ({reliable_count/len(valid_df)*100:.1f}%)" if len(valid_df) > 0 else "High Quality Data: N/A")
    
    if len(valid_df) > 0:
        total_inverted = (valid_df["iv_skew"] < 0).sum()
        global_inversion_pct = (total_inverted / len(valid_df)) * 100
        global_mean_skew = valid_df["iv_skew"].mean()
        global_median_skew = valid_df["iv_skew"].median()
        
        print(f"Total Valid Measurements: {len(valid_df)}")
        print(f"Global Inverted Count: {total_inverted}")
        print(f"Global Inversion Rate: {global_inversion_pct:.1f}%")
        print(f"Global Mean IV Skew: {global_mean_skew:.4f}")
        print(f"Global Median IV Skew: {global_median_skew:.4f}")
        
        if global_inversion_pct > 70:
            print("\n🔴🚨 EXTREME BUBBLE ALERT!")
            print("   CRITICAL: Over 70% of stocks show inverted skew!")
            print("   This represents extreme speculative mania across markets.")
            print("   Historical precedent suggests significant correction risk.")
        elif global_inversion_pct > CRITICAL_INVERSION_THRESHOLD:
            print("\n🚨 CRITICAL GLOBAL BUBBLE WARNING!")
            print("   Excessive call demand detected across multiple markets!")
            print("   IV skew heavily inverted - extreme speculative mania.")
        elif global_inversion_pct > WARNING_INVERSION_THRESHOLD:
            print("\n🟡 ELEVATED GLOBAL RISK")
            print("   Above-normal speculative activity detected.")
            print("   Monitor for further deterioration in skew patterns.")
        elif global_inversion_pct > 25:
            print("\n🟠 CAUTION: Elevated Inversion Levels")
            print("   Inversion above historical norms but not yet critical.")
    else:
        print("\n⚠️  Insufficient data to generate global summary.")
        print("   This may be due to market hours, API limitations, or data availability.")
    
    save_iv_skew_snapshot(df_combined)
    save_bubble_summary_by_index(df_combined)
    
    if show_plot and len(valid_df) > 0:
        plot_global_iv_skew(df_combined)
    
    return df_combined


def save_iv_skew_snapshot(df: pd.DataFrame) -> None:
    """Append current IV skew snapshot to historical CSV file."""
    df_with_timestamp = df.copy()
    df_with_timestamp["timestamp"] = datetime.now()
    
    csv_path = Path(DAILY_SNAPSHOT_CSV)
    
    try:
        if csv_path.exists():
            historical_df = pd.read_csv(csv_path)
            combined_df = pd.concat([historical_df, df_with_timestamp], ignore_index=True)
        else:
            combined_df = df_with_timestamp
        
        combined_df.to_csv(csv_path, index=False)
        print(f"\nDetailed snapshot saved to {DAILY_SNAPSHOT_CSV}")
        
    except Exception as e:
        print(f"\nFailed to save snapshot: {e}")


def save_bubble_summary_by_index(df: pd.DataFrame) -> None:
    """Save daily aggregate bubble summary by index AND globally to CSV."""
    timestamp = datetime.now()
    date_only = timestamp.date()
    
    summary_rows = []
    
    for index_key in df["index"].unique():
        index_df = df[df["index"] == index_key]
        index_name = index_df["index_name"].iloc[0]
        valid_df = index_df.dropna(subset=["iv_skew"])
        
        if len(valid_df) == 0:
            continue
        
        inverted_count = (valid_df["iv_skew"] < 0).sum()
        total_valid = len(valid_df)
        inversion_pct = (inverted_count / total_valid) * 100
        
        if inversion_pct > CRITICAL_INVERSION_THRESHOLD:
            alert_level = "CRITICAL"
        elif inversion_pct > WARNING_INVERSION_THRESHOLD:
            alert_level = "WARNING"
        else:
            alert_level = "NORMAL"
        
        summary_rows.append({
            "date": date_only,
            "timestamp": timestamp,
            "index": index_key,
            "index_name": index_name,
            "total_tickers": len(index_df),
            "valid_measurements": total_valid,
            "inverted_count": inverted_count,
            "inversion_pct": round(inversion_pct, 2),
            "mean_iv_skew": round(valid_df["iv_skew"].mean(), 4),
            "median_iv_skew": round(valid_df["iv_skew"].median(), 4),
            "std_iv_skew": round(valid_df["iv_skew"].std(), 4),
            "min_iv_skew": round(valid_df["iv_skew"].min(), 4),
            "max_iv_skew": round(valid_df["iv_skew"].max(), 4),
            "alert_level": alert_level
        })
    
    valid_df_global = df.dropna(subset=["iv_skew"])
    if len(valid_df_global) > 0:
        inverted_count_global = (valid_df_global["iv_skew"] < 0).sum()
        total_valid_global = len(valid_df_global)
        inversion_pct_global = (inverted_count_global / total_valid_global) * 100
        
        if inversion_pct_global > CRITICAL_INVERSION_THRESHOLD:
            alert_level_global = "CRITICAL"
        elif inversion_pct_global > WARNING_INVERSION_THRESHOLD:
            alert_level_global = "WARNING"
        else:
            alert_level_global = "NORMAL"
        
        summary_rows.append({
            "date": date_only,
            "timestamp": timestamp,
            "index": "GLOBAL",
            "index_name": "Global All Indices",
            "total_tickers": len(df),
            "valid_measurements": total_valid_global,
            "inverted_count": inverted_count_global,
            "inversion_pct": round(inversion_pct_global, 2),
            "mean_iv_skew": round(valid_df_global["iv_skew"].mean(), 4),
            "median_iv_skew": round(valid_df_global["iv_skew"].median(), 4),
            "std_iv_skew": round(valid_df_global["iv_skew"].std(), 4),
            "min_iv_skew": round(valid_df_global["iv_skew"].min(), 4),
            "max_iv_skew": round(valid_df_global["iv_skew"].max(), 4),
            "alert_level": alert_level_global
        })
    
    summary_path = Path(BUBBLE_SUMMARY_CSV)
    
    try:
        new_summary_df = pd.DataFrame(summary_rows)
        
        if summary_path.exists():
            historical_summary = pd.read_csv(summary_path)
            historical_summary['date'] = pd.to_datetime(historical_summary['date']).dt.date
            
            historical_summary = historical_summary[
                ~((historical_summary['date'] == date_only) & 
                  (historical_summary['index'].isin(new_summary_df['index'])))
            ]
            
            combined_summary = pd.concat([historical_summary, new_summary_df], ignore_index=True)
        else:
            combined_summary = new_summary_df
        
        combined_summary = combined_summary.sort_values(['date', 'index'], ascending=[True, True])
        combined_summary.to_csv(summary_path, index=False)
        print(f"Bubble summary by index saved to {BUBBLE_SUMMARY_CSV}")
        
    except Exception as e:
        print(f"Failed to save bubble summary: {e}")


def plot_global_iv_skew(df: pd.DataFrame) -> None:
    """Visualize IV skew metrics by index."""
    indices = df["index"].unique()
    
    fig, axes = plt.subplots(len(indices), 1, figsize=(14, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]
    
    for idx, index_key in enumerate(indices):
        index_df = df[df["index"] == index_key].sort_values("iv_skew")
        index_name = index_df["index_name"].iloc[0]
        
        colors = ['red' if x < 0 else 'green' for x in index_df["iv_skew"]]
        
        axes[idx].bar(index_df["ticker"], index_df["iv_skew"], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[idx].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        axes[idx].set_title(f"{index_name}", fontsize=12, fontweight='bold')
        axes[idx].set_ylabel("IV Skew", fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


# ---------------- CLI ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global market bubble detector via IV skew inversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel worker threads"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display IV skew visualizations"
    )
    
    args = parser.parse_args()
    
    run_global_iv_skew_analysis(n_workers=args.workers, show_plot=args.plot)