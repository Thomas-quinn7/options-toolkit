"""Daily option-chain snapshot capture.

Free data sources have no history: yfinance serves TODAY's option chains and
nothing else. Every study this repo wants to do next on surface DYNAMICS -
how fitted SSVI parameters move day to day (eSSVI once per-expiry skew
lands), sticky-strike vs sticky-delta,
term-structure signals, event vol - needs a time series of chains that no
free source provides. The only way to have one is to have been capturing it,
so this script makes each day's chains a one-command habit:

    python vol_snapshots/capture.py                 # default ticker list
    python vol_snapshots/capture.py SPY QQQ TSLA    # explicit tickers

Each run writes one tidy csv.gz per ticker under
``vol_snapshots/data/<YYYY-MM-DD>/`` and is idempotent per day (already
captured tickers are skipped, so a cron/Task-Scheduler double-fire is
harmless). Captured columns are exactly what the repo's fitters need later:
bid/ask (band calibration fits the quoted interval, not a mid), strike,
expiry, spot at capture, volume/open-interest (liquidity gates), and
yfinance's own IV field kept only for cross-checking - the repo inverts its
own IVs from prices.

For the scheduled daily run use ``daily_capture.ps1`` (captures, logs to
``capture.log``, and auto-commits the new data) - registration one-liner in
this folder's README.

``normalize_chain`` (pure) and ``load_snapshots`` are importable and tested
offline; only ``capture_ticker``/``main`` touch the network.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
from typing import Iterable, Optional

import pandas as pd

# Universe: index ETFs + mega-cap singles (the original core), plus the
# macro complex - sector rotation (XLF/XLE/XLK), rates (TLT), gold (GLD) and
# the vol complex itself (VXX) - added 2026-07-26. Data not captured now
# never exists later, and each ETF adds only ~50-100 KB/day compressed.
DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM",                      # index
    "AAPL", "MSFT", "NVDA", "TSLA",           # mega-cap singles (earnings vol)
    "XLF", "XLE", "XLK",                      # sector rotation
    "TLT", "GLD", "VXX",                      # rates, gold, vol complex
]

COLUMNS = ["snapshot_date", "ticker", "spot", "riskfree", "expiry", "otype",
           "strike", "bid", "ask", "last", "volume", "open_interest", "iv_yf"]

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def normalize_chain(chain_df: pd.DataFrame, *, ticker: str, spot: float,
                    expiry: str, otype: str, snapshot_date: str,
                    riskfree: float = float("nan")) -> pd.DataFrame:
    """One yfinance chain side (calls or puts) -> tidy rows in the snapshot schema.

    Pure function of its inputs so the schema is testable offline. Rows with
    no two-sided quote (bid or ask missing/zero) are kept - the liquidity
    decision belongs to the consumer, the capture's job is fidelity - but the
    schema is enforced: exactly COLUMNS, one row per (strike, otype).
    """
    out = pd.DataFrame({
        "snapshot_date": snapshot_date,
        "ticker": ticker,
        "spot": float(spot),
        "riskfree": riskfree,
        "expiry": expiry,
        "otype": otype,
        "strike": pd.to_numeric(chain_df["strike"], errors="coerce"),
        "bid": pd.to_numeric(chain_df.get("bid"), errors="coerce"),
        "ask": pd.to_numeric(chain_df.get("ask"), errors="coerce"),
        "last": pd.to_numeric(chain_df.get("lastPrice"), errors="coerce"),
        "volume": pd.to_numeric(chain_df.get("volume"), errors="coerce"),
        "open_interest": pd.to_numeric(chain_df.get("openInterest"), errors="coerce"),
        "iv_yf": pd.to_numeric(chain_df.get("impliedVolatility"), errors="coerce"),
    })
    out = out.dropna(subset=["strike"]).sort_values("strike")
    return out[COLUMNS].reset_index(drop=True)


def _snapshot_path(root: str, snapshot_date: str, ticker: str) -> str:
    return os.path.join(root, snapshot_date, f"{ticker.upper()}.csv.gz")


def save_snapshot(df: pd.DataFrame, root: str = DATA_ROOT) -> str:
    """Write one ticker's normalized rows for one day; returns the path."""
    if list(df.columns) != COLUMNS:
        raise ValueError(f"snapshot frame must have columns {COLUMNS}")
    date = str(df["snapshot_date"].iloc[0])
    ticker = str(df["ticker"].iloc[0])
    path = _snapshot_path(root, date, ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return path


def load_snapshots(root: str = DATA_ROOT, tickers: Optional[Iterable[str]] = None
                   ) -> pd.DataFrame:
    """Load every captured day into one tidy DataFrame (empty if none yet)."""
    frames = []
    want = {t.upper() for t in tickers} if tickers is not None else None
    if os.path.isdir(root):
        for day in sorted(os.listdir(root)):
            day_dir = os.path.join(root, day)
            if not os.path.isdir(day_dir):
                continue
            for fn in sorted(os.listdir(day_dir)):
                if not fn.endswith(".csv.gz"):
                    continue
                if want is not None and fn[:-len(".csv.gz")].upper() not in want:
                    continue
                frames.append(pd.read_csv(os.path.join(day_dir, fn)))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out[COLUMNS]


# --------------------------------------------------------------------------- #
# Network side (yfinance) - kept thin so everything above stays offline-testable
# --------------------------------------------------------------------------- #
def _fetch_riskfree() -> float:
    """13-week T-bill yield (^IRX, quoted in %) as a decimal; NaN if unavailable."""
    try:
        import yfinance as yf
        h = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]) / 100.0
    except Exception:
        return float("nan")


def capture_ticker(ticker: str, snapshot_date: str, riskfree: float,
                   root: str = DATA_ROOT) -> Optional[str]:
    """Fetch and store all expiries for one ticker; returns the path written,
    or None if today's file already exists or the fetch failed entirely."""
    path = _snapshot_path(root, snapshot_date, ticker)
    if os.path.exists(path):
        print(f"  {ticker}: already captured today - skipped")
        return None
    import yfinance as yf
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")["Close"].dropna()
    if hist.empty:
        print(f"  {ticker}: no spot price - skipped")
        return None
    spot = float(hist.iloc[-1])
    parts = []
    for expiry in tk.options:
        try:
            chain = tk.option_chain(expiry)
        except Exception as e:  # one bad expiry should not kill the capture
            print(f"  {ticker} {expiry}: fetch failed ({e}) - skipped")
            continue
        for otype, side in (("call", chain.calls), ("put", chain.puts)):
            if side is not None and len(side):
                parts.append(normalize_chain(
                    side, ticker=ticker.upper(), spot=spot, expiry=expiry,
                    otype=otype, snapshot_date=snapshot_date, riskfree=riskfree))
    if not parts:
        print(f"  {ticker}: no chains returned - skipped")
        return None
    df = pd.concat(parts, ignore_index=True)
    out = save_snapshot(df, root)
    n_exp = df["expiry"].nunique()
    print(f"  {ticker}: {len(df)} quotes across {n_exp} expiries -> {out}")
    return out


def main(argv: Optional[list] = None) -> None:
    args = list(argv) if argv else []
    force = "--force" in args
    tickers = [a for a in args if a != "--force"] or DEFAULT_TICKERS
    # Weekend captures serve Friday's after-close quotes: junk mids that once
    # implied a -37% AAPL dividend yield downstream. Refuse unless forced.
    if _dt.date.today().weekday() >= 5 and not force:
        print("Market closed (weekend) - today's chains are stale after-close "
              "quotes. Skipping capture; pass --force to capture anyway.")
        return
    snapshot_date = _dt.date.today().isoformat()
    riskfree = _fetch_riskfree()
    print(f"Capturing option chains for {snapshot_date} "
          f"(riskfree ^IRX = {riskfree if riskfree == riskfree else 'n/a'})")
    for t in tickers:
        try:
            capture_ticker(t, snapshot_date, riskfree)
        except Exception as e:  # a dead ticker should not kill the batch
            print(f"  {t}: capture failed ({e})")


if __name__ == "__main__":
    main(sys.argv[1:])
