"""Fit the arbitrage-free SSVI band surface to each captured day's chains and
track the fitted parameters over time.

This is what the daily snapshots exist FOR: every captured (day, ticker) chain
is turned into one row of ``surface_history.csv`` - the fitted SSVI
``(rho, eta, gamma)``, the ATM vol term structure read off the fitted surface,
and the surface-level no-arbitrage diagnostics - so surface *dynamics* (skew
moves, term-structure shifts, event vol) accumulate as a small tidy time
series while the raw chains stay in the data store.

The pipeline is the repo's own, end to end:

1. **Forward from put-call parity.** Per expiry, ``C_mid - P_mid =
   e^{-rT}(F - K)`` implied at the strikes nearest spot (median). No borrow or
   dividend assumption is imported from anywhere - the option market itself
   prices the forward, and the implied dividend yield follows as
   ``q = r - ln(F/S)/T``.
2. **IVs inverted from prices** with ``vol_surface.iv_from_price`` - bid and
   ask separately, so every quote becomes a total-variance *interval*.
   yfinance's IV field is never used.
3. **OTM quotes only** (puts below the forward, calls above), the standard
   surface-building convention: the OTM side is the liquid one and carries no
   early-exercise premium to speak of.
4. **Global SSVI band fit** (``fit_ssvi_band``): the surface must thread every
   bid-ask interval, tight ATM bands pinning it, wide wings constraining it
   only loosely - plus the Gatheral-Jacquier butterfly penalty.
5. **No-arb diagnostics recorded, not assumed**: min Durrleman g and the
   calendar gap of the fitted surface go into the history row, so a day the
   fit could NOT be made arbitrage-free is visible in the data.

Run after a capture (the scheduled task does):

    python vol_snapshots/fit_history.py             # fit all unfitted days
    python vol_snapshots/fit_history.py --refit     # refit everything
    python vol_snapshots/fit_history.py --no-chart  # skip the chart

Everything below the loaders is pure and covered offline by
``tests/test_fit_history.py`` on a synthetic day generated from a known SSVI
surface.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# Keep `python vol_snapshots/fit_history.py` (and the scheduled task) working
# from a plain clone: repo root on the path so package-absolute imports
# resolve without `pip install -e .`.
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vol_snapshots.capture import DATA_ROOT, load_snapshots  # noqa: E402
from pricing_and_vol_surface.vol_surface import (  # noqa: E402
    SSVIParams,
    calendar_min_gap,
    fit_ssvi_band,
    iv_from_price,
    min_butterfly_g_ssvi,
    ssvi_w,
)

HISTORY_CSV = os.path.join(_HERE, "surface_history.csv")
CHART_PATH = os.path.join(_REPO, "charts", "vol_surface", "surface_history.png")

DEFAULT_RISK_FREE = 0.04     # used when a snapshot's ^IRX fetch failed (NaN)

# Slice/quote gates. The capture keeps everything; the fit is where liquidity
# decisions belong (vol_snapshots/README.md).
MAX_SPREAD_FRAC = 0.35       # drop quotes with (ask-bid)/mid above this
MIN_T = 10 / 365.0           # sub-2-week expiries: pin risk, stale wings
MAX_T = 1.6                  # beyond ~19 months quotes are too sparse
K_ABS_MAX = 0.6              # log-moneyness window the fit is asked to explain
MIN_QUOTES_PER_SLICE = 8
MIN_SLICES = 3
IV_MIN, IV_MAX = 0.03, 2.0
Q_ABS_MAX = 0.08             # parity-implied |div/borrow| above this means the
                             # quotes are stale, not that the carry is real

# The history CSV records every captured ticker, but the chart caps at 8
# series - the palette has 8 fixed slots (plotstyle) and more lines per panel
# stop being readable. Priority order below; remaining slots fill from
# whatever else is in the history, so a narrower capture list charts fully.
CHART_PRIORITY = ["SPY", "QQQ", "IWM", "VXX", "TLT", "AAPL", "NVDA", "TSLA"]
MAX_CHART_SERIES = 8

HISTORY_COLUMNS = [
    "snapshot_date", "ticker", "spot", "riskfree", "n_expiries", "n_quotes",
    "rho", "eta", "gamma", "atm_vol_30d", "atm_vol_91d", "atm_vol_182d",
    "term_slope", "min_butterfly_g", "calendar_min_gap",
]


# --------------------------------------------------------------------------- #
# Forward and slice construction (pure)                                       #
# --------------------------------------------------------------------------- #
def implied_forward(slice_df: pd.DataFrame, r: float, T: float,
                    spot: float, n_pairs: int = 5) -> float:
    """Forward implied by put-call parity: C - P = e^{-rT}(F - K).

    Uses mid prices at the ``n_pairs`` strikes nearest spot where BOTH the
    call and the put have a two-sided quote; returns the median implied F,
    or nan if no usable pair exists.
    """
    two_sided = slice_df[(slice_df["bid"] > 0) & (slice_df["ask"] >= slice_df["bid"])]
    calls = two_sided[two_sided["otype"] == "call"].set_index("strike")
    puts = two_sided[two_sided["otype"] == "put"].set_index("strike")
    strikes = calls.index.intersection(puts.index)
    if len(strikes) == 0:
        return float("nan")
    strikes = strikes.sort_values()
    nearest = strikes[np.argsort(np.abs(strikes.values - spot))[:n_pairs]]
    c_mid = 0.5 * (calls.loc[nearest, "bid"] + calls.loc[nearest, "ask"])
    p_mid = 0.5 * (puts.loc[nearest, "bid"] + puts.loc[nearest, "ask"])
    fwd = nearest.values + np.exp(r * T) * (c_mid.values - p_mid.values)
    return float(np.median(fwd))


def build_slice(slice_df: pd.DataFrame, snapshot_date: str, expiry: str,
                spot: float, r: float) -> Optional[Dict]:
    """One expiry's quotes -> {k, w_bid, w_ask, T, theta} or None if unusable.

    OTM only (puts below the implied forward, calls above), IVs inverted from
    bid and ask separately so each quote is a total-variance interval.
    """
    T = ( _dt.date.fromisoformat(str(expiry)[:10])
          - _dt.date.fromisoformat(str(snapshot_date)[:10]) ).days / 365.0
    if not (MIN_T <= T <= MAX_T):
        return None

    F = implied_forward(slice_df, r, T, spot)
    if not np.isfinite(F) or F <= 0:
        return None
    # Dividend/borrow implied by the forward itself; the inverter then prices
    # with an (S, q) pair exactly consistent with F.
    q = r - np.log(F / spot) / T
    if abs(q) > Q_ABS_MAX:
        return None

    otm = slice_df[
        (slice_df["bid"] > 0) & (slice_df["ask"] >= slice_df["bid"])
        & (((slice_df["otype"] == "put") & (slice_df["strike"] <= F))
           | ((slice_df["otype"] == "call") & (slice_df["strike"] >= F)))
    ].copy()
    mid = 0.5 * (otm["bid"] + otm["ask"])
    otm = otm[(otm["ask"] - otm["bid"]) <= MAX_SPREAD_FRAC * mid]
    if otm.empty:
        return None

    k, w_bid, w_ask = [], [], []
    for row in otm.itertuples(index=False):
        iv_b = iv_from_price(row.bid, spot, row.strike, T, r, q, otype=row.otype)
        iv_a = iv_from_price(row.ask, spot, row.strike, T, r, q, otype=row.otype)
        if not (np.isfinite(iv_b) and np.isfinite(iv_a)):
            continue
        if not (IV_MIN <= iv_b <= IV_MAX and IV_MIN <= iv_a <= IV_MAX):
            continue
        kk = np.log(row.strike / F)
        if abs(kk) > K_ABS_MAX:
            continue
        k.append(kk)
        w_bid.append(iv_b**2 * T)
        w_ask.append(iv_a**2 * T)

    if len(k) < MIN_QUOTES_PER_SLICE:
        return None
    order = np.argsort(k)
    k = np.asarray(k)[order]
    w_bid = np.asarray(w_bid)[order]
    w_ask = np.asarray(w_ask)[order]
    if not (k[0] < 0.0 < k[-1]):       # theta needs an ATM straddle-able slice
        return None
    w_mid = 0.5 * (w_bid + w_ask)
    theta = float(np.interp(0.0, k, w_mid))
    return {"k": k, "w_bid": w_bid, "w_ask": w_ask, "T": T, "theta": theta}


def fit_day_ticker(df: pd.DataFrame) -> Optional[Dict]:
    """All of one (day, ticker)'s quotes -> one surface_history row, or None.

    Slices are fitted jointly with the global SSVI band fit; the row records
    the parameters, the fitted ATM term structure, and the surface-level
    no-arbitrage diagnostics.
    """
    snapshot_date = str(df["snapshot_date"].iloc[0])
    ticker = str(df["ticker"].iloc[0])
    spot = float(df["spot"].iloc[0])
    r = float(df["riskfree"].iloc[0])
    if not np.isfinite(r):
        r = DEFAULT_RISK_FREE

    slices = []
    for expiry, sl in df.groupby("expiry"):
        built = build_slice(sl, snapshot_date, expiry, spot, r)
        if built is not None:
            slices.append(built)
    if len(slices) < MIN_SLICES:
        return None
    slices.sort(key=lambda s: s["T"])

    Ts = np.array([s["T"] for s in slices])
    thetas = np.array([s["theta"] for s in slices])
    p = fit_ssvi_band([s["k"] for s in slices],
                      [s["w_bid"] for s in slices],
                      [s["w_ask"] for s in slices], thetas)

    k_grid = np.linspace(-K_ABS_MAX, K_ABS_MAX, 241)
    min_g = min_butterfly_g_ssvi(thetas, p, k_grid)
    cal_gap = calendar_min_gap(thetas, p, k_grid)      # thetas in T order

    def atm_vol(T_target: float) -> float:
        if not (Ts[0] <= T_target <= Ts[-1]):
            return float("nan")
        th = float(np.interp(T_target, Ts, thetas))
        return float(np.sqrt(ssvi_w(0.0, th, p) / T_target))

    v30, v91, v182 = atm_vol(30 / 365), atm_vol(91 / 365), atm_vol(182 / 365)
    return {
        "snapshot_date": snapshot_date,
        "ticker": ticker,
        "spot": round(spot, 4),
        "riskfree": round(r, 6),
        "n_expiries": len(slices),
        "n_quotes": int(sum(len(s["k"]) for s in slices)),
        "rho": round(float(p.rho), 6),
        "eta": round(float(p.eta), 6),
        "gamma": round(float(p.gamma), 6),
        "atm_vol_30d": round(v30, 6),
        "atm_vol_91d": round(v91, 6),
        "atm_vol_182d": round(v182, 6),
        "term_slope": round(v182 - v30, 6),
        "min_butterfly_g": round(float(min_g), 6),
        "calendar_min_gap": round(float(cal_gap), 8),
    }


# --------------------------------------------------------------------------- #
# History bookkeeping                                                          #
# --------------------------------------------------------------------------- #
def load_history(path: str = HISTORY_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"snapshot_date": str, "ticker": str})
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def update_history(rows: List[Dict], path: str = HISTORY_CSV) -> pd.DataFrame:
    """Upsert rows keyed on (snapshot_date, ticker); returns the full history."""
    hist = load_history(path)
    new = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if len(new):
        if len(hist):
            keys = set(zip(new["snapshot_date"], new["ticker"]))
            keep = [(d, t) not in keys
                    for d, t in zip(hist["snapshot_date"], hist["ticker"])]
            hist = pd.concat([hist[keep], new], ignore_index=True)
        else:
            hist = new
        hist = hist.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)
        hist.to_csv(path, index=False)
    return hist


# --------------------------------------------------------------------------- #
# Chart                                                                        #
# --------------------------------------------------------------------------- #
def chart_tickers(available) -> List[str]:
    """The <= MAX_CHART_SERIES tickers the chart shows: CHART_PRIORITY order
    first, then whatever else the history holds, alphabetically."""
    available = set(available)
    ordered = [t for t in CHART_PRIORITY if t in available]
    ordered += [t for t in sorted(available) if t not in CHART_PRIORITY]
    return ordered[:MAX_CHART_SERIES]


def plot_history(hist: pd.DataFrame, out: str = CHART_PATH) -> Optional[str]:
    """Four panels: ATM 30d vol, skew rho, and term slope over time, plus the
    latest day's fitted ATM term structure. Single-day histories draw as
    markers; lines emerge as days accumulate. Shows the chart_tickers()
    subset; the full universe lives in the history CSV."""
    if hist.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    from plotstyle import apply_style, series_color

    apply_style()
    tickers = chart_tickers(hist["ticker"].unique())
    hist = hist[hist["ticker"].isin(tickers)]
    dates = pd.to_datetime(hist["snapshot_date"])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    pad = max((dates.max() - dates.min()) * 0.05, pd.Timedelta(days=2))

    def per_ticker(ax, col, title, ylabel, pct=False):
        for i, t in enumerate(tickers):
            sub = hist[hist["ticker"] == t]
            y = sub[col] * (100 if pct else 1)
            ax.plot(pd.to_datetime(sub["snapshot_date"]), y, marker="o",
                    markersize=4, color=series_color(i), label=t)
        ax.set_xlim(dates.min() - pad, dates.max() + pad)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    per_ticker(axes[0, 0], "atm_vol_30d", "ATM vol, 30d (fitted)", "vol (%)", pct=True)
    per_ticker(axes[0, 1], "rho", "SSVI rho (skew)", "rho")
    per_ticker(axes[1, 0], "term_slope", "Term slope: ATM 182d - 30d", "vol pts (%)", pct=True)

    ax = axes[1, 1]
    last_day = hist["snapshot_date"].max()
    latest = hist[hist["snapshot_date"] == last_day]
    horizons = [(30, "atm_vol_30d"), (91, "atm_vol_91d"), (182, "atm_vol_182d")]
    for i, t in enumerate(tickers):
        sub = latest[latest["ticker"] == t]
        if sub.empty:
            continue
        xs = [d for d, c in horizons if np.isfinite(sub[c].iloc[0])]
        ys = [100 * sub[c].iloc[0] for _, c in horizons if np.isfinite(sub[c].iloc[0])]
        ax.plot(xs, ys, marker="o", markersize=4, color=series_color(i), label=t)
    ax.set_title(f"ATM term structure, {last_day}")
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("vol (%)")

    axes[0, 0].legend(ncol=2)
    for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
        a.tick_params(axis="x", rotation=30)
    fig.suptitle("Fitted SSVI surface history (band fit, own IVs, arb-checked daily)",
                 fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--history", default=HISTORY_CSV)
    ap.add_argument("--chart", default=CHART_PATH)
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--refit", action="store_true",
                    help="refit every (day, ticker), not just unfitted ones")
    args = ap.parse_args(argv)

    snaps = load_snapshots(args.data_root)
    if snaps.empty:
        print("no snapshot data found - run vol_snapshots/capture.py first")
        return
    hist = load_history(args.history)
    done = set(zip(hist["snapshot_date"], hist["ticker"])) if not args.refit else set()

    rows, skipped = [], []
    for (day, ticker), df in snaps.groupby(["snapshot_date", "ticker"]):
        if (str(day), str(ticker)) in done:
            continue
        row = fit_day_ticker(df)
        if row is None:
            skipped.append((day, ticker))
            continue
        rows.append(row)
        print(f"  {day} {ticker:<5} rho={row['rho']:+.3f}  "
              f"atm30={row['atm_vol_30d']:.4f}  slope={row['term_slope']:+.4f}  "
              f"min_g={row['min_butterfly_g']:+.4f}  "
              f"cal_gap={row['calendar_min_gap']:+.6f}  "
              f"({row['n_quotes']} quotes / {row['n_expiries']} expiries)")
    for day, ticker in skipped:
        print(f"  {day} {ticker}: not enough usable slices - skipped")

    if rows:
        hist = update_history(rows, args.history)
        print(f"history: {len(hist)} rows -> {args.history}")
    else:
        print("nothing new to fit")
    if not args.no_chart:
        out = plot_history(hist, args.chart)
        if out:
            print(f"chart -> {out}")


if __name__ == "__main__":
    main()
