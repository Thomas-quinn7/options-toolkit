"""Realised-vs-implied replay: delta-hedged short straddles on captured chains.

The market-making simulator's headline result - a delta-hedged short option's
P&L tracks realised-minus-implied vol through the gamma-P&L identity - was
demonstrated on synthetic paths and validated against the closed form. This
module runs the SAME hedging conventions through the repo's real snapshot
history, one position at a time:

* On the first usable snapshot day (and again each time a position closes),
  sell one ATM straddle per ticker at the quoted MID, targeting 25-45 DTE.
  Entry implied vol is inverted from the traded prices; the forward (and so
  the implied dividend/borrow) comes from put-call parity at entry.
* Every snapshot day after: delta-hedge at the ENTRY implied vol (hedge at
  the vol you marked at - the identity's convention, same as mm_sim), accrue
  financing on cash at the entry T-bill rate and the dividend yield on the
  hedge book.
* Alongside the cash P&L, accrue the discrete gamma-P&L theory term
  0.5*Gamma*S^2*((dlnS)^2 - sigma_entry^2 dt), future-valued to expiry - so
  every position carries its own sim-vs-theory comparison, exactly like
  Experiment A but on real prices.
* At expiry, settle at intrinsic and liquidate the hedge. A position the data
  has not yet carried to expiry is marked at the current quoted mid (or
  Black-Scholes at entry vol when the strike has no usable quote - flagged in
  the output, never silent).

Honest conventions, stated: entry and mark at MID (this measures the
realised-vs-implied effect, not execution - the entry half-spread is recorded
per position as the cost a real desk would pay); one snapshot per day means
daily discrete hedging, so per-position hedging error is material and only
shrinks by averaging across positions as the history grows; consecutive days
with an identical spot (weekend/holiday captures serving stale quotes) are
dropped from every path.

Stateless by design: each run recomputes every position from the raw data
store, writes ``replay_history.csv`` and the chart, and prints the table.

    python vol_snapshots/replay.py               # all tickers, default output
    python vol_snapshots/replay.py --tickers SPY QQQ

Covered offline by ``tests/test_replay.py`` on synthetic GBM histories with
known realised and implied vols.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# Keep `python vol_snapshots/replay.py` (and the scheduled task) working from
# a plain clone: repo root on the path so package imports resolve.
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from market_making.mm_sim import bs_delta, bs_gamma, bs_price  # noqa: E402
from vol_snapshots.capture import DATA_ROOT, load_snapshots  # noqa: E402
from vol_snapshots.fit_history import DEFAULT_RISK_FREE, implied_forward  # noqa: E402
from pricing_and_vol_surface.vol_surface import iv_from_price  # noqa: E402

HISTORY_CSV = os.path.join(_HERE, "replay_history.csv")
CHART_PATH = os.path.join(_REPO, "charts", "market_making",
                          "replay_realised_vs_implied.png")

DTE_LO, DTE_HI = 25, 45          # entry expiry window
DTE_TARGET = 35
MAX_SPREAD_FRAC = 0.25           # entry legs must be this liquid at the mid

HISTORY_COLUMNS = [
    "ticker", "entry_date", "expiry", "status", "strike", "entry_spot",
    "sigma_entry", "riskfree", "div_yield", "premium_mid", "half_spread_cost",
    "days_held", "realised_vol", "pnl", "theory_pnl", "mark_source",
]


def _date(s) -> _dt.date:
    return _dt.date.fromisoformat(str(s)[:10])


# --------------------------------------------------------------------------- #
# Entry selection (pure)                                                      #
# --------------------------------------------------------------------------- #
def pick_entry(day_df: pd.DataFrame, snapshot_date: str,
               r: float) -> Optional[Dict]:
    """Choose the short-straddle entry from one (day, ticker) chain.

    Expiry: the listed expiry inside [DTE_LO, DTE_HI] nearest DTE_TARGET.
    Strike: the listed strike nearest the parity-implied forward with liquid
    two-sided quotes on BOTH legs. Returns None when nothing qualifies.
    """
    spot = float(day_df["spot"].iloc[0])
    day = _date(snapshot_date)

    candidates = []
    for expiry in day_df["expiry"].unique():
        dte = (_date(expiry) - day).days
        if DTE_LO <= dte <= DTE_HI:
            candidates.append((abs(dte - DTE_TARGET), dte, str(expiry)))
    if not candidates:
        return None

    for _, dte, expiry in sorted(candidates):
        sl = day_df[day_df["expiry"] == expiry]
        T = dte / 365.0
        F = implied_forward(sl, r, T, spot)
        if not np.isfinite(F) or F <= 0:
            continue
        q = r - np.log(F / spot) / T

        two_sided = sl[(sl["bid"] > 0) & (sl["ask"] >= sl["bid"])]
        calls = two_sided[two_sided["otype"] == "call"].set_index("strike")
        puts = two_sided[two_sided["otype"] == "put"].set_index("strike")
        shared = calls.index.intersection(puts.index)
        if len(shared) == 0:
            continue

        for K in shared[np.argsort(np.abs(shared.values - F))]:
            legs = {}
            for otype, side in (("call", calls), ("put", puts)):
                bid, ask = float(side.loc[K, "bid"]), float(side.loc[K, "ask"])
                mid = 0.5 * (bid + ask)
                if mid <= 0 or (ask - bid) > MAX_SPREAD_FRAC * mid:
                    legs = None
                    break
                iv = iv_from_price(mid, spot, float(K), T, r, q, otype=otype)
                if not np.isfinite(iv):
                    legs = None
                    break
                legs[otype] = {"mid": mid, "half": 0.5 * (ask - bid), "iv": iv}
            if legs is None:
                continue
            return {
                "expiry": expiry,
                "strike": float(K),
                "T0": T,
                "spot": spot,
                "riskfree": r,
                "div_yield": float(q),
                "sigma_entry": 0.5 * (legs["call"]["iv"] + legs["put"]["iv"]),
                "premium_mid": legs["call"]["mid"] + legs["put"]["mid"],
                "half_spread_cost": legs["call"]["half"] + legs["put"]["half"],
            }
    return None


# --------------------------------------------------------------------------- #
# One position, hedged through the days the data actually has (pure)          #
# --------------------------------------------------------------------------- #
@dataclass
class ReplayResult:
    ticker: str
    entry_date: str
    expiry: str
    status: str                  # "closed" | "open"
    strike: float
    entry_spot: float
    sigma_entry: float
    riskfree: float
    div_yield: float
    premium_mid: float
    half_spread_cost: float
    days_held: int
    realised_vol: float          # annualised, over the held path
    pnl: float                   # closed: settled; open: marked-to-market
    theory_pnl: float            # gamma-P&L accrual on the same path
    mark_source: str             # "settled" | "market_mid" | "bs_entry_vol"
    dates: List[str] = field(default_factory=list)
    pnl_path: List[float] = field(default_factory=list)
    theory_path: List[float] = field(default_factory=list)


def _straddle(fn, S, K, tau, r, sigma, q):
    return (fn(S, K, tau, r, sigma, q, "call")
            + fn(S, K, tau, r, sigma, q, "put"))


def replay_position(days: List[Dict], entry: Dict, ticker: str,
                    entry_date: str) -> ReplayResult:
    """Hedge one short straddle through ``days`` (dicts with date/spot/df).

    ``days`` starts at the entry day and stops at expiry or at the end of the
    data, whichever comes first. Cash conventions mirror mm_sim exactly.
    """
    K, T0 = entry["strike"], entry["T0"]
    r, q, sig = entry["riskfree"], entry["div_yield"], entry["sigma_entry"]
    expiry_day = _date(entry["expiry"])

    S = entry["spot"]
    cash = entry["premium_mid"]                      # sold the straddle at mid
    delta = _straddle(bs_delta, S, K, T0, r, sig, q)
    H = delta                                        # hedge the short position
    cash -= H * S
    theory = 0.0
    log_moves = []

    dates, pnl_path, theory_path = [entry_date], [0.0], [0.0]
    tau = T0

    for prev, cur in zip(days[:-1], days[1:]):
        dt_step = (cur["date"] - prev["date"]).days / 365.0
        S_new = cur["spot"]
        move = np.log(S_new / S)
        log_moves.append((move, dt_step))

        gamma = 2.0 * bs_gamma(S, K, tau, r, sig, q)   # call and put gammas equal
        tau_new = max(T0 - (cur["date"] - days[0]["date"]).days / 365.0, 0.0)
        theory += (0.5 * (-1.0) * gamma * S**2 * (move**2 - sig**2 * dt_step)
                   * np.exp(r * (tau - 0.5 * dt_step)))

        cash = cash * np.exp(r * dt_step) + H * S * (np.exp(q * dt_step) - 1.0)
        S, tau = S_new, tau_new

        delta = _straddle(bs_delta, S, K, tau, r, sig, q)
        cash -= (delta - H) * S
        H = delta

        mark = _straddle(bs_price, S, K, tau, r, sig, q)
        dates.append(cur["date"].isoformat())
        pnl_path.append(cash + H * S - mark)
        theory_path.append(theory)

    last = days[-1]
    if last["date"] >= expiry_day:
        # settle: short straddle pays intrinsic, hedge liquidates
        intrinsic = max(S - K, 0.0) + max(K - S, 0.0)
        pnl = cash + H * S - intrinsic
        status, mark_source = "closed", "settled"
    else:
        mark, mark_source = _mark_straddle(last["df"], K, entry["expiry"])
        if mark is None:
            mark = _straddle(bs_price, S, K, tau, r, sig, q)
            mark_source = "bs_entry_vol"
        pnl = cash + H * S - mark
        status = "open"
    pnl_path[-1] = pnl

    if len(log_moves) >= 2:
        moves = np.array([m for m, _ in log_moves])
        dts = np.array([d for _, d in log_moves])
        realised = float(np.sqrt(np.sum(moves**2) / np.sum(dts)))
    else:
        realised = float("nan")

    return ReplayResult(
        ticker=ticker, entry_date=entry_date, expiry=entry["expiry"],
        status=status, strike=K, entry_spot=entry["spot"],
        sigma_entry=round(sig, 6), riskfree=round(r, 6),
        div_yield=round(q, 6), premium_mid=round(entry["premium_mid"], 4),
        half_spread_cost=round(entry["half_spread_cost"], 4),
        days_held=len(days) - 1, realised_vol=round(realised, 6),
        pnl=round(float(pnl), 6), theory_pnl=round(float(theory), 6),
        mark_source=mark_source, dates=dates,
        pnl_path=[round(float(x), 6) for x in pnl_path],
        theory_path=[round(float(x), 6) for x in theory_path],
    )


def _mark_straddle(day_df: pd.DataFrame, K: float, expiry: str):
    """Quoted mid for the K-straddle of THIS expiry, or (None, ...) if unusable."""
    sl = day_df[day_df["expiry"].astype(str) == str(expiry)]
    total = 0.0
    for otype in ("call", "put"):
        leg = sl[(sl["otype"] == otype)
                 & (np.isclose(sl["strike"].astype(float), K))]
        if leg.empty:
            return None, None
        bid, ask = float(leg["bid"].iloc[0]), float(leg["ask"].iloc[0])
        if not (bid > 0 and ask >= bid):
            return None, None
        total += 0.5 * (bid + ask)
    return total, "market_mid"


# --------------------------------------------------------------------------- #
# Whole-history driver                                                        #
# --------------------------------------------------------------------------- #
def ticker_days(df_ticker: pd.DataFrame) -> List[Dict]:
    """Snapshot days for one ticker, oldest first, stale duplicates dropped.

    A weekend/holiday capture serves the previous session's quotes; its spot
    is byte-identical to the prior day's, so those days carry no information
    and would dilute realised vol with fake zero-move days.
    """
    out = []
    for day, df in sorted(df_ticker.groupby("snapshot_date"),
                          key=lambda kv: str(kv[0])):
        spot = float(df["spot"].iloc[0])
        if out and spot == out[-1]["spot"]:
            continue
        out.append({"date": _date(day), "spot": spot, "df": df})
    return out


def run_replay(snaps: pd.DataFrame, tickers=None) -> List[ReplayResult]:
    """Sequential non-overlapping short straddles per ticker over the history."""
    results: List[ReplayResult] = []
    for ticker, df_t in snaps.groupby("ticker"):
        if tickers and ticker not in tickers:
            continue
        days = ticker_days(df_t)
        i = 0
        while i < len(days):
            day = days[i]
            r = float(day["df"]["riskfree"].iloc[0])
            if not np.isfinite(r):
                r = DEFAULT_RISK_FREE
            entry = pick_entry(day["df"], day["date"].isoformat(), r)
            if entry is None:
                i += 1
                continue
            expiry_day = _date(entry["expiry"])
            span = [d for d in days[i:] if d["date"] <= expiry_day]
            results.append(replay_position(span, entry, str(ticker),
                                           day["date"].isoformat()))
            i += len(span)      # next position starts after this one's data
    return results


def write_history(results: List[ReplayResult], path: str = HISTORY_CSV) -> pd.DataFrame:
    rows = [{c: getattr(x, c) for c in HISTORY_COLUMNS} for x in results]
    out = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    out.to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------- #
# Chart                                                                        #
# --------------------------------------------------------------------------- #
def plot_replay(results: List[ReplayResult], out: str = CHART_PATH) -> Optional[str]:
    """Left: per-position hedged P&L paths (sim solid, theory dashed).
    Right: final P&L vs realised-minus-entry vol - the identity's signature,
    accumulating one point per position as the history grows."""
    if not results:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    import plotstyle as ps
    ps.apply_style()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    tickers = sorted({x.ticker for x in results})
    color = {t: ps.series_color(i % 8) for i, t in enumerate(tickers)}

    for x in results:
        d = pd.to_datetime(x.dates)
        axL.plot(d, x.pnl_path, color=color[x.ticker], label=x.ticker)
        axL.plot(d, x.theory_path, color=color[x.ticker], ls="--", alpha=0.6)
    axL.axhline(0, color=ps.BASELINE, lw=0.8)
    axL.set_title("Hedged short-straddle P&L (solid) vs gamma-P&L theory (dashed)")
    axL.set_ylabel("P&L per straddle ($)")
    axL.tick_params(axis="x", rotation=30)
    handles, labels = axL.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axL.legend(seen.values(), seen.keys(), ncol=2)

    for x in results:
        if np.isfinite(x.realised_vol):
            gap = 100 * (x.realised_vol - x.sigma_entry)
            marker = "o" if x.status == "closed" else "^"
            axR.scatter(gap, x.pnl, color=color[x.ticker], marker=marker, s=45)
    axR.axhline(0, color=ps.BASELINE, lw=0.8)
    axR.axvline(0, color=ps.BASELINE, lw=0.8)
    axR.set_title("Final P&L vs realised - entry implied vol\n"
                  "(o closed, ^ still open/marked)")
    axR.set_xlabel("realised - implied (vol pts)")
    axR.set_ylabel("P&L per straddle ($)")

    fig.suptitle("Realised-vs-implied replay on captured chains "
                 "(short ATM straddles, hedged daily at entry IV)",
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
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--history", default=HISTORY_CSV)
    ap.add_argument("--chart", default=CHART_PATH)
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args(argv)

    snaps = load_snapshots(args.data_root)
    if snaps.empty:
        print("no snapshot data found - run vol_snapshots/capture.py first")
        return
    results = run_replay(snaps, tickers=set(args.tickers) if args.tickers else None)
    if not results:
        print("no entries qualified (need liquid ATM straddles 25-45 DTE)")
        return

    write_history(results, args.history)
    print(f"{'ticker':<6} {'entry':<11} {'status':<7} {'held':>4} "
          f"{'iv_entry':>8} {'realised':>8} {'pnl':>9} {'theory':>9}  mark")
    for x in results:
        rv = f"{x.realised_vol:.3f}" if np.isfinite(x.realised_vol) else "  n/a"
        print(f"{x.ticker:<6} {x.entry_date:<11} {x.status:<7} {x.days_held:>4} "
              f"{x.sigma_entry:>8.3f} {rv:>8} {x.pnl:>9.3f} "
              f"{x.theory_pnl:>9.3f}  {x.mark_source}")
    print(f"history: {len(results)} positions -> {args.history}")
    if not args.no_chart:
        out = plot_replay(results, args.chart)
        if out:
            print(f"chart -> {out}")


if __name__ == "__main__":
    main()
