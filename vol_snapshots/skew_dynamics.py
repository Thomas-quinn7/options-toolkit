"""Skew dynamics: does the smile steepen when the market falls?

Two classic empirical facts about equity vol surfaces, tested on the repo's
own daily fits:

* **Skew-return correlation**: index skew steepens in selloffs (crash
  protection bids up). Measured as the regression of daily changes in the
  fitted 30d fixed-moneyness skew - ``sigma(-0.05) - sigma(+0.05)``,
  reconstructed from each day's SSVI fit - on the MARKET return (SPY's log
  spot change). Expected beta < 0: market down, skew up.
* **The leverage effect**: a name's own ATM vol rises when its price falls.
  Measured as the regression of daily 30d ATM vol changes on the ticker's OWN
  return. Expected beta < 0.

Betas are reported in vol points per 1% move with 2-s.e. bars. Stated limits:
daily fits give few observations per week, so the study gates itself on
``MIN_PAIRS`` matched day-pairs per ticker and stays silent (logging what it
is waiting for) until then; near-zero-move days carry no signal and are
excluded; SPY's skew regression is on its own return (that is the classic
index-skew fact, not a bug).

    python vol_snapshots/skew_dynamics.py

Pinned offline by ``tests/test_skew_dynamics.py`` on histories with KNOWN
injected dynamics (skew wired to market level, vol wired to own price).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# Keep `python vol_snapshots/skew_dynamics.py` (and the scheduled task)
# working from a plain clone: repo root on the path.
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vol_snapshots.fit_history import HISTORY_CSV, load_history  # noqa: E402
from vol_snapshots.sticky import smile_from_row  # noqa: E402

SUMMARY_CSV = os.path.join(_HERE, "skew_dynamics.csv")
CHART_PATH = os.path.join(_REPO, "charts", "vol_surface", "skew_dynamics.png")

MARKET = "SPY"
DK = 0.05                # fixed-moneyness wing for the skew measure
MIN_MOVE = 0.001         # a pair needs at least a 0.1% move to carry signal
MAX_GAP_DAYS = 4
MIN_PAIRS = 15


def skew_measure(row) -> float:
    """30d fixed-moneyness skew sigma(-DK) - sigma(+DK) from one fitted row."""
    sig = smile_from_row(row)
    if sig is None:
        return float("nan")
    return float(sig(-DK) - sig(DK))


def ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Slope and its standard error, intercept included."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    beta = float(np.sum(xc * yc) / np.sum(xc**2))
    resid = yc - beta * xc
    dof = max(len(x) - 2, 1)
    se = float(np.sqrt(np.sum(resid**2) / dof / np.sum(xc**2)))
    return beta, se


def day_pairs(hist_t: pd.DataFrame):
    """(prev_row, next_row) for consecutive fits within MAX_GAP_DAYS."""
    hist_t = hist_t.sort_values("snapshot_date").reset_index(drop=True)
    dates = pd.to_datetime(hist_t["snapshot_date"])
    for i in range(len(hist_t) - 1):
        if 1 <= (dates.iloc[i + 1] - dates.iloc[i]).days <= MAX_GAP_DAYS:
            yield hist_t.iloc[i], hist_t.iloc[i + 1]


@dataclass
class TickerDynamics:
    ticker: str
    n_pairs: int
    beta_skew: float          # d(skew)/d(market log return)
    se_skew: float
    beta_lev: float           # d(ATM vol)/d(own log return)
    se_lev: float


def run_study(hist: pd.DataFrame) -> Tuple[List[TickerDynamics], List[str]]:
    # Market returns keyed by the (from, to) date pair they span.
    spy = hist[hist["ticker"] == MARKET]
    mkt_ret: Dict[Tuple[str, str], float] = {}
    for a, b in day_pairs(spy):
        mkt_ret[(str(a["snapshot_date"]), str(b["snapshot_date"]))] = float(
            np.log(b["spot"] / a["spot"]))

    studies, waiting = [], []
    for ticker, hist_t in hist.groupby("ticker"):
        d_skew, r_mkt, d_atm, r_own = [], [], [], []
        for a, b in day_pairs(hist_t):
            key = (str(a["snapshot_date"]), str(b["snapshot_date"]))
            own = float(np.log(b["spot"] / a["spot"]))
            if abs(own) >= MIN_MOVE and np.isfinite(a["atm_vol_30d"]) \
                    and np.isfinite(b["atm_vol_30d"]):
                d_atm.append(float(b["atm_vol_30d"] - a["atm_vol_30d"]))
                r_own.append(own)
            m = mkt_ret.get(key)
            if m is None or abs(m) < MIN_MOVE:
                continue
            sk_a, sk_b = skew_measure(a), skew_measure(b)
            if np.isfinite(sk_a) and np.isfinite(sk_b):
                d_skew.append(sk_b - sk_a)
                r_mkt.append(m)
        if len(d_skew) < MIN_PAIRS or len(d_atm) < MIN_PAIRS:
            waiting.append(f"{ticker}: {min(len(d_skew), len(d_atm))}/"
                           f"{MIN_PAIRS} matched pairs")
            continue
        b_sk, se_sk = ols(np.array(r_mkt), np.array(d_skew))
        b_lv, se_lv = ols(np.array(r_own), np.array(d_atm))
        studies.append(TickerDynamics(
            ticker=str(ticker), n_pairs=len(d_skew),
            beta_skew=b_sk, se_skew=se_sk, beta_lev=b_lv, se_lev=se_lv))
    return studies, waiting


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
def plot_dynamics(studies: List[TickerDynamics],
                  out: str = CHART_PATH) -> Optional[str]:
    """Two errorbar panels: skew-vs-market beta and the leverage-effect beta,
    per ticker, both expected below zero."""
    if not studies:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    import plotstyle as ps
    ps.apply_style()

    studies = sorted(studies, key=lambda s: s.ticker)
    names = [s.ticker for s in studies]
    xs = np.arange(len(studies))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, beta, se, title in (
        (axL, [s.beta_skew for s in studies], [s.se_skew for s in studies],
         "Skew steepening: d(30d skew) / d(market return)\n(< 0: selloffs bid up crash protection)"),
        (axR, [s.beta_lev for s in studies], [s.se_lev for s in studies],
         "Leverage effect: d(30d ATM vol) / d(own return)\n(< 0: vol rises as price falls)"),
    ):
        ax.errorbar(xs, beta, yerr=[2 * s for s in se], fmt="o",
                    color=ps.SERIES[0], ecolor=ps.MUTED, capsize=4)
        ax.axhline(0, color=ps.BASELINE, lw=1.2)
        ax.set_xticks(xs, names)
        ax.set_title(title)
        ax.set_ylabel("vol pts per 1% move")

    fig.suptitle("Skew and vol dynamics on the fitted daily surfaces "
                 "(2 s.e. bars)", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", default=HISTORY_CSV)
    ap.add_argument("--summary", default=SUMMARY_CSV)
    ap.add_argument("--chart", default=CHART_PATH)
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args(argv)

    hist = load_history(args.history)
    if hist.empty:
        print("no surface history yet - run vol_snapshots/fit_history.py first")
        return
    studies, waiting = run_study(hist)
    for line in waiting:
        print(f"  accumulating  {line}")
    if not studies:
        print(f"insufficient data: need {MIN_PAIRS} matched day-pairs "
              f"(|move| >= {MIN_MOVE:.1%}, {MARKET} fitted the same days) - "
              f"the scheduled run will produce this automatically.")
        return

    print(f"{'ticker':<7} {'pairs':>5} {'beta_skew':>10} {'2se':>6} "
          f"{'beta_lev':>9} {'2se':>6}")
    rows = []
    for s in studies:
        print(f"{s.ticker:<7} {s.n_pairs:>5} {s.beta_skew:>10.2f} "
              f"{2 * s.se_skew:>6.2f} {s.beta_lev:>9.2f} {2 * s.se_lev:>6.2f}")
        rows.append({k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in s.__dict__.items()})
    pd.DataFrame(rows).to_csv(args.summary, index=False)
    print(f"summary -> {args.summary}")
    if not args.no_chart:
        out = plot_dynamics(studies, args.chart)
        if out:
            print(f"chart -> {out}")


if __name__ == "__main__":
    main()
