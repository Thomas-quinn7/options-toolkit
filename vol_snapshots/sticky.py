"""Sticky-strike vs sticky-delta: how the fitted smile moves when spot moves.

The classic desk question (Derman's smile regimes): after a spot move, does
implied vol stay pinned to the STRIKE (a call struck at 480 keeps yesterday's
vol) or to the MONEYNESS (the 25-delta line keeps its vol, so a fixed strike's
vol changes by the smile slope times the move)? The hedging consequence is
real: sticky-delta adds a term to every option's effective delta.

This study answers it from the repo's own daily SSVI fits. For each ticker
and each consecutive-day pair (t, t+1) with a meaningful spot move, the two
fitted 30d smiles are compared against both predictions in strike space
(k measured against the day-(t+1) forward; yesterday's smile enters shifted
by the log spot move ``d``):

* sticky-strike prediction:    sigma_hat(k) = sigma_t(k + d)
* sticky-delta prediction:     sigma_hat(k) = sigma_t(k)

Two discriminators, because they fail differently:

* **Demeaned RMS** (primary): RMS error of each prediction over an ATM band
  after removing each prediction's own ATM offset - this isolates how the
  smile SHAPE repositions and is immune to the day's parallel vol change.
* **Beta regression** (classic, reported with its caveat): the observed vol
  change at yesterday's ATM strike regressed on the sticky-delta-predicted
  change (-slope * move). beta ~ 0 reads sticky-strike, ~ 1 sticky-delta,
  > 1 the "sticky-local-vol" crash regime. Caveat stated up front: spot-vol
  correlation (the leverage effect) contaminates beta upward even in a
  sticky-strike world, which is exactly why the demeaned RMS is primary.

The study GATES itself: below ``MIN_PAIRS`` usable day-pairs per ticker it
prints what it is still waiting for and writes nothing, so the scheduled
daily run can call it unconditionally and the chart simply appears once the
history is long enough.

    python vol_snapshots/sticky.py               # runs, or says what's missing

Core comparisons are pure functions of smile callables and are pinned offline
by ``tests/test_sticky.py`` on synthetic smiles with KNOWN regimes.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# Keep `python vol_snapshots/sticky.py` (and the scheduled task) working from
# a plain clone: repo root on the path so package imports resolve.
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pricing_and_vol_surface.vol_surface import SSVIParams, ssvi_w  # noqa: E402
from vol_snapshots.fit_history import (  # noqa: E402
    HISTORY_CSV, load_history, usable_history,
)

SUMMARY_CSV = os.path.join(_HERE, "sticky_summary.csv")
CHART_PATH = os.path.join(_REPO, "charts", "vol_surface", "sticky_regimes.png")

TENOR = 30.0 / 365.0     # the smile compared day to day (interpolation-free:
                         # surface_history stores the fitted 30d ATM vol)
MIN_MOVE = 0.002         # |log spot move| below this: no signal, pair skipped
MAX_GAP_DAYS = 4         # t -> t+1 may span a weekend, not a data hole
MIN_PAIRS = 15           # per ticker, before any output is produced
BAND_STDS = 1.5          # ATM band half-width in sigma_atm*sqrt(T) units


# --------------------------------------------------------------------------- #
# Pure core: one day-pair, two smile callables                                #
# --------------------------------------------------------------------------- #
def pair_metrics(sig_t: Callable, sig_t1: Callable, d: float,
                 band: float, n_grid: int = 41) -> Dict[str, float]:
    """Compare day-(t+1)'s smile against both predictions from day t's.

    ``sig_t``/``sig_t1`` map log-moneyness (vs each day's own forward) to
    implied vol; ``d`` is the log forward move; ``band`` the ATM half-width.
    Returns demeaned RMS errors for both predictions plus the beta-regression
    observation pair (observed vs sticky-delta-predicted ATM-strike change).
    """
    k = np.linspace(-band, band, n_grid)
    actual = sig_t1(k)
    pred_strike = sig_t(k + d)     # yesterday's vol at the same STRIKE
    pred_delta = sig_t(k)          # yesterday's vol at the same MONEYNESS

    def demeaned_rms(pred):
        e = actual - pred
        return float(np.sqrt(np.mean((e - e.mean()) ** 2)))

    # Yesterday's ATM strike sits at k = -d in today's moneyness.
    obs = float(sig_t1(np.array([-d]))[0] - sig_t(np.array([0.0]))[0])
    pred = float(sig_t(np.array([-d]))[0] - sig_t(np.array([0.0]))[0])
    return {
        "rms_strike": demeaned_rms(pred_strike),
        "rms_delta": demeaned_rms(pred_delta),
        "beta_obs": obs,
        "beta_pred": pred,
    }


def regime_summary(pairs: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate one ticker's day-pairs: mean RMS per prediction and the OLS
    beta (with standard error) of observed on sticky-delta-predicted changes."""
    x = np.array([p["beta_pred"] for p in pairs])
    y = np.array([p["beta_obs"] for p in pairs])
    beta = float(np.sum(x * y) / np.sum(x * x))
    resid = y - beta * x
    dof = max(len(x) - 1, 1)
    se = float(np.sqrt(np.sum(resid**2) / dof / np.sum(x * x)))
    return {
        "n_pairs": len(pairs),
        "rms_strike": float(np.mean([p["rms_strike"] for p in pairs])),
        "rms_delta": float(np.mean([p["rms_delta"] for p in pairs])),
        "beta": beta,
        "beta_se": se,
    }


# --------------------------------------------------------------------------- #
# History adapter: surface_history rows -> smile callables                    #
# --------------------------------------------------------------------------- #
def smile_from_row(row) -> Optional[Callable]:
    """The fitted 30d smile sigma(k) reconstructed from one history row."""
    if not np.isfinite(row["atm_vol_30d"]):
        return None
    p = SSVIParams(rho=float(row["rho"]), eta=float(row["eta"]),
                   gamma=float(row["gamma"]))
    theta = float(row["atm_vol_30d"]) ** 2 * TENOR

    def sigma(k):
        return np.sqrt(ssvi_w(np.asarray(k, dtype=float), theta, p) / TENOR)

    return sigma


@dataclass
class TickerStudy:
    ticker: str
    summary: Dict[str, float]


def collect_pairs(hist_t: pd.DataFrame) -> List[Dict[str, float]]:
    """Usable consecutive-day pairs for one ticker's history rows."""
    hist_t = hist_t.sort_values("snapshot_date").reset_index(drop=True)
    dates = pd.to_datetime(hist_t["snapshot_date"])
    out = []
    for i in range(len(hist_t) - 1):
        gap = (dates.iloc[i + 1] - dates.iloc[i]).days
        if not (1 <= gap <= MAX_GAP_DAYS):
            continue
        row_t, row_t1 = hist_t.iloc[i], hist_t.iloc[i + 1]
        d = float(np.log(row_t1["spot"] / row_t["spot"]))
        if abs(d) < MIN_MOVE:
            continue
        sig_t, sig_t1 = smile_from_row(row_t), smile_from_row(row_t1)
        if sig_t is None or sig_t1 is None:
            continue
        band = BAND_STDS * float(row_t["atm_vol_30d"]) * np.sqrt(TENOR)
        out.append(pair_metrics(sig_t, sig_t1, d, band))
    return out


def run_study(hist: pd.DataFrame) -> (List[TickerStudy], List[str]):
    """Study every ticker with enough pairs; report what the rest still need."""
    studies, waiting = [], []
    for ticker, hist_t in hist.groupby("ticker"):
        pairs = collect_pairs(hist_t)
        if len(pairs) >= MIN_PAIRS:
            studies.append(TickerStudy(str(ticker), regime_summary(pairs)))
        else:
            waiting.append(f"{ticker}: {len(pairs)}/{MIN_PAIRS} usable pairs")
    return studies, waiting


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
def plot_study(studies: List[TickerStudy], out: str = CHART_PATH) -> Optional[str]:
    """Left: beta per ticker with 2-se bars against the 0 (sticky-strike) and
    1 (sticky-delta) reference lines. Right: the primary discriminator -
    demeaned RMS of both predictions, side by side per ticker."""
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

    beta = [s.summary["beta"] for s in studies]
    err = [2 * s.summary["beta_se"] for s in studies]
    axL.errorbar(xs, beta, yerr=err, fmt="o", color=ps.SERIES[0],
                 ecolor=ps.MUTED, capsize=4)
    axL.axhline(0, color=ps.BASELINE, lw=1.2)
    axL.axhline(1, color=ps.BASELINE, lw=1.2, ls="--")
    axL.text(len(xs) - 0.4, 0.02, "sticky-strike", color=ps.INK_2, fontsize=9)
    axL.text(len(xs) - 0.4, 1.02, "sticky-delta", color=ps.INK_2, fontsize=9)
    axL.set_xticks(xs, names)
    axL.set_title("Regime beta (observed vs sticky-delta-predicted, ±2 s.e.)")
    axL.set_ylabel("beta")

    w = 0.38
    axR.bar(xs - w / 2, [1e4 * s.summary["rms_strike"] for s in studies],
            width=w, color=ps.SERIES[0], label="sticky-strike error")
    axR.bar(xs + w / 2, [1e4 * s.summary["rms_delta"] for s in studies],
            width=w, color=ps.SERIES[1], label="sticky-delta error")
    axR.set_xticks(xs, names)
    axR.set_title("Demeaned RMS of each prediction (lower = better regime fit)")
    axR.set_ylabel("vol error (bp)")
    axR.legend()

    fig.suptitle("Sticky-strike vs sticky-delta on the fitted daily 30d smiles",
                 fontweight="bold")
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

    hist = usable_history(load_history(args.history))
    if hist.empty:
        print("no surface history yet - run vol_snapshots/fit_history.py first")
        return
    studies, waiting = run_study(hist)

    for line in waiting:
        print(f"  accumulating  {line}")
    if not studies:
        print(f"insufficient data: need {MIN_PAIRS} usable day-pairs "
              f"(consecutive fits, |move| >= {MIN_MOVE:.1%}) for at least one "
              f"ticker - the scheduled run will produce this automatically.")
        return

    print(f"{'ticker':<7} {'pairs':>5} {'beta':>7} {'2se':>6} "
          f"{'rms_strike(bp)':>15} {'rms_delta(bp)':>14}  verdict")
    rows = []
    for s in studies:
        m = s.summary
        verdict = ("sticky-delta" if m["rms_delta"] < m["rms_strike"]
                   else "sticky-strike")
        print(f"{s.ticker:<7} {m['n_pairs']:>5} {m['beta']:>7.2f} "
              f"{2 * m['beta_se']:>6.2f} {1e4 * m['rms_strike']:>15.1f} "
              f"{1e4 * m['rms_delta']:>14.1f}  {verdict}")
        rows.append({"ticker": s.ticker, **{k: round(v, 6) for k, v in m.items()},
                     "verdict": verdict})
    pd.DataFrame(rows).to_csv(args.summary, index=False)
    print(f"summary -> {args.summary}")
    if not args.no_chart:
        out = plot_study(studies, args.chart)
        if out:
            print(f"chart -> {out}")


if __name__ == "__main__":
    main()
