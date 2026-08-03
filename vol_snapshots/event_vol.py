"""Event vol: term-structure build-up, inversion and crush around vol events.

Before a scheduled announcement (earnings, for the single names in the capture
list) the event's variance is concentrated in the expiries that span it:
short-dated ATM vol inflates, the term structure inverts, and the moment the
news is out the event variance vanishes - the "vol crush". The daily fitted
surfaces record all three phases, and this study extracts them.

**Events are detected endogenously** - no earnings calendar is imported. An
event is a one-day drop in the fitted 30d ATM vol of at least ``CRUSH_ABS``
vol points AND ``CRUSH_REL`` of the pre-day level, between consecutive fits.
For the mega-cap names in the capture universe such crushes are almost always
earnings; the honest claim is "a vol event resolved here", which is exactly
what the surface says. Each event records:

* the **build-up window**: 30d ATM vol and term slope for the days before and
  after the crush (aligned at the crush day for the averaged chart);
* the **implied event move**: from the pre-event term structure, treating the
  182d vol as the diffusive base, the extra short-dated variance is
  ``(sigma_30^2 - sigma_182^2) * T_30`` and its square root is the move the
  options market charged for the event;
* the **realized move**: the log spot change over the crush day - so every
  event contributes one point to an implied-vs-realized scatter, the event
  analogue of the replay's realised-vs-implied result.

Stated limits: one fit per day means an event's timing is resolved only to
the day; the 182d-as-base decomposition attributes ALL excess short-dated
variance to a single event inside 30 days (a second event or an elevated
short-vol regime inflates it); detection needs the pre-day fit to exist.

The study gates itself: with no detected events it prints what it is watching
and writes nothing, so the scheduled run calls it unconditionally and the
outputs appear once the first event (earnings season) lands in the data.

    python vol_snapshots/event_vol.py

Pinned offline by ``tests/test_event_vol.py`` on histories with an injected
synthetic event and a flat control.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# Keep `python vol_snapshots/event_vol.py` (and the scheduled task) working
# from a plain clone: repo root on the path so package imports resolve.
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vol_snapshots.fit_history import (  # noqa: E402
    HISTORY_CSV, load_history, usable_history,
)

EVENTS_CSV = os.path.join(_HERE, "event_vol.csv")
CHART_PATH = os.path.join(_REPO, "charts", "vol_surface", "event_vol.png")

T_SHORT = 30.0 / 365.0
CRUSH_ABS = 0.03         # >= 3 vol points off the 30d ATM in one day, and
CRUSH_REL = 0.10         # >= 10% of the pre-day level
MAX_GAP_DAYS = 4         # crush must resolve between consecutive fits
WINDOW_PRE, WINDOW_POST = 10, 5   # calendar days charted around the crush

EVENT_COLUMNS = [
    "ticker", "event_date", "pre_vol_30d", "post_vol_30d", "crush",
    "pre_term_slope", "implied_event_move", "realized_move",
]


def implied_event_move(vol_30d: float, vol_182d: float) -> float:
    """Move the market charged for the event: sqrt of the excess short-dated
    variance, treating the 182d vol as the diffusive base."""
    if not (np.isfinite(vol_30d) and np.isfinite(vol_182d)):
        return float("nan")
    return float(np.sqrt(max(vol_30d**2 - vol_182d**2, 0.0) * T_SHORT))


@dataclass
class VolEvent:
    ticker: str
    event_date: str              # the crush-observation day (first post fit)
    pre_vol_30d: float
    post_vol_30d: float
    crush: float
    pre_term_slope: float
    implied_event_move: float
    realized_move: float
    window_offsets: List[int] = field(default_factory=list)
    window_vol_30d: List[float] = field(default_factory=list)
    window_term_slope: List[float] = field(default_factory=list)


def detect_events(hist_t: pd.DataFrame) -> List[VolEvent]:
    """All crush events in one ticker's history rows."""
    hist_t = hist_t.sort_values("snapshot_date").reset_index(drop=True)
    dates = pd.to_datetime(hist_t["snapshot_date"])
    events = []
    for i in range(len(hist_t) - 1):
        gap = (dates.iloc[i + 1] - dates.iloc[i]).days
        if not (1 <= gap <= MAX_GAP_DAYS):
            continue
        pre, post = hist_t.iloc[i], hist_t.iloc[i + 1]
        v0, v1 = float(pre["atm_vol_30d"]), float(post["atm_vol_30d"])
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        drop = v0 - v1
        if drop < CRUSH_ABS or drop < CRUSH_REL * v0:
            continue

        t0 = dates.iloc[i + 1]
        in_window = (dates - t0).dt.days.between(-WINDOW_PRE, WINDOW_POST)
        win = hist_t[in_window]
        events.append(VolEvent(
            ticker=str(pre["ticker"]),
            event_date=str(post["snapshot_date"]),
            pre_vol_30d=round(v0, 6),
            post_vol_30d=round(v1, 6),
            crush=round(drop, 6),
            pre_term_slope=round(float(pre["term_slope"]), 6),
            implied_event_move=round(
                implied_event_move(v0, float(pre["atm_vol_182d"])), 6),
            realized_move=round(
                abs(float(np.log(post["spot"] / pre["spot"]))), 6),
            window_offsets=[int(x) for x in
                            (pd.to_datetime(win["snapshot_date"]) - t0).dt.days],
            window_vol_30d=[round(float(x), 6) for x in win["atm_vol_30d"]],
            window_term_slope=[round(float(x), 6) for x in win["term_slope"]],
        ))
    return events


def run_study(hist: pd.DataFrame) -> List[VolEvent]:
    events: List[VolEvent] = []
    for _, hist_t in hist.groupby("ticker"):
        events.extend(detect_events(hist_t))
    events.sort(key=lambda e: (e.event_date, e.ticker))
    return events


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
def plot_events(events: List[VolEvent], out: str = CHART_PATH) -> Optional[str]:
    """Left: each event's 30d ATM vol path aligned at the crush day (term
    slope dashed, right axis). Right: implied event move vs realized gap -
    one point per event, y=x as the fair-pricing reference."""
    if not events:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    import plotstyle as ps
    ps.apply_style()

    tickers = sorted({e.ticker for e in events})
    color = {t: ps.series_color(i % 8) for i, t in enumerate(tickers)}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    axL2 = axL.twinx()
    for e in events:
        axL.plot(e.window_offsets, [100 * v for v in e.window_vol_30d],
                 marker="o", markersize=3, color=color[e.ticker],
                 label=f"{e.ticker} {e.event_date}")
        axL2.plot(e.window_offsets, [100 * s for s in e.window_term_slope],
                  color=color[e.ticker], ls="--", alpha=0.5)
    axL.axvline(0, color=ps.BASELINE, lw=1.0)
    axL2.axhline(0, color=ps.BASELINE, lw=0.8, ls=":")
    axL.set_title("30d ATM vol into and out of the event\n"
                  "(solid, left; dashed = term slope, right)")
    axL.set_xlabel("days from crush")
    axL.set_ylabel("30d ATM vol (%)")
    axL2.set_ylabel("term slope (vol pts)")
    axL2.grid(False)
    axL.legend(fontsize=8)

    imp = [100 * e.implied_event_move for e in events]
    rea = [100 * e.realized_move for e in events]
    for e, x, y in zip(events, imp, rea):
        axR.scatter(x, y, color=color[e.ticker], s=55)
        axR.annotate(e.ticker, (x, y), textcoords="offset points",
                     xytext=(5, 4), fontsize=8, color=ps.INK_2)
    lim = max(imp + rea + [1.0]) * 1.15
    axR.plot([0, lim], [0, lim], color=ps.MUTED, lw=1.0, ls="--")
    axR.set_xlim(0, lim)
    axR.set_ylim(0, lim)
    axR.set_title("Event move: implied by the pre-event term structure\n"
                  "vs realized over the crush day (y = x fair)")
    axR.set_xlabel("implied event move (%)")
    axR.set_ylabel("realized move (%)")

    fig.suptitle("Event vol on the fitted daily surfaces "
                 "(endogenously detected crushes)", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", default=HISTORY_CSV)
    ap.add_argument("--events", default=EVENTS_CSV)
    ap.add_argument("--chart", default=CHART_PATH)
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args(argv)

    hist = usable_history(load_history(args.history))
    if hist.empty:
        print("no surface history yet - run vol_snapshots/fit_history.py first")
        return
    events = run_study(hist)
    if not events:
        n = hist["snapshot_date"].nunique()
        print(f"no vol-crush events detected yet ({n} fitted day(s); watching "
              f"for a one-day 30d-ATM drop >= {CRUSH_ABS:.0%} abs and "
              f"{CRUSH_REL:.0%} relative - earnings season will provide).")
        return

    print(f"{'ticker':<7} {'event':<11} {'pre':>6} {'post':>6} {'crush':>6} "
          f"{'slope':>7} {'implied':>8} {'realized':>8}")
    for e in events:
        print(f"{e.ticker:<7} {e.event_date:<11} {100*e.pre_vol_30d:>6.1f} "
              f"{100*e.post_vol_30d:>6.1f} {100*e.crush:>6.1f} "
              f"{100*e.pre_term_slope:>7.2f} {100*e.implied_event_move:>7.2f}% "
              f"{100*e.realized_move:>7.2f}%")
    pd.DataFrame([{c: getattr(e, c) for c in EVENT_COLUMNS}
                  for e in events]).to_csv(args.events, index=False)
    print(f"events -> {args.events}")
    if not args.no_chart:
        out = plot_events(events, args.chart)
        if out:
            print(f"chart -> {out}")


if __name__ == "__main__":
    main()
