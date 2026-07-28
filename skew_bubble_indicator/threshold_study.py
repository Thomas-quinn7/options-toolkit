"""Measure the IV-skew scanner's metric and thresholds on the snapshot history.

``IV_skew.py`` flags an index segment when the fraction of names with
*inverted* skew (OTM call IV above OTM put IV) crosses
``WARNING_INVERSION_THRESHOLD`` (40%) or ``CRITICAL_INVERSION_THRESHOLD``
(60%). Those thresholds were chosen by judgment, not measured against data.
This script runs the scanner's exact pipeline - the same quality gates, the
same own-IV inversion, both strike-anchoring modes - over every captured day
in ``vol_snapshots/data/`` and reports:

* the per-day cross-sectional inversion rate (what the thresholds gate on),
* the pooled distribution of the per-name skew metric (percentiles),
* how often each threshold would have fired,
* whether the delta-anchored and moneyness-band metrics disagree.

**What this can and cannot claim.** With days-to-weeks of snapshots this
calibrates the metric's RANGE in one (calm) regime - where "normal" sits, how
far the thresholds are above it. It CANNOT validate bubble-prediction power:
that needs the thresholds to fire ahead of a drawdown at least once, i.e.
years of history spanning at least one froth episode. The script is stateless
and recomputes from the raw data store, so the same command strengthens the
table as the daily capture accrues.

Stale weekend/holiday captures (identical spot to the previous captured day,
per ticker) are dropped, matching ``vol_snapshots/replay.py``.

Run:  python skew_bubble_indicator/threshold_study.py
      (paste the printed markdown into skew_bubble_indicator/README.md)

Per-observation rows land in ``skew_bubble_indicator/threshold_study.csv``.
Pinned offline by ``tests/test_threshold_study.py`` on a synthetic snapshot
day priced from a known smile.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from skew_bubble_indicator import IV_skew as X  # noqa: E402
from vol_snapshots.capture import DATA_ROOT, load_snapshots  # noqa: E402

OUT_CSV = os.path.join(_HERE, "threshold_study.csv")

DEFAULT_RISK_FREE = 0.04
TARGET_DTE = 30          # the scanner prefers the expiry nearest 30 DTE
PCTS = [10, 25, 50, 75, 90]


# --------------------------------------------------------------------------- #
# Snapshot -> scanner-shaped chains (pure)                                     #
# --------------------------------------------------------------------------- #
def dedupe_stale_days(snaps: pd.DataFrame) -> pd.DataFrame:
    """Drop (day, ticker) blocks whose spot is identical to that ticker's
    previous captured day - weekend/holiday captures re-serving the prior
    session's close (same convention as vol_snapshots/replay.py)."""
    keep = []
    for _, tdf in snaps.groupby("ticker"):
        prev_spot = None
        for day in sorted(tdf["snapshot_date"].unique()):
            spot = float(tdf.loc[tdf["snapshot_date"] == day, "spot"].iloc[0])
            if prev_spot is None or spot != prev_spot:
                keep.append((day, str(tdf["ticker"].iloc[0])))
            prev_spot = spot
    if not keep:
        return snaps.iloc[0:0]
    key = pd.MultiIndex.from_frame(snaps[["snapshot_date", "ticker"]])
    return snaps[key.isin(keep)]


def pick_expiry(day_df: pd.DataFrame, snapshot_date: str) -> Optional[str]:
    """The scanner's expiry choice: within [MIN_DTE, MAX_DTE], nearest 30d."""
    snap = _dt.date.fromisoformat(str(snapshot_date)[:10])
    best, best_gap = None, None
    for expiry in day_df["expiry"].unique():
        dte = (_dt.date.fromisoformat(str(expiry)[:10]) - snap).days
        if not (X.MIN_DTE <= dte <= X.MAX_DTE):
            continue
        gap = abs(dte - TARGET_DTE)
        if best_gap is None or gap < best_gap:
            best, best_gap = expiry, gap
    return best


def to_scanner_frame(side_df: pd.DataFrame) -> pd.DataFrame:
    """Snapshot schema -> the yfinance-shaped columns IV_skew consumes."""
    return pd.DataFrame({
        "strike": side_df["strike"].astype(float).values,
        "bid": side_df["bid"].astype(float).values,
        "ask": side_df["ask"].astype(float).values,
        "volume": side_df["volume"].astype(float).values,
        "openInterest": side_df["open_interest"].astype(float).values,
        "lastPrice": side_df["last"].astype(float).values,
        "impliedVolatility": side_df["iv_yf"].astype(float).values,
    })


def _skew_at(puts: pd.DataFrame, calls: pd.DataFrame, put_k: float,
             call_k: float, spot: float) -> Dict[str, float]:
    """The scanner's metric at given anchor strikes, volume-weighted fallback
    included (mirrors compute_iv_skew_metric)."""
    put_iv = X.find_mean_iv_at_strike(puts, put_k)
    call_iv = X.find_mean_iv_at_strike(calls, call_k)
    if np.isnan(put_iv) or np.isnan(call_iv):
        put_iv, call_iv = X.compute_volume_weighted_iv_skew(puts, calls, spot)
    return {"put_iv": put_iv, "call_iv": call_iv, "skew": put_iv - call_iv}


def ticker_day_row(day_df: pd.DataFrame) -> Optional[Dict]:
    """One (day, ticker) snapshot -> both skew metrics through the scanner's
    exact pipeline, or None if the day fails the scanner's gates."""
    day = str(day_df["snapshot_date"].iloc[0])
    ticker = str(day_df["ticker"].iloc[0])
    spot = float(day_df["spot"].iloc[0])
    r = float(day_df["riskfree"].iloc[0])
    if not np.isfinite(r):
        r = DEFAULT_RISK_FREE

    expiry = pick_expiry(day_df, day)
    if expiry is None:
        return None
    sl = day_df[day_df["expiry"] == expiry]
    dte = (_dt.date.fromisoformat(str(expiry)[:10])
           - _dt.date.fromisoformat(day[:10])).days
    T = max(dte, 1) / 365.0

    puts = X.validate_option_data(
        to_scanner_frame(sl[sl["otype"] == "put"]), spot, T, r, "put")
    calls = X.validate_option_data(
        to_scanner_frame(sl[sl["otype"] == "call"]), spot, T, r, "call")
    if len(puts) < 3 or len(calls) < 3:      # the scanner's minimum-data gate
        return None

    chain = SimpleNamespace(puts=puts, calls=calls)
    band_put_k, band_call_k = X.get_otm_strike_targets(spot, chain)
    band = _skew_at(puts, calls, band_put_k, band_call_k, spot)

    targets = X.get_delta_anchored_strike_targets(puts, calls, spot, T, r)
    if targets is not None:
        delta = _skew_at(puts, calls, targets[0], targets[1], spot)
        delta_put_k, delta_call_k = targets
    else:
        delta = {"put_iv": np.nan, "call_iv": np.nan, "skew": np.nan}
        delta_put_k = delta_call_k = np.nan

    return {
        "snapshot_date": day, "ticker": ticker, "spot": spot,
        "expiry": str(expiry), "dte": dte,
        "n_puts": len(puts), "n_calls": len(calls),
        "band_put_strike": band_put_k, "band_call_strike": band_call_k,
        "skew_band": band["skew"],
        "delta_available": targets is not None,
        "delta_put_strike": delta_put_k, "delta_call_strike": delta_call_k,
        "skew_delta": delta["skew"],
    }


def run_study(root: str = DATA_ROOT) -> pd.DataFrame:
    """Every usable (market day, ticker) in the store -> one row each."""
    snaps = load_snapshots(root)
    if snaps.empty:
        return pd.DataFrame()
    snaps = dedupe_stale_days(snaps)
    rows = []
    for (_, _), df in snaps.groupby(["snapshot_date", "ticker"]):
        row = ticker_day_row(df)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summaries (pure)                                                             #
# --------------------------------------------------------------------------- #
def per_day_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per market day: the cross-sectional inversion rate each metric implies
    - the quantity the WARNING/CRITICAL thresholds actually gate on."""
    out = []
    for day, ddf in rows.groupby("snapshot_date"):
        band = ddf["skew_band"].dropna()
        delta = ddf["skew_delta"].dropna()
        out.append({
            "snapshot_date": day,
            "n_names": len(ddf),
            "inv_pct_delta": 100.0 * (delta < 0).mean() if len(delta) else np.nan,
            "inv_pct_band": 100.0 * (band < 0).mean() if len(band) else np.nan,
            "median_skew_delta": delta.median() if len(delta) else np.nan,
            "median_skew_band": band.median() if len(band) else np.nan,
            "delta_fallbacks": int((~ddf["delta_available"]).sum()),
        })
    return pd.DataFrame(out).sort_values("snapshot_date").reset_index(drop=True)


def pooled_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled per-name skew distribution for each metric."""
    out = []
    for label, col in (("delta-anchored", "skew_delta"),
                       ("moneyness-band", "skew_band")):
        s = rows[col].dropna()
        rec = {"metric": label, "n_obs": len(s)}
        for p in PCTS:
            rec[f"p{p}"] = float(np.percentile(s, p)) if len(s) else np.nan
        rec["min"] = s.min() if len(s) else np.nan
        rec["max"] = s.max() if len(s) else np.nan
        rec["share_inverted_pct"] = 100.0 * (s < 0).mean() if len(s) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def agreement_summary(rows: pd.DataFrame) -> Dict[str, float]:
    """Do the two anchoring modes tell the same story per name?"""
    both = rows.dropna(subset=["skew_band", "skew_delta"])
    n = len(both)
    if n == 0:
        return {"n_obs": 0, "sign_agree_pct": np.nan,
                "median_abs_diff": np.nan, "corr": np.nan,
                "fallback_pct": 100.0 * (~rows["delta_available"]).mean()
                if len(rows) else np.nan}
    same_sign = (np.sign(both["skew_band"]) == np.sign(both["skew_delta"]))
    return {
        "n_obs": n,
        "sign_agree_pct": 100.0 * same_sign.mean(),
        "median_abs_diff": float((both["skew_band"] - both["skew_delta"]).abs().median()),
        "corr": float(both["skew_band"].corr(both["skew_delta"])) if n >= 3 else np.nan,
        "fallback_pct": 100.0 * (~rows["delta_available"]).mean(),
    }


def threshold_context(day_summary: pd.DataFrame) -> Dict[str, object]:
    """Where the current thresholds sit against the observed history."""
    out: Dict[str, object] = {"n_days": len(day_summary)}
    for label, col in (("delta", "inv_pct_delta"), ("band", "inv_pct_band")):
        s = day_summary[col].dropna()
        out[f"max_inv_pct_{label}"] = s.max() if len(s) else np.nan
        out[f"warning_fired_{label}"] = int((s > X.WARNING_INVERSION_THRESHOLD).sum())
        out[f"critical_fired_{label}"] = int((s > X.CRITICAL_INVERSION_THRESHOLD).sum())
    return out


def _fmt(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def format_markdown(rows: pd.DataFrame, run_date: Optional[str] = None) -> str:
    """The dated README block: tables plus the honest interpretation."""
    run_date = run_date or _dt.date.today().isoformat()
    if rows.empty:
        return (f"### Threshold study — run {run_date}\n\n"
                "No usable snapshot history yet: run `vol_snapshots/capture.py` "
                "daily and re-run this script.")
    days = per_day_summary(rows)
    pooled = pooled_summary(rows)
    agree = agreement_summary(rows)
    ctx = threshold_context(days)

    lines: List[str] = []
    lines.append(f"### Threshold study — run {run_date}, "
                 f"{ctx['n_days']} distinct session close(s), "
                 f"{len(rows)} (day, name) observations")
    lines.append("")
    lines.append("Per-day cross-section (the inversion rate is what the "
                 "40%/60% thresholds gate on):")
    lines.append("")
    lines.append("| day | names | inverted % (delta) | inverted % (band) | "
                 "median skew (delta) | median skew (band) | delta fallbacks |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in days.itertuples(index=False):
        lines.append(
            f"| {r.snapshot_date} | {r.n_names} "
            f"| {r.inv_pct_delta:.0f}% | {r.inv_pct_band:.0f}% "
            f"| {_fmt(r.median_skew_delta)} | {_fmt(r.median_skew_band)} "
            f"| {r.delta_fallbacks} |")
    lines.append("")
    lines.append("Pooled per-name skew distribution (vol points, put minus call):")
    lines.append("")
    lines.append("| metric | n | min | p10 | p25 | p50 | p75 | p90 | max | share inverted |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in pooled.itertuples(index=False):
        lines.append(
            f"| {r.metric} | {r.n_obs} | {_fmt(r.min)} "
            f"| {_fmt(r.p10)} | {_fmt(r.p25)} | {_fmt(r.p50)} "
            f"| {_fmt(r.p75)} | {_fmt(r.p90)} | {_fmt(r.max)} "
            f"| {r.share_inverted_pct:.0f}% |")
    lines.append("")
    lines.append(
        f"Threshold context: the WARNING (>{X.WARNING_INVERSION_THRESHOLD}%) / "
        f"CRITICAL (>{X.CRITICAL_INVERSION_THRESHOLD}%) inversion thresholds "
        f"would have fired on "
        f"{ctx['warning_fired_delta']}/{ctx['n_days']} and "
        f"{ctx['critical_fired_delta']}/{ctx['n_days']} day(s) (delta-anchored; "
        f"{ctx['warning_fired_band']}/{ctx['n_days']} and "
        f"{ctx['critical_fired_band']}/{ctx['n_days']} with moneyness bands). "
        f"Highest observed daily inversion rate: "
        f"{ctx['max_inv_pct_delta']:.0f}% (delta), "
        f"{ctx['max_inv_pct_band']:.0f}% (band).")
    lines.append("")
    lines.append(
        f"Anchor agreement: on {agree['n_obs']} observations where both "
        f"anchors produced a skew, signs agree on "
        f"{agree['sign_agree_pct']:.0f}%; median |band − delta| = "
        f"{agree['median_abs_diff']:.4f}"
        + (f"; correlation {agree['corr']:.2f}"
           if np.isfinite(agree["corr"]) else "")
        + f". Delta anchoring fell back to moneyness bands on "
          f"{agree['fallback_pct']:.0f}% of observations.")
    lines.append("")
    lines.append(
        f"**Honest interpretation.** {ctx['n_days']} market day(s) of history "
        "calibrates the metric's *range* in the captured (calm) regime — it "
        "shows where \"normal\" sits and how far the thresholds are above it. "
        "It cannot validate bubble-prediction power: that requires the "
        "thresholds to fire ahead of a drawdown at least once, i.e. years of "
        "daily history spanning at least one froth episode. Until then the "
        "40%/60% thresholds remain judgment calls — now with measured "
        "context. This table strengthens automatically: the capture runs "
        "nightly and this script recomputes from the full store.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out-csv", default=OUT_CSV)
    args = ap.parse_args(argv)

    rows = run_study(args.data_root)
    if not rows.empty:
        rows.to_csv(args.out_csv, index=False)
        print(f"per-observation rows -> {args.out_csv}\n")
    print(format_markdown(rows))


if __name__ == "__main__":
    main()
