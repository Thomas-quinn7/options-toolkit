"""Offline tests for the threshold study (skew_bubble_indicator/threshold_study.py).

A synthetic snapshot store is built in the capture schema: two tickers priced
from KNOWN smiles - one put-skewed (normal), one call-skewed (inverted) - over
two market days, plus a stale weekend duplicate of the second day. The study
must recover the smile signs from prices alone, dedupe the stale day, and the
summaries/markdown must report the 50% inversion cross-section correctly.

Run:  python -m pytest tests/test_threshold_study.py -q
"""

import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from pricing_and_vol_surface.vol_surface import bs_call
from skew_bubble_indicator import threshold_study as TS
from vol_snapshots.capture import COLUMNS, save_snapshot

S0, R0 = 100.0, 0.03
DAYS = ["2026-01-05", "2026-01-06"]
STALE_DAY = "2026-01-07"                      # duplicates DAYS[1]'s spot
SPOTS = {"2026-01-05": 100.0, "2026-01-06": 101.0, STALE_DAY: 101.0}


def put_skewed(K: float, spot: float) -> float:
    """Normal equity smile: vol rises as strikes fall below spot. Kinked AT
    spot so the skew is nonzero at the 25-delta anchors, not just the wings."""
    return 0.20 + 0.6 * max(0.0, (spot - K) / spot)


def call_skewed(K: float, spot: float) -> float:
    """Inverted smile: calls richer than puts (the bubble signature)."""
    return 0.20 + 0.6 * max(0.0, (K - spot) / spot)


def snapshot_frame(day: str, ticker: str, smile) -> pd.DataFrame:
    """One (day, ticker) snapshot in the capture schema, priced from the smile;
    the vendor IV column is garbage on purpose (the study must invert prices)."""
    spot = SPOTS[day]
    expiry = (_dt.date.fromisoformat(day) + _dt.timedelta(days=30)).isoformat()
    T = 30 / 365.0
    rows = []
    for K in np.arange(80.0, 121.0, 2.5):
        sigma = smile(K, spot)
        call = bs_call(spot, K, T, R0, sigma)
        for otype in ("call", "put"):
            price = call if otype == "call" else call - spot + K * np.exp(-R0 * T)
            half = 0.03 * price
            rows.append({
                "snapshot_date": day, "ticker": ticker, "spot": spot,
                "riskfree": R0, "expiry": expiry, "otype": otype, "strike": K,
                "bid": price - half, "ask": price + half, "last": price,
                "volume": 500, "open_interest": 1000, "iv_yf": 9.99,
            })
    return pd.DataFrame(rows)[COLUMNS]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("snapdata"))
    for day in DAYS + [STALE_DAY]:
        save_snapshot(snapshot_frame(day, "AAA", put_skewed), root)
        save_snapshot(snapshot_frame(day, "BBB", call_skewed), root)
    return root


@pytest.fixture(scope="module")
def rows(store):
    return TS.run_study(store)


def test_study_recovers_smile_signs_and_dedupes_stale_day(rows):
    # 2 tickers x 2 market days; the stale third day (identical spot) is gone
    assert len(rows) == 4
    assert set(rows["snapshot_date"]) == set(DAYS)
    aaa = rows[rows["ticker"] == "AAA"]
    bbb = rows[rows["ticker"] == "BBB"]
    assert (aaa["skew_band"] > 0).all() and (aaa["skew_delta"] > 0).all()
    assert (bbb["skew_band"] < 0).all() and (bbb["skew_delta"] < 0).all()
    # delta anchoring worked everywhere on these full liquid chains...
    assert rows["delta_available"].all()
    # ...and anchored OTM on the correct side of spot
    assert (rows["delta_put_strike"] < rows["spot"]).all()
    assert (rows["delta_call_strike"] > rows["spot"]).all()


def test_per_day_summary_reports_the_inversion_cross_section(rows):
    days = TS.per_day_summary(rows)
    assert list(days["snapshot_date"]) == DAYS
    assert (days["n_names"] == 2).all()
    # one of two names inverted -> 50% on both metrics, every day
    assert np.allclose(days["inv_pct_delta"], 50.0)
    assert np.allclose(days["inv_pct_band"], 50.0)
    assert (days["delta_fallbacks"] == 0).all()


def test_pooled_and_agreement_summaries(rows):
    pooled = TS.pooled_summary(rows)
    assert set(pooled["metric"]) == {"delta-anchored", "moneyness-band"}
    assert (pooled["n_obs"] == 4).all()
    assert np.allclose(pooled["share_inverted_pct"], 50.0)
    assert (pooled["p10"] <= pooled["p50"]).all()
    assert (pooled["p50"] <= pooled["p90"]).all()
    agree = TS.agreement_summary(rows)
    assert agree["n_obs"] == 4
    assert agree["sign_agree_pct"] == 100.0
    assert agree["fallback_pct"] == 0.0


def test_threshold_context_counts_fired_days(rows):
    ctx = TS.threshold_context(TS.per_day_summary(rows))
    assert ctx["n_days"] == 2
    # 50% inversion: above WARNING (40), below CRITICAL (60), on both metrics
    assert ctx["warning_fired_delta"] == 2 and ctx["critical_fired_delta"] == 0
    assert ctx["warning_fired_band"] == 2 and ctx["critical_fired_band"] == 0


def test_markdown_block_is_dated_and_honest(rows):
    md = TS.format_markdown(rows, run_date="2026-07-28")
    assert "Threshold study — run 2026-07-28" in md
    assert "2 distinct session close(s)" in md
    assert "cannot validate bubble-prediction power" in md
    # empty history degrades gracefully rather than crashing
    md_empty = TS.format_markdown(pd.DataFrame(), run_date="2026-07-28")
    assert "No usable snapshot history yet" in md_empty
