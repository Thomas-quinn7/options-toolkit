# IV-skew bubble indicator

`IV_skew.py` scans a large set of US names across market segments, applies
data-quality gates (volume, open interest, bid-ask spread, IV sanity, DTE
window), inverts implied vols **from bid-ask mid prices with the repo's own
Brent solver** (`vol_surface.iv_from_price` — yfinance's IV field is kept only
as a diagnostic), and computes the OTM put-minus-call IV skew per name.
Widespread *inverted* skew (calls richer than puts) is flagged as a
speculative-froth signal.

```bash
python skew_bubble_indicator/IV_skew.py --workers 5 --plot
python skew_bubble_indicator/IV_skew.py --anchor moneyness      # legacy bands
python skew_bubble_indicator/IV_skew.py --put-delta 0.10 --call-delta 0.10
```

## Delta-anchored strike selection

The OTM anchors are **delta-anchored** by default: the listed strikes nearest
the target Black-Scholes deltas — 25-delta put / 25-delta call, the
market-standard skew anchors — with deltas computed by the repo's own pricing
machinery (`pricing_and_vol_surface.black.greeks`) at each quote's
price-inverted IV.

Why it matters: a fixed price band compares different parts of the smile
across names. At ~30 DTE, 10% OTM is roughly a 35-delta strike on a high-vol
single name and a ~5-delta wing on a low-vol index ETF. On the captured
snapshots the difference is visible directly: SPY's 25-delta put anchor sits
at ~3% OTM (716 vs spot 739) where the old 90% band read the 664 wing, while
on TSLA the two selections nearly coincide. Delta anchoring puts every name's
measurement at the same point of its own smile.

Mechanics and fallback:

* Targets are configurable (`--put-delta` / `--call-delta`, or
  `TARGET_PUT_DELTA` / `TARGET_CALL_DELTA`; the put target's sign is
  optional).
* Candidates are OTM-side strikes with finite inverted IVs. If no strike gets
  within `DELTA_ANCHOR_TOLERANCE` (0.10) of the target delta — sparse or
  ATM-only chains — the delta anchor reports unavailable rather than silently
  anchoring elsewhere on the smile.
* Unavailability on **either** side makes the name fall back to the explicit
  moneyness-band path (90%/110% of spot), never a mixed anchor;
  `--anchor moneyness` forces the band path everywhere.
* Delta defaults to on because the snapshot-history study below found it
  available on 100% of observations with no material disagreement in sign
  versus the bands (and it is the convention the rest of the vol world
  quotes skew in).

Deltas are Black-Scholes deltas from European-inverted IVs by default; with
`--american` the IVs are CRR-inverted but the anchor delta is still the BS
delta at that IV — for the ~25-delta strikes consumed here the difference is
well inside one strike spacing.

## Inversion thresholds — heuristics with measured context

The segment-level alert thresholds (`WARNING_INVERSION_THRESHOLD = 40`,
`CRITICAL_INVERSION_THRESHOLD = 60`, in % of a segment's names with inverted
skew) were judgment calls. `threshold_study.py` now measures them against the
accumulated daily snapshot history (`vol_snapshots/data/`), running the
scanner's exact pipeline — same gates, same own-IV inversion, both anchoring
modes — over every captured (day, name):

```bash
python skew_bubble_indicator/threshold_study.py     # regenerates the table below
```

### Threshold study — run 2026-10-26, 35 distinct session close(s), 313 (day, name) observations

Per-day cross-section (the inversion rate is what the 40%/60% thresholds gate on):

| day | names | inverted % (delta) | inverted % (band) | median skew (delta) | median skew (band) | delta fallbacks |
|---|---|---|---|---|---|---|
| 2026-07-25 | 7 | 14% | 14% | +0.0816 | +0.1357 | 0 |
| 2026-07-27 | 9 | 0% | 0% | +0.0342 | +0.0584 | 0 |
| 2026-07-28 | 9 | 0% | 0% | +0.0274 | +0.0553 | 0 |
| 2026-07-29 | 9 | 12% | 11% | +0.0452 | +0.0687 | 1 |
| 2026-07-30 | 9 | 0% | 0% | +0.0402 | +0.0474 | 1 |
| 2026-07-31 | 9 | 0% | 0% | +0.0364 | +0.0459 | 0 |
| 2026-08-03 | 9 | 11% | 0% | +0.0320 | +0.0362 | 0 |
| 2026-08-04 | 9 | 25% | 22% | +0.0196 | +0.0297 | 1 |
| 2026-08-05 | 9 | 22% | 11% | +0.0137 | +0.0327 | 0 |
| 2026-08-06 | 2 | 0% | 0% | +0.0260 | +0.0604 | 0 |
| 2026-08-07 | 9 | 22% | 22% | +0.0217 | +0.0443 | 0 |
| 2026-08-10 | 9 | 22% | 22% | +0.0281 | +0.0402 | 0 |
| 2026-08-11 | 9 | 22% | 22% | +0.0301 | +0.0360 | 0 |
| 2026-08-12 | 9 | 22% | 22% | +0.0181 | +0.0350 | 0 |
| 2026-08-13 | 9 | 22% | 22% | +0.0264 | +0.0447 | 0 |
| 2026-08-14 | 9 | 11% | 22% | +0.0269 | +0.0503 | 0 |
| 2026-08-17 | 12 | 22% | 25% | +0.0237 | +0.0249 | 3 |
| 2026-08-18 | 11 | 22% | 18% | +0.0284 | +0.0407 | 2 |
| 2026-08-19 | 12 | 22% | 33% | +0.0289 | +0.0317 | 3 |
| 2026-08-20 | 12 | 30% | 33% | +0.0202 | +0.0304 | 2 |
| 2026-08-21 | 11 | 30% | 27% | +0.0190 | +0.0286 | 1 |
| 2026-08-24 | 9 | 22% | 11% | +0.0171 | +0.0323 | 0 |
| 2026-08-25 | 9 | 22% | 11% | +0.0217 | +0.0331 | 0 |
| 2026-08-26 | 9 | 33% | 11% | +0.0109 | +0.0288 | 0 |
| 2026-08-27 | 9 | 22% | 0% | +0.0168 | +0.0285 | 0 |
| 2026-08-28 | 9 | 11% | 11% | +0.0160 | +0.0321 | 0 |
| 2026-08-31 | 9 | 25% | 11% | +0.0266 | +0.0319 | 1 |
| 2026-09-01 | 9 | 22% | 11% | +0.0186 | +0.0507 | 0 |
| 2026-09-02 | 9 | 22% | 11% | +0.0163 | +0.0428 | 0 |
| 2026-09-03 | 9 | 22% | 22% | +0.0170 | +0.0345 | 0 |
| 2026-09-04 | 9 | 22% | 22% | +0.0211 | +0.0382 | 0 |
| 2026-09-11 | 9 | 22% | 0% | +0.0269 | +0.0400 | 0 |
| 2026-09-25 | 9 | 22% | 0% | +0.0234 | +0.0441 | 0 |
| 2026-10-02 | 3 | 0% | 0% | +0.0356 | +0.1001 | 0 |
| 2026-10-09 | 9 | 22% | 0% | +0.0250 | +0.0454 | 0 |

Pooled per-name skew distribution (vol points, put minus call):

| metric | n | min | p10 | p25 | p50 | p75 | p90 | max | share inverted |
|---|---|---|---|---|---|---|---|---|---|
| delta-anchored | 298 | -0.2861 | -0.0096 | +0.0085 | +0.0251 | +0.0416 | +0.0554 | +0.1377 | 19% |
| moneyness-band | 313 | -0.1861 | -0.0085 | +0.0118 | +0.0400 | +0.0984 | +0.1184 | +0.1800 | 14% |

Threshold context: the WARNING (>40%) / CRITICAL (>60%) inversion thresholds
would have fired on 0/35 and 0/35 day(s) (delta-anchored; 0/35 and 0/35 with
moneyness bands). Highest observed daily inversion rate: 33% (delta), 33%
(band).

Anchor agreement: on 298 observations where both anchors produced a skew,
signs agree on 92%; median |band − delta| = 0.0204; correlation 0.85. Delta
anchoring fell back to moneyness bands on 5% of observations.

The inverted tail is shallow: 19% of observations sit below zero, but the
10th percentile is only −0.01 vol points — most "inverted" names are
marginally so, not calls-rich manias (the original 2026-07-25 AAPL
pre-earnings observation at −0.16 remains the instructive deep one).

**Honest interpretation.** 35 market days of history calibrates the metric's
*range* in the captured (calm) regime — it shows where "normal" sits and how
far the thresholds are above it. It cannot validate bubble-prediction power:
that requires the thresholds to fire ahead of a drawdown at least once, i.e.
years of daily history spanning at least one froth episode. Until then the
40%/60% thresholds remain judgment calls — now with measured context. This
table strengthens automatically: the capture runs nightly and this script
recomputes from the full store; re-run it and refresh this section as the
history grows.

## Outputs and tests

Snapshots append to `daily_IV_skew_snapshot.csv` / `bubble_summary.csv` -
both idempotent per day (a same-day re-run replaces that day's rows instead
of double-counting the same delayed session in every downstream average);
the study writes per-observation rows to `threshold_study.csv`. The bubble
summary carries quality-gated companion columns
(`reliable_measurements`, `inversion_pct_reliable`) next to the headline
ones, and the ALERT LEVEL fires on the reliable-only inversion rate
whenever at least 5 reliable measurements exist (a low-quality row can no
longer move an alert on its own); below that it falls back to the all-valid
rate, and the basis each alert used is recorded on the row (`alert_basis`)
so the time series is self-describing. The headline `inversion_pct` column
keeps the all-valid definition for series continuity. Inversion prices the
chain's own dividend yield (put-call parity on matched strikes, refused to
0 when the implied carry is absurd) instead of assuming q=0, and the spot
fetched with the chain is reused rather than re-scraped. Everything is
unit-tested offline on synthetic chains priced from known smiles:
`tests/test_iv_skew.py` (own-IV pipeline, delta anchoring, fallback, config
plumbing, idempotent snapshot save, summary companion columns, alert basis,
parity-implied q) and `tests/test_threshold_study.py` (the study end-to-end
on a fixture snapshot store, including stale-weekend dedupe).
