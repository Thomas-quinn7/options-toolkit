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
* Delta defaults to on because it is the convention the rest of the vol
  world quotes skew in. On the snapshot history below it was available on
  95% of observations and agreed in sign with the bands on about 90%. It is
  NOT available everywhere: the sector ETFs (XLE, XLF, XLK) never produced a
  delta anchor after the quality gates and always fell back to the bands.
  Each scanner row records which path produced its number (`anchor`).

Deltas are computed at the same parity-implied dividend yield the IVs were
inverted at. They are Black-Scholes deltas from European-inverted IVs by default; with
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

The study inverts at each chain's parity-implied dividend yield, as the live
scanner does (until 2026-11-04 it inverted at q=0, which flipped the sign of
11 of 292 delta-anchored observations on a 32-session subset).

### Threshold study — run 2026-11-04, 35 distinct session close(s), 313 (day, name) observations

Per-day cross-section (the inversion rate is what the 40%/60% thresholds gate on):

| day | names | inverted % (delta) | inverted % (band) | median skew (delta) | median skew (band) | delta fallbacks |
|---|---|---|---|---|---|---|
| 2026-07-25 | 7 | 29% | 14% | +0.0617 | +0.1220 | 0 |
| 2026-07-27 | 9 | 11% | 11% | +0.0294 | +0.0545 | 0 |
| 2026-07-28 | 9 | 11% | 22% | +0.0250 | +0.0578 | 0 |
| 2026-07-29 | 9 | 0% | 11% | +0.0437 | +0.0644 | 1 |
| 2026-07-30 | 9 | 0% | 0% | +0.0415 | +0.0428 | 1 |
| 2026-07-31 | 9 | 0% | 0% | +0.0255 | +0.0502 | 0 |
| 2026-08-03 | 9 | 11% | 0% | +0.0216 | +0.0323 | 0 |
| 2026-08-04 | 9 | 25% | 22% | +0.0140 | +0.0304 | 1 |
| 2026-08-05 | 9 | 22% | 11% | +0.0134 | +0.0308 | 0 |
| 2026-08-06 | 2 | 0% | 0% | +0.0286 | +0.0622 | 0 |
| 2026-08-07 | 9 | 22% | 22% | +0.0134 | +0.0350 | 0 |
| 2026-08-10 | 9 | 22% | 22% | +0.0146 | +0.0334 | 0 |
| 2026-08-11 | 9 | 22% | 22% | +0.0211 | +0.0319 | 0 |
| 2026-08-12 | 9 | 22% | 22% | +0.0139 | +0.0318 | 0 |
| 2026-08-13 | 9 | 22% | 22% | +0.0170 | +0.0336 | 0 |
| 2026-08-14 | 9 | 22% | 22% | +0.0156 | +0.0408 | 0 |
| 2026-08-17 | 12 | 22% | 25% | +0.0168 | +0.0256 | 3 |
| 2026-08-18 | 11 | 22% | 18% | +0.0205 | +0.0353 | 2 |
| 2026-08-19 | 12 | 22% | 33% | +0.0159 | +0.0301 | 3 |
| 2026-08-20 | 12 | 30% | 33% | +0.0166 | +0.0213 | 2 |
| 2026-08-21 | 11 | 30% | 27% | +0.0161 | +0.0239 | 1 |
| 2026-08-24 | 9 | 11% | 22% | +0.0138 | +0.0348 | 0 |
| 2026-08-25 | 9 | 22% | 22% | +0.0176 | +0.0362 | 0 |
| 2026-08-26 | 9 | 22% | 22% | +0.0108 | +0.0310 | 0 |
| 2026-08-27 | 9 | 33% | 11% | +0.0121 | +0.0294 | 0 |
| 2026-08-28 | 9 | 11% | 22% | +0.0112 | +0.0308 | 0 |
| 2026-08-31 | 9 | 25% | 22% | +0.0154 | +0.0352 | 1 |
| 2026-09-01 | 9 | 22% | 11% | +0.0184 | +0.0499 | 0 |
| 2026-09-02 | 9 | 22% | 22% | +0.0166 | +0.0435 | 0 |
| 2026-09-03 | 9 | 22% | 22% | +0.0164 | +0.0356 | 0 |
| 2026-09-04 | 9 | 22% | 22% | +0.0172 | +0.0391 | 0 |
| 2026-09-11 | 9 | 22% | 0% | +0.0222 | +0.0404 | 0 |
| 2026-09-25 | 9 | 22% | 0% | +0.0203 | +0.0443 | 0 |
| 2026-10-02 | 3 | 0% | 0% | +0.0364 | +0.1009 | 0 |
| 2026-10-09 | 9 | 11% | 0% | +0.0222 | +0.0449 | 0 |

Pooled per-name skew distribution (IV as a decimal, put minus call; 0.01 = 1 vol point):

| metric | n | min | p10 | p25 | p50 | p75 | p90 | max | share inverted |
|---|---|---|---|---|---|---|---|---|---|
| delta-anchored | 298 | -0.2681 | -0.0089 | +0.0054 | +0.0176 | +0.0371 | +0.0487 | +0.1377 | 19% |
| moneyness-band | 313 | -0.1865 | -0.0080 | +0.0091 | +0.0365 | +0.0955 | +0.1160 | +0.1481 | 17% |

Threshold context: the WARNING (>40%) / CRITICAL (>60%) inversion thresholds would have fired on 0/35 and 0/35 day(s) (delta-anchored; 0/35 and 0/35 with moneyness bands). Highest observed daily inversion rate: 33% (delta), 33% (band).

Anchor agreement: on 298 observations where both anchors produced a skew, signs agree on 91%; median |band − delta| = 0.0206; correlation 0.85. Delta anchoring fell back to moneyness bands on 5% of observations.

Per name (where the daily inversion rate comes from):

| name | days | inverted % (delta) | inverted % (band) | median skew (delta) | delta fallback % |
|---|---|---|---|---|---|
| VXX | 4 | 100% | 100% | -0.2273 | 50% |
| GLD | 32 | 81% | 59% | -0.0095 | 0% |
| TSLA | 33 | 73% | 42% | -0.0051 | 0% |
| MSFT | 34 | 9% | 9% | +0.0156 | 3% |
| AAPL | 33 | 3% | 3% | +0.0142 | 3% |
| TLT | 32 | 3% | 22% | +0.0103 | 0% |
| IWM | 34 | 0% | 0% | +0.0384 | 0% |
| NVDA | 33 | 0% | 0% | +0.0223 | 3% |
| QQQ | 35 | 0% | 0% | +0.0448 | 3% |
| SPY | 34 | 0% | 0% | +0.0376 | 0% |
| XLE | 1 | n/a | 100% | n/a | 100% |
| XLF | 3 | n/a | 100% | n/a | 100% |
| XLK | 5 | n/a | 40% | n/a | 100% |

Concentration: GLD, TSLA, VXX are inverted on more than 50% of their days, so with ~9 names a day they set most of the daily rate on their own. Excluding them, the pooled delta-anchored inversion rate is 2%.

**Honest interpretation.** 35 market day(s) of history on 13 names (the `vol_snapshots` capture list, mostly index and sector ETFs) calibrates the metric's *range* in the captured (calm) regime. It does not calibrate the scanner's 20-name segment thresholds directly: none of the segments' own universes is in the history, and the daily rate here is a count over a few names. It cannot validate bubble-prediction power: that requires the thresholds to fire ahead of a drawdown at least once, i.e. years of daily history spanning at least one froth episode. Until then the 40%/60% thresholds remain judgment calls. The script recomputes from the full store; the last captured session is 2026-10-09.

Most inversions are shallow: the delta-anchored 10th percentile is under 1
vol point below zero, so a name sitting near zero can flip sign on bid-ask
noise. The inversion rate is a sign count and does not see this.

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
fetched with the chain is reused rather than re-scraped.

Each run fetches every distinct name once (the segments overlap: 220 slots,
179 distinct names) and shares the chain across segments, so a name has one
number per run. The GLOBAL row counts each distinct name once. Every row in
the snapshot CSV carries `anchor` (`delta`, `band` or `volume_weighted`),
the path that produced its skew. Results are only saved on a live trading
session: weekends are refused, and weekdays are checked against SPY's last
daily bar (same rule as `vol_snapshots/capture.py`). An off-session run still
prints, and `--force` saves it anyway. The legacy rows in `bubble_summary.csv`
(2025-10 to 2026-02) predate the own-IV inversion, the delta anchors and this
guard, and several fell on weekends, so they are not comparable with rows
written by the current code. Everything is
unit-tested offline on synthetic chains priced from known smiles:
`tests/test_iv_skew.py` (own-IV pipeline, delta anchoring, fallback, config
plumbing, idempotent snapshot save, summary companion columns, alert basis,
parity-implied q, q in delta anchoring, skew-source column, one fetch per
name, de-duplicated GLOBAL, session guard) and `tests/test_threshold_study.py`
(the study end-to-end on a fixture snapshot store, including stale-weekend
dedupe, parity-q replay and the per-name concentration table).
