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

### Threshold study — run 2026-07-28, 2 distinct session close(s), 16 (day, name) observations

Per-day cross-section (the inversion rate is what the 40%/60% thresholds gate on):

| day | names | inverted % (delta) | inverted % (band) | median skew (delta) | median skew (band) | delta fallbacks |
|---|---|---|---|---|---|---|
| 2026-07-25 | 7 | 14% | 14% | +0.0816 | +0.1357 | 0 |
| 2026-07-27 | 9 | 0% | 0% | +0.0342 | +0.0584 | 0 |

Pooled per-name skew distribution (vol points, put minus call):

| metric | n | min | p10 | p25 | p50 | p75 | p90 | max | share inverted |
|---|---|---|---|---|---|---|---|---|---|
| delta-anchored | 16 | -0.1588 | +0.0011 | +0.0246 | +0.0541 | +0.0729 | +0.1221 | +0.1377 | 6% |
| moneyness-band | 16 | -0.0798 | +0.0022 | +0.0296 | +0.0894 | +0.1357 | +0.1414 | +0.1800 | 6% |

Threshold context: the WARNING (>40%) / CRITICAL (>60%) inversion thresholds
would have fired on 0/2 and 0/2 day(s) (delta-anchored; 0/2 and 0/2 with
moneyness bands). Highest observed daily inversion rate: 14% (delta), 14%
(band).

Anchor agreement: on 16 observations where both anchors produced a skew,
signs agree on 100%; median |band − delta| = 0.0210; correlation 0.90. Delta
anchoring fell back to moneyness bands on 0% of observations.

The single inverted observation is itself instructive: AAPL on 2026-07-25
(pre-earnings call demand; the stock then gapped +4.7% on the next session).
The delta anchors read it at −0.16 vs the band's −0.08 — the call bid
concentrated nearer the money, exactly where delta anchoring looks.

**Honest interpretation.** 2 market days of history calibrates the metric's
*range* in one calm regime — where "normal" sits (median per-name skew ~+0.03
to +0.09, daily inversion rates 0–14%) and how far below the thresholds that
is. It **cannot validate bubble-prediction power**: that requires the
thresholds to fire ahead of a drawdown at least once, i.e. years of daily
history spanning at least one froth episode. Until then the 40%/60%
thresholds remain heuristics — now with measured context instead of none.
The study is stateless and recomputes from the full data store, so the same
command strengthens the table as the nightly capture
(`vol_snapshots/daily_capture.ps1`) accrues; re-run it and refresh this
section as the history grows.

## Outputs and tests

Snapshots append to `daily_IV_skew_snapshot.csv` / `bubble_summary.csv`; the
study writes per-observation rows to `threshold_study.csv`. Everything is
unit-tested offline on synthetic chains priced from known smiles:
`tests/test_iv_skew.py` (own-IV pipeline, delta anchoring, fallback, config
plumbing) and `tests/test_threshold_study.py` (the study end-to-end on a
fixture snapshot store, including stale-weekend dedupe).
