# Skew bubble indicator: code guide

Written 2026-11-13. Covers the code only. Results and caveats are in `README.md` next to this file.

## What it does

For each US stock or ETF it measures the implied volatility (IV) skew: the IV of an out-of-the-money put minus the IV of an out-of-the-money call, on one expiry near 30 days. Normally puts are richer, so the number is positive. A negative number is called an inverted skew, meaning calls are richer than puts. The scanner counts what share of each market segment is inverted and raises an alert level when that share is high. The idea is that widespread call-heavy pricing is a sign of speculative froth. Nothing has tested whether that idea predicts anything.

## Files

| File | Role |
|---|---|
| `skew_bubble_indicator/IV_skew.py` | The live scanner. Fetches chains, computes skew per name, scores segments, saves snapshots. About 1350 lines. |
| `skew_bubble_indicator/threshold_study.py` | Replays the scanner's own pipeline over saved history to see where the alert thresholds sit. Prints the README table. |
| `skew_bubble_indicator/bubble_summary.csv` | One row per segment per day. Rows before 2026-02 are from old code and not comparable. |
| `skew_bubble_indicator/threshold_study.csv` | One row per (day, name) from the last study run. |
| `pricing_and_vol_surface/vol_surface.py` | `iv_from_price`, the Brent solver that turns a mid price into an IV. |
| `pricing_and_vol_surface/black.py` | `greeks`, used to find the 25-delta strikes. |
| `pricing_and_vol_surface/american.py` | `american_iv_vector`, only used with `--american`. |
| `vol_snapshots/data/<date>/<TICKER>.csv.gz` | Captured chains the study reads. Written by `vol_snapshots/capture.py` and `auto_capture.py`. |
| `tests/test_iv_skew.py`, `tests/test_threshold_study.py` | Offline tests on synthetic chains. |

The scanner needs `yfinance`, `pandas`, `numpy` and `matplotlib` (plotting only). The three `pricing_and_vol_surface` modules above must be importable, which they are from a clone.

## Parts of `IV_skew.py`, in pipeline order

1. **Universe.** `GLOBAL_INDICES` holds the segments, about 20 names each. 220 slots cover 179 distinct names.
2. **Session guard.** `market_session_status` refuses to save on weekends or when SPY has no daily bar for today. `--force` overrides it.
3. **Fetch.** `fetch_all_chains` fetches each distinct name once with a thread pool. `fetch_option_chain` picks the expiry with `get_best_expiry` (valid range 7 to 60 days, closest to 30), runs `validate_option_data`, and attaches the spot price and the dividend yield to the chain.
4. **Own IVs.** `add_own_iv` inverts IV from the bid-ask mid with `iv_from_price`. The yfinance IV is kept as a diagnostic only. Mids outside the no-arbitrage bounds get no IV.
5. **Dividend yield.** `parity_implied_q` infers q from put-call parity on matched strikes. If the implied carry is absurd (above `Q_PARITY_MAX`, 0.15) it falls back to 0.
6. **Quality.** `compute_data_quality_score` rates each chain. A name with a score of 0.5 or more counts as reliable.
7. **Strike anchors.** `get_delta_anchored_strike_targets` finds the listed strikes nearest 25-delta put and 25-delta call, using Black-Scholes delta at each quote's own IV. If either side has no strike within 0.10 of the target delta, `get_otm_strike_targets` takes over and uses 90% and 110% of spot for both sides. It never mixes the two.
8. **Skew number.** `compute_iv_skew_metric` takes the mean IV near each anchor strike (`find_mean_iv_at_strike`, a window that widens up to 15%) and returns put IV minus call IV. It also records which path produced the number: `delta`, `band` or `volume_weighted`.
9. **Segment scoring.** `analyze_index` builds one row per name. `alert_inversion_rate` gives the share of names with negative skew, using only reliable names when there are at least 5 of them. The share maps to a level: 25% CAUTION, 40% WARNING, 60% CRITICAL, 70% EXTREME, otherwise NORMAL.
10. **Global run.** `run_global_iv_skew_analysis` ties it together. The GLOBAL row counts each distinct name once.
11. **Saving.** `save_iv_skew_snapshot` and `save_bubble_summary_by_index` replace the same day's rows if you rerun, so a day is never counted twice.

## How `threshold_study.py` works

It reads every saved chain in `vol_snapshots/data/`, drops stale weekend copies (`dedupe_stale_days`), picks the expiry nearest 30 days (`pick_expiry`) and runs the same gates, IV inversion, parity q and anchoring as the scanner (`ticker_day_row`). It then summarises per day, pooled, per name and anchor agreement, and counts how often the 40% and 60% lines would have fired. `format_markdown` prints the block that goes in the README.

## Running it

```
python skew_bubble_indicator/IV_skew.py --workers 5 --plot
python skew_bubble_indicator/IV_skew.py --anchor moneyness
python skew_bubble_indicator/threshold_study.py
python -m pytest tests/test_iv_skew.py tests/test_threshold_study.py
```

## Known gaps

- The fixed scanner has not yet been run live and saved. `bubble_summary.csv` has only old rows.
- The thresholds are judgment calls. They have not fired once in the 42 sessions of history, and the history covers 13 names, not the scanner's segments.
- A few names (GLD, TSLA, VXX) are inverted most days, so they dominate any small segment.
- Nothing links the signal to later drawdowns.
