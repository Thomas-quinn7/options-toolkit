# Daily option-chain snapshots

Free data has no history: yfinance serves **today's** option chains and
nothing else. Everything this repo wants to do next on surface *dynamics* —
how fitted eSSVI parameters move day to day, sticky-strike vs sticky-delta
behaviour, term-structure signals, event vol — needs a time series of chains
that no free source provides. The only way to have that dataset is to have
been capturing it. This folder makes it a one-command daily habit:

```bash
python vol_snapshots/capture.py                 # default: 13-name universe below
python vol_snapshots/capture.py SPY TSLA        # explicit tickers
```

Each run writes one tidy `csv.gz` per ticker under `data/<YYYY-MM-DD>/`
(~1.5 MB/day for the default list) and is **idempotent per day** — already
captured tickers are skipped, so a scheduler double-fire is harmless.

The default universe (13 names): index ETFs (SPY, QQQ, IWM), mega-cap singles
with earnings vol (AAPL, MSFT, NVDA, TSLA), sector rotation (XLF, XLE, XLK),
rates (TLT), gold (GLD), and the vol complex itself (VXX). The macro complex
was added 2026-07-26 — data not captured now never exists later. The history
chart shows a fixed-priority subset of at most 8 series (the palette's slot
count); `surface_history.csv` always records every fitted name.

## Fitted surface history — what the snapshots are for

`fit_history.py` turns each captured (day, ticker) chain into one row of
`surface_history.csv`: forwards implied from put-call parity, IVs inverted
from bid/ask **prices** with the repo's own inverter, OTM quotes only, and a
global **SSVI band fit** whose no-arbitrage diagnostics (min Durrleman `g`,
calendar gap) are recorded in the row — a day the fit could *not* be made
arb-free is visible in the data, not assumed away. The accumulated history
(ATM vol, skew `rho`, term-structure slope per ticker over time) is charted to
`charts/vol_surface/surface_history.png`.

```bash
python vol_snapshots/fit_history.py             # fit all unfitted days
python vol_snapshots/fit_history.py --refit     # refit everything
```

The whole pipeline is pinned offline by `tests/test_fit_history.py` on a
synthetic day priced from a known SSVI surface.

## Realised-vs-implied replay — the sim's headline result, on real chains

`replay.py` runs the market-making simulator's central claim through the real
history: one short ATM straddle per ticker (25-45 DTE, entered at the quoted
mid, entry IV inverted from the traded prices), **delta-hedged at the entry
implied vol every snapshot day**, financing accrued, settled at expiry — then
immediately re-entered, so positions chain through the history. Each position
carries its own discrete gamma-P&L theory accrual, making the output
Experiment A's sim-vs-theory comparison on market data: as positions settle,
final P&L should line up against realised-minus-entry-implied vol.

```bash
python vol_snapshots/replay.py               # all tickers -> replay_history.csv + chart
```

Stated conventions: entry/mark at mid (the entry half-spread is recorded as
the execution cost a real desk would pay); one snapshot a day means daily
hedging, so single-position hedging noise is material and conclusions come
from the accumulating cross-section; stale weekend/holiday captures (identical
spot) are dropped. Stateless — every run recomputes from the raw data store.
Pinned offline by `tests/test_replay.py` on GBM histories with known
realised/implied vols (sign both ways, theory tracking, multi-expiry marking).

## Where the data lives

Raw chains would grow the public repo by ~1 MB/day forever, so they live in
their own repo: **`vol_snapshots/data/` is a nested git checkout of
[`options-toolkit-data`](https://github.com/Thomas-quinn7/options-toolkit-data)**
(private), gitignored by the toolkit repo. The toolkit repo carries only the
small derived artifacts — `surface_history.csv` and the history chart. The
scheduled run pushes the raw data daily so the dataset stays durable
off-machine; if the nested repo is ever missing, the script bootstraps it
automatically (and logs loudly if it cannot).

## Scheduled daily run

`daily_capture.ps1` wraps the daily habit for Windows Task Scheduler:
capture → commit + push raw chains in the nested data repo → fit the day's
surfaces → commit `surface_history.csv` + chart to the toolkit repo. All
output is appended to `vol_snapshots/capture.log` (gitignored); only the
named paths are ever staged, so work-in-progress is never swept into an
automated commit. Registered with:

```
schtasks /create /tn options-toolkit-snapshot /sc daily /st 22:00 /tr ^
    "powershell -NoProfile -ExecutionPolicy Bypass -File R:\Quant_projects\options-toolkit\vol_snapshots\daily_capture.ps1"
```

22:00 local covers the US close year-round from Ireland (21:00 both in
summer, BST/EDT, and winter, GMT/EST). If the machine is off at 22:00 the
task fires on the next boot only if you enable "Run task as soon as possible
after a scheduled start is missed" in Task Scheduler's UI — worth doing, and
the capture stays valid all evening since yfinance serves the session's
closing quotes until the next open. Check `capture.log` if a day looks
missing.

## Schema

One row per (ticker, expiry, type, strike):

| column | meaning |
|---|---|
| `snapshot_date` | capture date (ISO) |
| `ticker`, `spot` | underlying and its price at capture |
| `riskfree` | 13-week T-bill yield (^IRX) at capture, decimal |
| `expiry`, `otype`, `strike` | contract identity |
| `bid`, `ask`, `last` | the quoted **interval**, not just a mid — the repo's band calibration (`fit_svi_slice_band` / `fit_ssvi_band`) fits exactly this |
| `volume`, `open_interest` | liquidity gates for the consumer to apply |
| `iv_yf` | yfinance's IV, kept **only for cross-checking** — surfaces are built from prices via the repo's own inverter |

Rows with one-sided or empty quotes are kept: the capture's job is fidelity,
the liquidity decision belongs to the analysis that consumes it.

```python
from vol_snapshots.capture import load_snapshots
df = load_snapshots()                    # every day, every ticker, one frame
```

`normalize_chain` (raw yfinance frame → schema) and the save/load round-trip
are pure and covered offline by `tests/test_snapshots.py`; only the fetch
itself touches the network.

## Honest limitations

* **One snapshot per day, after the close** — nothing intraday can ever be
  claimed from this dataset, and quotes captured out of market hours are the
  session's last, which can be stale in the wings.
* yfinance is an unofficial API; per-expiry failures are caught and logged
  rather than killing the batch, but a schema change upstream would need a
  `normalize_chain` update (the offline tests pin the output schema).
* Fits need enough usable quotes: a (day, ticker) with fewer than 3 slices
  passing the liquidity gates is skipped and simply has no history row — check
  `capture.log` if a name goes quiet.
