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

A run on a **non-trading day refuses to capture** unless given `--force`,
because what yfinance serves then is the previous session's after-close
marks: stale mids that once implied a −37% AAPL dividend yield downstream.
Whether today is a session is decided by asking if a daily bar exists for it,
not from a holiday table — so market holidays (a holiday is a *weekday*, and
a weekend-only guard would have written a stale Labor Day snapshot) and
unscheduled closures are both covered, and there is no table to keep current.

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

Two gates are worth knowing about, because both were tightened by what real
chains actually did:

- **Stale-quote gate on the forward basis, not on `q`.** A forward far from
  spot means stale marks, but `q = r - ln(F/S)/T` is a *continuous*-yield
  reading, so one discrete dividend inside a short window annualises into a
  huge `|q|` even when the quotes are perfect — TLT's monthly distribution
  showed up as `|q| = 15%` on the 16d slice. Bounding `|ln(F/S) - rT|` instead
  keeps the staleness test without the `1/T` blow-up that was silently
  deleting every near-dated expiry of the dividend payers, and with them the
  30d ATM vol the history exists to record.
- **`fit_ok`** marks rows whose fit *collapsed* — `|rho|` on the optimiser's
  bound, or total variance falling with maturity by more than half a theta.
  These are still written to the CSV with their diagnostics intact; the
  dynamics studies just refuse to read them as observations. Thin chains
  (XLF, XLK) are where this fires: when only long-dated slices survive, the
  fit has nothing pinning the short end and runs to its bound.

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

## Sticky-strike vs sticky-delta — armed, waiting for data

`sticky.py` answers Derman's smile-regime question from the daily fits: after
a spot move, does a fixed strike keep its vol (sticky-strike) or does the
smile move with moneyness (sticky-delta, which adds a term to every option's
effective delta)? Each consecutive-day pair of fitted 30d smiles is compared
against both predictions; the **demeaned RMS** of each prediction is the
primary discriminator (immune to the day's parallel vol change), with the
classic regime **beta** reported alongside — including its stated caveat that
spot-vol correlation biases beta upward even in a sticky-strike world.

The study gates itself on `MIN_PAIRS` usable day-pairs per ticker (consecutive
fits with a >= 0.2% move). Until then the scheduled run just logs what it is
waiting for; once the history is long enough, `sticky_summary.csv` and
`charts/vol_surface/sticky_regimes.png` appear on their own. Core comparisons
are pure functions pinned by `tests/test_sticky.py` on exactly-synthesised
regimes of both kinds.

```bash
python vol_snapshots/sticky.py               # runs, or says what's missing
```

## Event vol — armed, waiting for earnings season

`event_vol.py` extracts the three phases of a scheduled vol event from the
daily fits: the build-up (short-dated ATM vol inflating), the term-structure
inversion, and the crush. **Events are detected endogenously** — a one-day
30d-ATM drop of >= 3 vol points and >= 10% of the pre-day level — so no
earnings calendar is imported and the claim stays exactly what the surface
shows. Each event records the pre-event **implied event move** (excess
short-dated variance over the 182d base, square-rooted: what the options
market charged for the announcement) against the **realized** crush-day move —
one implied-vs-realized point per event. Gated until the first event lands
(the August earnings cycle for AAPL/MSFT/NVDA/TSLA is already in scope);
outputs `event_vol.csv` + `charts/vol_surface/event_vol.png`.

## Skew dynamics — armed, accumulating

`skew_dynamics.py` tests the two classic surface-dynamics facts on the fits:
**skew steepens when the market falls** (daily changes in the fixed-moneyness
30d skew regressed on SPY's return — expected beta < 0) and the **leverage
effect** (a name's ATM vol vs its own return — expected beta < 0), reported
in vol points per 1% move with 2-s.e. bars. Gated at 15 matched day-pairs per
ticker; outputs `skew_dynamics.csv` + `charts/vol_surface/skew_dynamics.png`.

Both studies (like `sticky.py`) run in the scheduled task every night and stay
silent apart from an "accumulating" log line until their data threshold trips.
Pinned offline by `tests/test_event_vol.py` (injected synthetic event + flat
control) and `tests/test_skew_dynamics.py` (known injected dynamics).

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
