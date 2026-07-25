# Daily option-chain snapshots

Free data has no history: yfinance serves **today's** option chains and
nothing else. Everything this repo wants to do next on surface *dynamics* —
how fitted eSSVI parameters move day to day, sticky-strike vs sticky-delta
behaviour, term-structure signals, event vol — needs a time series of chains
that no free source provides. The only way to have that dataset is to have
been capturing it. This folder makes it a one-command daily habit:

```bash
python vol_snapshots/capture.py                 # default: SPY QQQ IWM AAPL MSFT NVDA TSLA
python vol_snapshots/capture.py SPY TSLA        # explicit tickers
```

Each run writes one tidy `csv.gz` per ticker under `data/<YYYY-MM-DD>/`
(~1 MB/day for the default list) and is **idempotent per day** — already
captured tickers are skipped, so a scheduler double-fire is harmless.

## Scheduled daily run

`daily_capture.ps1` wraps the capture for Windows Task Scheduler: it runs
`capture.py`, appends all output to `vol_snapshots/capture.log` (gitignored),
and **commits the new data** — staging only `vol_snapshots/data`, so
work-in-progress elsewhere in the repo is never swept into an automated
commit, and skipping the commit entirely when nothing new arrived (weekends,
double-fires). Registered with:

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
* The `data/` folder is tracked in git and auto-committed by the scheduled
  run (~1 MB/day, compressed). If the repo ever gets heavy, the dataset can
  be migrated to its own repo or LFS without losing history — the point is
  that it exists somewhere durable from day one.
