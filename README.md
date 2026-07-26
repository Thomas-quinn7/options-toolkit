# Options Toolkit

![tests](https://github.com/Thomas-quinn7/options-toolkit/actions/workflows/tests.yml/badge.svg)

A small set of options-analytics tools: Black-Scholes pricing with the full
Greeks, a real-data implied-volatility skew scanner, an **arbitrage-free SVI/SSVI
vol surface**, static no-arbitrage checks, and a delta-hedged options
**market-making simulator**. Built to explore how options markets price risk and
where that pricing breaks down.

## Contents

### `pricing_and_vol_surface/`
- **`black.py`** - Black-Scholes pricing (calls/puts, with dividend yield), the
  full closed-form Greeks (delta, gamma, theta, vega, rho), and a Newton-Raphson
  implied-vol solver, built on JAX (JIT-compiled; the solver's derivative comes
  from autodiff). Pricing
  and the IV solver take spot and rate as parameters; `stock_data()` and
  `get_riskfree_rate()` are helpers for sourcing live inputs at the call site.
  Also includes a `price_heatmap()` (price/profit vs spot and vol) and a
  single-snapshot `skew_surface()` 3D plot. Importing the module has no side
  effects; the `skew_surface` demo runs only under `__main__`.
- **`main.py`** - A no-network smoke driver: prices a call/put, checks put-call
  parity, prints the Greeks, and runs an implied-vol round-trip
  (price -> implied vol -> price). Run `python main.py` from this folder.
- **`vol_surface.py`** - a real **arbitrage-free** IV surface: fits SVI per
  expiry and a global SSVI (Gatheral-Jacquier), and *proves* no butterfly
  arbitrage (Durrleman `g(k) >= 0`, i.e. non-negative density) and no calendar
  arbitrage (total variance rising with maturity). Demonstrates that a naive
  spline through noisy quotes admits butterfly arbitrage that SSVI removes.
  Supports **vega/liquidity-weighted calibration** so noisy illiquid wings don't
  drag the fit off the reliable ATM quotes, and **bid-ask band calibration**
  (`fit_svi_slice_band` per slice, `fit_ssvi_band` for the global surface) that
  fits the quoted interval instead of a point mid —
  the quote structure itself does the weighting. Ships its own Brent IV inverter
  for building surfaces from prices instead of yfinance's IV field (`IV_skew.py`
  uses it on live chains; the bundled demo fits synthetic quotes
  directly). See `pricing_and_vol_surface/VOL_SURFACE.md` for the write-up and
  figures. `Skew_surface_example.png` shows the older single-snapshot
  `skew_surface()` plot, kept for contrast.

```bash
python pricing_and_vol_surface/vol_surface.py    # fit, prove arb-free, write charts/vol_surface/
python -m pytest tests/test_vol_surface.py -q
```

### `skew_bubble_indicator/`
`IV_skew.py` scans a large set of US names across market segments, fetches option
chains concurrently, applies data-quality gates (volume, open interest, bid-ask
spread, IV sanity, DTE window), and computes the OTM put-minus-call IV skew per
name. Implied vols are **inverted from bid-ask mid prices by the repo's own
Brent solver** (`vol_surface.iv_from_price`); yfinance's `impliedVolatility`
field is kept only as a diagnostic column, and a mid outside the no-arbitrage
price bounds fails inversion and drops out — a free data-quality gate. Inverted
skew (calls richer than puts) across enough names is flagged as a
speculative-froth signal. Snapshots are appended to CSV
(`daily_IV_skew_snapshot.csv`, `bubble_summary.csv`). The pipeline's IV logic is
unit-tested offline on synthetic chains (`tests/test_iv_skew.py`).

```bash
python skew_bubble_indicator/IV_skew.py --workers 5 --plot
```

### `arbitrage/`
`arb_scan.py` runs static no-arbitrage checks on yfinance option chains. One
scanner, two modes:
- **Institutional (default)** - put-call parity, both-direction box spreads,
  negative-cost butterflies, and calendar monotonicity (assumes short selling).
- **Retail (`--retail`)** - checks a no-short account can act on: vertical
  monotonicity, buy-side underpriced boxes, negative-cost butterflies, and
  retail calendar mispricings.

```bash
python arbitrage/arb_scan.py AAPL MSFT            # institutional mode
python arbitrage/arb_scan.py AAPL --retail        # no short selling
```

### `market_making/`
`mm_sim.py` is a delta-hedged options market-making simulator. It quotes a
two-sided market around Black-Scholes fair value with Avellaneda-Stoikov-style
fill intensities and inventory skew, delta-hedges the resulting book, and
decomposes P&L into **spread capture** vs **vol / hedging P&L**. It shows that a
net-short desk's total P&L falls as realised vol rises above the implied vol it
quoted — the spread has to pay for the vol risk of the inventory taken on. The
hedging engine is validated against the closed-form Black-Scholes gamma-P&L
identity. It also models **adverse selection / toxic flow** in both kinds:
*directional* informed flow costs a delta-hedged desk through hedge latency
(so speed fixes it), while **vol-informed (vega-toxic) flow** — clients who buy
options precisely on the paths that will realise high vol — survives instant
hedging entirely and must be priced via a vol-space markup, which has an
interior optimum because it trades vega edge against volume. The desk also
**estimates both toxicity kinds online from its own fills**: a directional
markout (bias-corrected EWMA) drives adaptive spread widening that beats both
fixed policies when toxicity switches regime, and a vega-space markout (fills
scored against the next bars' realised variance, marked up only above a
calibrated null threshold) detects vol-informed flow — cleanly, though one
book's vega markout is noisy enough that per-book repricing recovers only part
of the oracle markup's edge, the honest asymmetry between the two kinds.
`glft.py` adds a **GLFT quoting engine** (Gueant-Lehalle-Fernandez-Tapia, the
tractable successor to Avellaneda-Stoikov): the exact finite-horizon solution
via the linearised HJB's matrix exponential, the exact stationary quotes from
its ground eigenvector, and the closed-form constant-spread-plus-linear-skew
approximation desks actually implement — cross-checked against each other in
tests. With `quote_policy="glft"` the simulator *derives* its spread and skew
from the fill model instead of hand-picking them, and Experiment G measures
what that buys under uncertain realised vol (inventory control is worth ~4 CE
points over none; risk aversion is the only free dial). Experiment H then
breaks the assumption every one of these models shares: fills arriving
**independently**. A volume-matched Hawkes mechanism makes flow self-excite
into same-side clusters (the realism gap the option-MM literature itself
flags — Baldacci 2020 §4.1.1), inventory excursions and P&L dispersion grow
with clustering, and the desk ranking **flips**: the hand-tuned quotes win in
the arena they were tuned for and lose to the derived conservative quotes
when the independence assumption fails. See `market_making/README.md` for the
write-up and figures. Charts across the repo share one colorblind-validated
style (`plotstyle.py`).

```bash
python market_making/mm_sim.py          # prints tables, writes charts/market_making/
python market_making/glft.py            # GLFT quote table + convergence figure
python -m pytest tests/test_mm.py tests/test_glft.py -q
```

### `vol_snapshots/`
Free data has no history, so `capture.py` builds the missing dataset one day
at a time: a one-command (or scheduled) capture of full option chains —
bid/ask **intervals** for the band calibration, spot, T-bill rate, volume/OI —
for a default list of liquid names, written as tidy `csv.gz` per ticker per
day, idempotent per day. This is the raw material for everything the vol
surface work wants next (eSSVI parameter dynamics, sticky-strike vs
sticky-delta, term-structure signals), none of which can be studied without a
time series that only accumulates if capture starts now. Schema and IO are
covered offline by `tests/test_snapshots.py`; see `vol_snapshots/README.md`.

`fit_history.py` is the daily consumer: it fits the repo's **SSVI band
surface** to every captured chain (forwards from put-call parity, IVs
inverted from prices, no-arb diagnostics recorded per fit) and accumulates
the fitted parameters — ATM vol, skew `rho`, term-structure slope — in
`surface_history.csv`, charted at `charts/vol_surface/surface_history.png`.
Raw chains live in a separate data repo (nested at `vol_snapshots/data/`) so
the growing dataset never bloats this one; the pipeline is tested offline on
a synthetic day from a known surface (`tests/test_fit_history.py`).

```bash
python vol_snapshots/capture.py         # capture today's chains (default list)
python vol_snapshots/fit_history.py     # fit surfaces, update the history
```

### `charts/`
Every generated figure lands here (`charts/market_making/`,
`charts/vol_surface/`), and [`charts/README.md`](charts/README.md) is a
one-page gallery of all of them with links back to each write-up.

## Testing

Alongside the per-module test files, `tests/test_properties.py` runs
**property-based tests** (hypothesis): put-call parity, price bounds and
monotonicities, IV round-trips, SVI derivative-vs-finite-difference checks,
and the Gatheral-Jacquier no-arbitrage guarantees — parameters satisfying the
sufficient butterfly conditions must produce a non-negative Durrleman density,
sampled all the way up to the conditions' boundary, plus the power-law
`eta (1+|rho|) <= 2` bound under which the surface must also be
calendar-free. The whole suite is offline:

```bash
python -m pytest tests/ -q
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Known limitations
- **`skew_surface()` (in `black.py`) is not arbitrage-free** - it is a
  single-snapshot `griddata` interpolation of market IVs. Use `vol_surface.py`
  for the fitted, butterfly/calendar-arbitrage-free SVI/SSVI surface;
  `skew_surface()` is kept only as the naive-interpolation contrast.
- **`IV_skew.py`'s skew/inversion thresholds are unvalidated heuristics** (the
  IVs themselves now come from the repo's own price inverter). A delta-target
  config exists but is not yet wired in.
- **The arbitrage scanner runs on delayed yfinance quotes.** Parity is now the
  American no-arbitrage band on executable (bid/ask-crossed) prices with a
  trailing-dividend term, so mid-price and dividend false positives are gone —
  but quotes can still be stale between the spot snapshot and each chain fetch,
  the reversal leg ignores stock-borrow cost, and the box/butterfly checks keep
  their European settlement logic. The scanner's live false-positive rate has
  never been measured. This is a teaching/diagnostic tool, not a live signal.

## Planned
- Pooling toxicity markouts across books/instruments in the MM simulator —
  the step that makes per-book-noisy vega toxicity actionable.
- Wiring `IV_skew.py`'s delta-target config in, and validating its inversion
  thresholds against the historical snapshots it has been accumulating.

## Note
Research and learning code - not investment advice. Data is pulled live from
public sources (yfinance).
