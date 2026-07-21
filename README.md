# Options Toolkit

A small set of options-analytics tools: Black-Scholes pricing with the full
Greeks, a real-data implied-volatility skew scanner, an **arbitrage-free SVI/SSVI
vol surface**, static no-arbitrage checks, and a delta-hedged options
**market-making simulator**. Built to explore how options markets price risk and
where that pricing breaks down.

## Contents

### `pricing_and_vol_surface/`
- **`black.py`** - Black-Scholes pricing (calls/puts, with dividend yield), the
  full Greeks (delta, gamma, theta, vega, rho), and a Newton-Raphson implied-vol
  solver, all built on JAX (JIT + `vmap` batching, Greeks via autodiff). Pricing
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
  (`fit_svi_slice_band`) that fits the quoted interval instead of a point mid —
  the quote structure itself does the weighting. Builds surfaces from prices via
  its own Brent IV inverter, not yfinance's IV
  field. See `pricing_and_vol_surface/VOL_SURFACE.md` for the write-up and
  figures. `Skew_surface_example.png` shows the older single-snapshot
  `skew_surface()` plot, kept for contrast.

```bash
python pricing_and_vol_surface/vol_surface.py    # fit, prove arb-free, write figures/
python -m pytest tests/test_vol_surface.py -q
```

### `skew_bubble_indicator/`
`IV_skew.py` scans a large set of US names across market segments, fetches option
chains concurrently, applies data-quality gates (volume, open interest, bid-ask
spread, IV sanity, DTE window), and computes the OTM put-minus-call IV skew per
name. Inverted skew (calls richer than puts) across enough names is flagged as a
speculative-froth signal. Snapshots are appended to CSV
(`daily_IV_skew_snapshot.csv`, `bubble_summary.csv`).

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
interior optimum because it trades vega edge against volume. See
`market_making/README.md` for the write-up and figures.

```bash
python market_making/mm_sim.py          # prints tables, writes figures/
python -m pytest tests/test_mm.py -q    # validates the hedging engine
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
- **`IV_skew.py` uses yfinance's own `impliedVolatility`** field rather than this
  repo's Newton-Raphson solver, and its skew thresholds are unvalidated
  heuristics. A delta-target config exists but is not yet wired in.
- **The arbitrage checks apply European relationships** (parity, box) to American
  yfinance options with **no dividend term**, so flagged trades can be spurious;
  realised edge is typically small relative to transaction costs. This is a
  teaching/diagnostic tool, not a live signal.

## Planned
- Online toxicity estimation in the market-making simulator: infer the informed
  fraction from the desk's own fill stream and adapt the spread/markup.
- Wiring the bid-ask band residual into the global SSVI fit (it is per-slice
  today).

## Note
Research and learning code - not investment advice. Data is pulled live from
public sources (yfinance).
