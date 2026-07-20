# Options Toolkit

A small set of options-analytics tools: Black-Scholes pricing with the full
Greeks, a real-data implied-volatility skew scanner, and static no-arbitrage
checks. Built to explore how options markets price risk and where that pricing
breaks down.

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
- `Skew_surface_example.png` - sample output of `skew_surface()`.

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

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Known limitations
- **The vol surface is not arbitrage-free.** `skew_surface()` is a single-snapshot
  `griddata` interpolation of market IVs, not a fitted (SVI/SABR),
  calendar/butterfly-arbitrage-free surface.
- **`IV_skew.py` uses yfinance's own `impliedVolatility`** field rather than this
  repo's Newton-Raphson solver, and its skew thresholds are unvalidated
  heuristics. A delta-target config exists but is not yet wired in.
- **The arbitrage checks apply European relationships** (parity, box) to American
  yfinance options with **no dividend term**, so flagged trades can be spurious;
  realised edge is typically small relative to transaction costs. This is a
  teaching/diagnostic tool, not a live signal.

## Planned
- A quoting + inventory-skew + delta-hedge **market-making simulator** (Phase 2).
- A fitted **arbitrage-free vol surface** using this repo's own IV solver
  (Phase 3).

## Note
Research and learning code - not investment advice. Data is pulled live from
public sources (yfinance).
