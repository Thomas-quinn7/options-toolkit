# Options Toolkit

A collection of options-analytics tools — pricing, implied-volatility surfaces, skew-based signals, and arbitrage checks. Built to explore how options markets price risk and where that pricing breaks down.

## Contents

### `pricing_and_vol_surface/`
Black-Scholes pricing and implied-volatility analysis. Reverse-engineers implied volatility from market prices, plots the **volatility smile** across moneyness and time-to-maturity, and produces a heatmap of underlying price vs. implied volatility to visualise option value. (`Skew_surface_example.png` shows sample output.)

### `skew_bubble_indicator/`
Detects **inverted call skew** in major US firms as a possible bubble signal. Computes the skew between put and call prices; when calls price richer than puts across enough names, it flags the structure. Snapshots are saved to CSV (`daily_IV_skew_snapshot.csv`, `bubble_summary.csv`) for later analysis.

### `arbitrage/`
Basic no-arbitrage checks for options markets — put-call parity and related relationships. Surfaces candidate trades (note: realised return is typically small relative to transaction costs — this is a teaching/diagnostic tool).

### `weather_options/`
Weather-derivative experiments using free historical weather data (Open-Meteo, no API key required) as the underlying.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then run any tool's `main.py` (or the named script) from its folder.

## Note
Research and learning code — not investment advice. Data is pulled live from public sources (yfinance, Open-Meteo).
