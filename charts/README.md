# Charts

Every figure the toolkit generates, in one place. Regenerate with:

```bash
python market_making/mm_sim.py                   # -> charts/market_making/
python market_making/glft.py                     # -> charts/market_making/glft_quotes.png
python pricing_and_vol_surface/vol_surface.py    # -> charts/vol_surface/
```

All charts share the repo's colorblind-validated style (`plotstyle.py`).

## Vol surface (`pricing_and_vol_surface/vol_surface.py`)

Write-up: [`pricing_and_vol_surface/VOL_SURFACE.md`](../pricing_and_vol_surface/VOL_SURFACE.md)

**SSVI surface** — the fitted arbitrage-free surface.
![SSVI surface](vol_surface/ssvi_surface.png)

**Density check** — Durrleman `g(k) >= 0`: a naive spline admits butterfly
arbitrage (negative density) that the SSVI fit provably removes.
![density check](vol_surface/density_check.png)

**Slice fit** — per-expiry SVI against market quotes.
![slice fit](vol_surface/slice_fit.png)

**Weighted calibration** — vega/liquidity weights keep noisy illiquid wings
from dragging the fit off the reliable ATM quotes.
![weighted calibration](vol_surface/weighted_calibration.png)

**Surface history** — the SSVI band fit applied to each day's captured real
chains (`python vol_snapshots/fit_history.py`): ATM vol, skew `rho` and term
slope per ticker over time, arb-checked daily.
![surface history](vol_surface/surface_history.png)

**Realised-vs-implied replay** — short ATM straddles on the captured chains,
delta-hedged daily at entry IV (`python vol_snapshots/replay.py`): hedged P&L
paths against their gamma-P&L theory, and final P&L vs realised-minus-implied
vol as positions settle.
![replay](market_making/replay_realised_vs_implied.png)

**Band fit** — calibrating to the bid-ask interval instead of a point mid.
![band fit](vol_surface/band_fit.png)

## Market making (`market_making/mm_sim.py`, `glft.py`)

Write-up: [`market_making/README.md`](../market_making/README.md)

**Hedging validation (Experiment A)** — simulated delta-hedged P&L matches
the closed-form gamma-P&L identity.
![hedging validation](market_making/hedging_validation.png)

**P&L vs realised vol (Experiment B)** — spread capture is flat, vol P&L
slopes down through implied; the spread must pay for inventory vol risk.
![MM P&L vs vol](market_making/mm_pnl_vs_vol.png)

**Sample inventory path** — one book, mean-reverted by the skew.
![sample inventory path](market_making/sample_inventory_path.png)

**Adverse selection (Experiment C)** — directional toxicity costs through
hedge latency; a wider spread buys tolerance.
![adverse selection](market_making/adverse_selection.png)

**Vol-informed flow (Experiment D)** — vega toxicity survives instant
hedging; the defence is a vol-space markup with an interior optimum.
![vol-informed flow](market_making/vol_informed_flow.png)

**Online toxicity estimation (Experiment E)** — the desk infers directional
toxicity from its own fill markouts and widens adaptively.
![online toxicity](market_making/online_toxicity.png)

**Online vol-toxicity estimation (Experiment F)** — the vega-space markout
detects vol-informed flow and prices the markup on the excess over its null.
![online vol toxicity](market_making/online_vol_toxicity.png)

**GLFT quotes** — exact finite-horizon quotes fading to c0 near the terminal
time and converging to the stationary quotes; the closed form's
constant-spread + linear-skew claim against the exact answer.
![GLFT quotes](market_making/glft_quotes.png)

**GLFT vs hand-tuned quoting (Experiment G)** — inventory control is worth
~4 CE points over none under uncertain realised vol; one derived dial spans
the mean-risk frontier.
![GLFT vs static](market_making/glft_vs_static.png)

**Hawkes clustered flow (Experiment H)** — volume-matched self-exciting flow
raises inventory risk, and the hand-tuned-vs-derived desk ranking flips when
the independence assumption fails.
![Hawkes clustering](market_making/hawkes_clustering.png)
