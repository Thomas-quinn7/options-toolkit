"""Options market-making simulator.

A delta-hedged options market-maker on a single European option. The simulator
demonstrates the two P&L engines a real options MM runs on:

  1. Spread capture - the edge earned quoting a two-sided market around the
     theoretical value and getting filled by incoming order flow.
  2. Vol / hedging P&L - the gamma P&L on the *net inventory* the desk is left
     holding, which is locked in by delta-hedging and is governed by the gap
     between realised and implied volatility.

The headline result: for a desk that absorbs one-sided client flow (and so runs
net short options), total P&L falls as realised vol rises above the implied vol
it quoted at - the spread has to be wide enough to pay for the vol risk of the
inventory taken on.

Design notes
------------
* Pricing here is a small *vectorised* closed-form Black-Scholes (numpy/scipy)
  so a whole fan of Monte-Carlo paths prices in one call. The repo's
  interactive / autodiff pricer is ``pricing_and_vol_surface/black.py``;
  ``tests/test_mm.py`` cross-checks this module's price/delta/gamma against it.
* The desk hedges at the *implied* vol delta (standard "hedge at the vol you
  marked at"). This makes the discrete-hedge P&L converge to the classic
  identity  0.5 * integral( Gamma_impl * S^2 * (sigma_impl^2 - sigma_real^2) )
  for a short position, which ``experiment_hedging_validation`` checks.
* Order arrival uses an Avellaneda-Stoikov style intensity, lambda = A*exp(-k*d)
  where d is the quote's distance from fair. Inventory is controlled by skewing
  the reservation price, which asymmetrically changes the two fill intensities.
* r is defaulted to 0 in the experiments to isolate the vol P&L from financing
  and discounting; it is a supported parameter throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Vectorised closed-form Black-Scholes (numpy). Handles tau <= 0 (expiry).     #
# --------------------------------------------------------------------------- #
def _d1_d2(S, K, tau, r, sigma, q=0.0):
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return d1, d2


def bs_price(S, K, tau, r, sigma, q=0.0, otype="call"):
    """Black-Scholes price. Vectorised over S (and scalars). tau<=0 -> intrinsic."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    alive = tau > 0
    tau_safe = np.where(alive, tau, 1.0)  # avoid div-by-zero; masked out below
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1_d2(S, K, tau_safe, r, sigma, q)
        if otype == "call":
            live = S * np.exp(-q * tau_safe) * norm.cdf(d1) - K * np.exp(-r * tau_safe) * norm.cdf(d2)
            intrinsic = np.maximum(S - K, 0.0)
        elif otype == "put":
            live = K * np.exp(-r * tau_safe) * norm.cdf(-d2) - S * np.exp(-q * tau_safe) * norm.cdf(-d1)
            intrinsic = np.maximum(K - S, 0.0)
        else:
            raise ValueError("otype must be 'call' or 'put'")
    return np.where(alive, live, intrinsic)


def bs_delta(S, K, tau, r, sigma, q=0.0, otype="call"):
    """Black-Scholes delta. tau<=0 -> step delta (0/1 boundaries)."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    alive = tau > 0
    tau_safe = np.where(alive, tau, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1_d2(S, K, tau_safe, r, sigma, q)
        if otype == "call":
            live = np.exp(-q * tau_safe) * norm.cdf(d1)
            expired = (S > K).astype(float)
        elif otype == "put":
            live = np.exp(-q * tau_safe) * (norm.cdf(d1) - 1.0)
            expired = -(S < K).astype(float)
        else:
            raise ValueError("otype must be 'call' or 'put'")
    return np.where(alive, live, expired)


def bs_gamma(S, K, tau, r, sigma, q=0.0):
    """Black-Scholes gamma (same for calls and puts). tau<=0 -> 0."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    alive = tau > 0
    tau_safe = np.where(alive, tau, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1_d2(S, K, tau_safe, r, sigma, q)
        g = np.exp(-q * tau_safe) * norm.pdf(d1) / (S * sigma * np.sqrt(tau_safe))
    return np.where(alive, g, 0.0)


# --------------------------------------------------------------------------- #
# Simulator                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class MMParams:
    S0: float = 100.0
    K: float = 100.0
    T: float = 0.25                 # years to expiry
    r: float = 0.0                  # rate (0 isolates the vol P&L)
    q: float = 0.0                  # dividend yield
    sigma_impl: float = 0.20        # vol the desk quotes / marks / hedges at
    otype: str = "call"
    n_steps: int = 126              # hedge/quote steps to expiry (~ daily over 0.25y)

    # quoting
    half_spread: float = 0.15       # base edge each side, in price units
    skew_coef: float = 0.02         # reservation shift per contract of inventory
    A: float = 250.0                # base arrival intensity at fair (fills / year)
    k: float = 8.0                  # intensity decay per unit quote distance
    flow_imbalance: float = 0.0     # >0 => clients net buyers (lift our ask)
    max_inventory: int = 25         # hard cap on |inventory|

    # adverse selection
    toxicity: float = 0.0           # fraction of flow informed about DIRECTION (0..1)
    vol_toxicity: float = 0.0       # fraction of flow informed about the VOL REGIME (0..1)
    hedge_lag: int = 0              # 0 = hedge before the move; 1 = hedge after it

    # quoting defence against vol-informed flow: half-spread in VOL space.
    # Asks are priced at sigma_impl + vol_spread, bids at sigma_impl - vol_spread,
    # so the quote automatically charges more vega edge where vega is high.
    vol_spread: float = 0.0

    # hedging
    tc_underlying: float = 0.0      # per-share hedge cost as fraction of notional

    contract_multiplier: float = 1.0


def simulate_paths(
    params: MMParams,
    sigma_real: float,
    n_sims: int,
    rng: np.random.Generator,
    quoting: bool = True,
    init_position: int = 0,
):
    """Vectorised Monte-Carlo over ``n_sims`` paths.

    ``sigma_real`` may be a scalar (one realised vol for all paths) or an array
    of shape (n_sims,) giving each path its own realised vol - the latter is
    what makes vol-informed ("vega-toxic") flow expressible: informed clients
    can condition on which vol regime their path is in.

    Returns a dict of per-path arrays (shape (n_sims,)) plus a couple of
    representative time series for plotting a single path.
    """
    p = params
    if p.toxicity + p.vol_toxicity > 1.0:
        raise ValueError("toxicity + vol_toxicity must be <= 1 (they partition the flow)")
    m = p.contract_multiplier
    dt = p.T / p.n_steps
    sqrt_dt = np.sqrt(dt)
    sigma_real = np.broadcast_to(np.asarray(sigma_real, dtype=float), (n_sims,))

    S = np.full(n_sims, p.S0, dtype=float)
    q_inv = np.full(n_sims, init_position, dtype=float)   # option inventory (contracts)
    H = np.zeros(n_sims)                                  # underlying hedge (shares)
    cash = np.zeros(n_sims)
    spread_capture = np.zeros(n_sims)
    vol_theory = np.zeros(n_sims)                         # analytic gamma-P&L accrual
    fills = np.zeros(n_sims)

    # If we start with a position, book it as executed at fair (zero edge) and
    # hedge it, so the static-position experiment is a pure vol-P&L test.
    if init_position != 0:
        theo0 = bs_price(S, p.K, p.T, p.r, p.sigma_impl, p.q, p.otype)
        cash -= init_position * theo0 * m
        d0 = bs_delta(S, p.K, p.T, p.r, p.sigma_impl, p.q, p.otype)
        H_target = -init_position * d0 * m
        cash -= (H_target - H) * S + p.tc_underlying * np.abs(H_target - H) * S
        H = H_target

    A_ask = p.A * (1.0 + p.flow_imbalance)   # client buys lift our ask
    A_bid = p.A * (1.0 - p.flow_imbalance)

    inv_track = np.zeros(p.n_steps + 1)      # mean inventory across paths, per step
    S_track = np.zeros(p.n_steps + 1)        # one sample path of the underlying
    inv_sample = np.zeros(p.n_steps + 1)     # inventory of that same sample path
    inv_track[0] = q_inv.mean()
    S_track[0] = S[0]
    inv_sample[0] = q_inv[0]

    var_gap = p.sigma_impl**2 - sigma_real**2

    for t in range(p.n_steps):
        tau = p.T - t * dt
        theo = bs_price(S, p.K, tau, p.r, p.sigma_impl, p.q, p.otype)

        # Draw this step's underlying shock up front so informed ("toxic") flow
        # can arrive on the side the imminent move will favour.
        z = rng.standard_normal(n_sims)

        if quoting:
            # Optional vol-space half-spread: price the ask at a marked-up vol
            # and the bid at a marked-down vol. Near expiry vega -> 0 and the
            # vol spread collapses naturally, as it should.
            if p.vol_spread > 0.0:
                ask_base = bs_price(S, p.K, tau, p.r, p.sigma_impl + p.vol_spread, p.q, p.otype)
                bid_base = bs_price(S, p.K, tau, p.r, max(p.sigma_impl - p.vol_spread, 1e-4),
                                    p.q, p.otype)
            else:
                ask_base = bid_base = theo
            skew_shift = p.skew_coef * q_inv
            bid = bid_base - skew_shift - p.half_spread
            ask = ask_base - skew_shift + p.half_spread
            d_bid = theo - bid          # distance from fair (drives fill intensity)
            d_ask = ask - theo

            lam_bid = A_bid * np.exp(-p.k * d_bid)
            lam_ask = A_ask * np.exp(-p.k * d_ask)
            prob_bid = np.clip(1.0 - np.exp(-lam_bid * dt), 0.0, 1.0)
            prob_ask = np.clip(1.0 - np.exp(-lam_ask * dt), 0.0, 1.0)

            tox = p.toxicity
            vtox = p.vol_toxicity
            # Uninformed flow: direction independent of the coming move.
            u_bid = rng.random(n_sims) < (1.0 - tox - vtox) * prob_bid
            u_ask = rng.random(n_sims) < (1.0 - tox - vtox) * prob_ask
            # Direction-informed flow: buys (lifts our ask) when the underlying
            # is about to rise, sells (hits our bid) when it is about to fall.
            up = z > 0.0
            i_ask = up & (rng.random(n_sims) < tox * prob_ask)
            i_bid = (~up) & (rng.random(n_sims) < tox * prob_bid)
            # Vol-informed flow: an option is long vega on either side, so vol
            # buyers lift our ask on paths whose realised vol will exceed the
            # implied they pay, and sell options to us on low-vol paths.
            # Delta-hedging cannot neutralise this - it selects which vol
            # regime each side of our book rides.
            hi_vol = sigma_real > p.sigma_impl
            v_ask = hi_vol & (rng.random(n_sims) < vtox * prob_ask)
            v_bid = (~hi_vol) & (rng.random(n_sims) < vtox * prob_bid)

            fill_bid = u_bid | i_bid | v_bid    # we BUY 1 @ bid
            fill_ask = u_ask | i_ask | v_ask    # we SELL 1 @ ask
            fill_bid &= q_inv < p.max_inventory
            fill_ask &= q_inv > -p.max_inventory

            q_inv += fill_bid
            cash -= fill_bid * bid * m
            spread_capture += fill_bid * (theo - bid) * m
            q_inv -= fill_ask
            cash += fill_ask * ask * m
            spread_capture += fill_ask * (ask - theo) * m
            fills += fill_bid + fill_ask

        # Hedge BEFORE the move (delta-neutral into it) unless a hedge lag is set,
        # in which case newly-acquired inventory rides the move unhedged - the
        # channel through which informed flow actually costs a hedged desk.
        if p.hedge_lag == 0:
            delta_opt = bs_delta(S, p.K, tau, p.r, p.sigma_impl, p.q, p.otype)
            H_target = -q_inv * delta_opt * m
            dH = H_target - H
            cash -= dH * S + p.tc_underlying * np.abs(dH) * S
            H = H_target

        # analytic gamma P&L accrued over [t, t+dt] on the carried inventory
        gamma_opt = bs_gamma(S, p.K, tau, p.r, p.sigma_impl, p.q)
        vol_theory += 0.5 * q_inv * gamma_opt * S**2 * (sigma_real**2 - p.sigma_impl**2) * dt * m

        # evolve the underlying with the REALISED vol using the pre-drawn shock
        S = S * np.exp((p.r - p.q - 0.5 * sigma_real**2) * dt + sigma_real * sqrt_dt * z)

        # Hedge AFTER the move if a lag is configured (adverse-selection channel).
        if p.hedge_lag == 1:
            delta_opt = bs_delta(S, p.K, tau, p.r, p.sigma_impl, p.q, p.otype)
            H_target = -q_inv * delta_opt * m
            dH = H_target - H
            cash -= dH * S + p.tc_underlying * np.abs(dH) * S
            H = H_target

        inv_track[t + 1] = q_inv.mean()
        S_track[t + 1] = S[0]
        inv_sample[t + 1] = q_inv[0]

    # expiry: settle options at intrinsic, liquidate the hedge
    if p.otype == "call":
        intrinsic = np.maximum(S - p.K, 0.0)
    else:
        intrinsic = np.maximum(p.K - S, 0.0)
    cash += q_inv * intrinsic * m
    cash += H * S - p.tc_underlying * np.abs(H) * S

    total_pnl = cash
    vol_hedge_pnl = total_pnl - spread_capture

    return {
        "total_pnl": total_pnl,
        "spread_capture": spread_capture,
        "vol_hedge_pnl": vol_hedge_pnl,
        "vol_theory": vol_theory,
        "fills": fills,
        "final_inventory": q_inv,
        "inv_track": inv_track,
        "S_track": S_track,
        "inv_sample": inv_sample,
        "var_gap": var_gap,
    }


def _summ(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))  # mean, standard error


# --------------------------------------------------------------------------- #
# Experiment A - validate the hedging engine against BS gamma-P&L theory       #
# --------------------------------------------------------------------------- #
def experiment_hedging_validation(params: MMParams, sigma_reals, n_sims, seed=0):
    """Static short 1 option, delta-hedged to expiry, no quoting.

    Per path the realised hedged P&L should match the analytic gamma-P&L
    integral; averaged over paths, mean P&L vs realised vol should trace the
    theoretical curve and cross zero at sigma_real == sigma_impl.
    """
    rng = np.random.default_rng(seed)
    rows = []
    scatter_sim, scatter_theory = [], []
    for sr in sigma_reals:
        res = simulate_paths(params, sr, n_sims, rng, quoting=False, init_position=-1)
        sim_m, sim_se = _summ(res["total_pnl"])
        th_m, th_se = _summ(res["vol_theory"])
        rows.append({"sigma_real": sr, "sim_mean": sim_m, "sim_se": sim_se,
                     "theory_mean": th_m, "theory_se": th_se})
        if abs(sr - params.sigma_impl) < 1e-9 or sr in (sigma_reals[0], sigma_reals[-1]):
            scatter_sim.append(res["total_pnl"])
            scatter_theory.append(res["vol_theory"])
    return {"rows": rows,
            "scatter_sim": np.concatenate(scatter_sim) if scatter_sim else np.array([]),
            "scatter_theory": np.concatenate(scatter_theory) if scatter_theory else np.array([])}


# --------------------------------------------------------------------------- #
# Experiment B - full MM, sweep realised vol, decompose P&L                    #
# --------------------------------------------------------------------------- #
def experiment_mm_vol_sweep(params: MMParams, sigma_reals, n_sims, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    sample = None
    for sr in sigma_reals:
        res = simulate_paths(params, sr, n_sims, rng, quoting=True, init_position=0)
        tot_m, tot_se = _summ(res["total_pnl"])
        sp_m, sp_se = _summ(res["spread_capture"])
        vh_m, vh_se = _summ(res["vol_hedge_pnl"])
        f_m, _ = _summ(res["fills"])
        inv_m, _ = _summ(res["final_inventory"])
        rows.append({"sigma_real": sr, "total_mean": tot_m, "total_se": tot_se,
                     "spread_mean": sp_m, "spread_se": sp_se,
                     "vol_mean": vh_m, "vol_se": vh_se,
                     "avg_fills": f_m, "avg_final_inv": inv_m})
        if abs(sr - params.sigma_impl) < 1e-9:
            sample = res
    if sample is None:  # fall back to the middle grid point
        sample = simulate_paths(params, params.sigma_impl, n_sims, rng, quoting=True)
    return {"rows": rows, "sample": sample}


# --------------------------------------------------------------------------- #
# Experiment C - adverse selection (toxic flow) and the hedge-latency channel  #
# --------------------------------------------------------------------------- #
def experiment_adverse_selection(base: MMParams, toxicities, n_sims, seed=2):
    """Sweep flow toxicity at realised == implied vol, symmetric base flow.

    Runs each toxicity twice: hedging BEFORE the move (lag 0, ~instantaneous)
    and AFTER it (lag 1, a realistic hedge latency). Delta-hedging neutralises
    the *direction* of informed flow, so at lag 0 toxicity mostly just costs
    round-trips (less spread capture); the adverse-selection loss proper shows
    up in the lag-1 residual (total minus spread), i.e. the inventory that rode
    the informed move unhedged.
    """
    rows = []
    for tox in toxicities:
        r0 = simulate_paths(replace(base, toxicity=tox, hedge_lag=0),
                            base.sigma_impl, n_sims, np.random.default_rng(seed), quoting=True)
        r1 = simulate_paths(replace(base, toxicity=tox, hedge_lag=1),
                            base.sigma_impl, n_sims, np.random.default_rng(seed), quoting=True)
        rows.append({
            "toxicity": tox,
            "avg_fills": float(r1["fills"].mean()),
            "lag0_total": _summ(r0["total_pnl"])[0],
            "lag0_resid": _summ(r0["vol_hedge_pnl"])[0],
            "lag1_total": _summ(r1["total_pnl"])[0],
            "lag1_spread": _summ(r1["spread_capture"])[0],
            "lag1_resid": _summ(r1["vol_hedge_pnl"])[0],   # ~ adverse-selection cost
        })
    return rows


def experiment_toxic_spread(base: MMParams, toxicities, half_spreads, n_sims, seed=3):
    """At a realistic hedge lag, does a wider quoted spread survive toxic flow?"""
    grid = {}
    for hs in half_spreads:
        totals = []
        for tox in toxicities:
            r = simulate_paths(replace(base, toxicity=tox, hedge_lag=1, half_spread=hs),
                               base.sigma_impl, n_sims, np.random.default_rng(seed), quoting=True)
            totals.append(_summ(r["total_pnl"])[0])
        grid[hs] = totals
    return grid


# --------------------------------------------------------------------------- #
# Experiment D - vol-informed (vega-toxic) flow: unhedgeable, must be priced   #
# --------------------------------------------------------------------------- #
def _regime_vols(base: MMParams, vol_shock: float, n_sims: int, seed: int):
    """Per-path realised vols: sigma_impl +/- vol_shock with p=1/2 each.

    Symmetric around implied, so a desk facing only uninformed flow has no
    systematic vol edge or cost - any loss under vol-informed flow is pure
    adverse selection, not a mispriced mark.
    """
    rng = np.random.default_rng(seed)
    hi = rng.random(n_sims) < 0.5
    return np.where(hi, base.sigma_impl + vol_shock, base.sigma_impl - vol_shock)


def experiment_vol_informed_flow(base: MMParams, toxicities, vol_shock=0.06,
                                 n_sims=4000, seed=4):
    """Directional vs vol-informed toxicity under INSTANT hedging (lag 0).

    Realised vol per path is sigma_impl +/- vol_shock with equal probability.
    Hedging before the move neutralises direction-informed flow, so its cost
    stays ~0 at any toxicity. Vol-informed flow is different in kind: the
    informed side buys our ask exactly on the paths whose realised vol will
    exceed implied and sells to us on the quiet paths, so the desk is
    systematically short gamma into storms and long gamma into calm. No hedge
    frequency fixes that - it is a vega bet selected against us, and the only
    defences are price (spread / vol markup) or flow discrimination.
    """
    rows = []
    for tox in toxicities:
        sig_paths = _regime_vols(base, vol_shock, n_sims, seed)
        r_dir = simulate_paths(replace(base, toxicity=tox, vol_toxicity=0.0, hedge_lag=0),
                               sig_paths, n_sims, np.random.default_rng(seed + 1), quoting=True)
        r_vol = simulate_paths(replace(base, toxicity=0.0, vol_toxicity=tox, hedge_lag=0),
                               sig_paths, n_sims, np.random.default_rng(seed + 1), quoting=True)
        rows.append({
            "toxicity": tox,
            "dir_total": _summ(r_dir["total_pnl"])[0],
            "dir_resid": _summ(r_dir["vol_hedge_pnl"])[0],
            "vol_total": _summ(r_vol["total_pnl"])[0],
            "vol_spreadcap": _summ(r_vol["spread_capture"])[0],
            "vol_resid": _summ(r_vol["vol_hedge_pnl"])[0],   # ~ vega adverse selection
        })
    return rows


def experiment_vol_spread_defence(base: MMParams, vol_spreads, tox=0.5,
                                  vol_shock=0.06, n_sims=4000, seed=5):
    """Does quoting a half-spread in VOL space defend against vol-informed flow?

    The vega markup charges the informed flow in its own currency, so the
    vega-adverse-selection residual shrinks toward zero as it widens - but a
    wider quote also kills volume (Avellaneda-Stoikov intensity decays in
    quote distance), so under toxic flow the optimum markup is interior, and
    under clean flow any markup is pure cost. Each row therefore also reports
    the same desk facing purely uninformed flow (``clean_total``): the markup
    is a *defence*, priced only when the flow warrants it, not a free lunch.
    """
    rows = []
    for vs in vol_spreads:
        sig_paths = _regime_vols(base, vol_shock, n_sims, seed)
        r = simulate_paths(replace(base, vol_toxicity=tox, hedge_lag=0, vol_spread=vs),
                           sig_paths, n_sims, np.random.default_rng(seed + 1), quoting=True)
        rc = simulate_paths(replace(base, vol_toxicity=0.0, hedge_lag=0, vol_spread=vs),
                            sig_paths, n_sims, np.random.default_rng(seed + 1), quoting=True)
        rows.append({
            "vol_spread": vs,
            "total": _summ(r["total_pnl"])[0],
            "spreadcap": _summ(r["spread_capture"])[0],
            "resid": _summ(r["vol_hedge_pnl"])[0],
            "clean_total": _summ(rc["total_pnl"])[0],
            "avg_fills": float(r["fills"].mean()),
        })
    return rows


# --------------------------------------------------------------------------- #
# Plotting (guarded - matplotlib optional) and CLI                            #
# --------------------------------------------------------------------------- #
def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _plot_validation(val, params, plt, outdir):
    rows = val["rows"]
    sr = [r["sigma_real"] for r in rows]
    sim = [r["sim_mean"] for r in rows]
    sim_se = [r["sim_se"] for r in rows]
    th = [r["theory_mean"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].errorbar(sr, sim, yerr=sim_se, fmt="o", label="simulated hedged P&L", capsize=3)
    ax[0].plot(sr, th, "-", label="BS gamma-P&L theory")
    ax[0].axvline(params.sigma_impl, color="grey", ls="--", lw=1, label="implied vol")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("realised vol"); ax[0].set_ylabel("P&L (short 1 option, hedged)")
    ax[0].set_title("Static short, delta-hedged: sim vs theory"); ax[0].legend(fontsize=8)
    if len(val["scatter_sim"]):
        s, t = val["scatter_theory"], val["scatter_sim"]
        ax[1].scatter(s, t, s=6, alpha=0.25)
        lo, hi = min(s.min(), t.min()), max(s.max(), t.max())
        ax[1].plot([lo, hi], [lo, hi], "r-", lw=1, label="y = x")
        ax[1].set_xlabel("analytic gamma P&L (per path)")
        ax[1].set_ylabel("simulated P&L (per path)")
        ax[1].set_title("Per-path identity (discrete-hedge noise)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/hedging_validation.png", dpi=130)
    plt.close(fig)


def _plot_sweep(sweep, params, plt, outdir):
    rows = sweep["rows"]
    gap = [r["sigma_real"] - params.sigma_impl for r in rows]
    tot = np.array([r["total_mean"] for r in rows]); tot_se = np.array([r["total_se"] for r in rows])
    sp = np.array([r["spread_mean"] for r in rows]); sp_se = np.array([r["spread_se"] for r in rows])
    vh = np.array([r["vol_mean"] for r in rows]); vh_se = np.array([r["vol_se"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for y, ye, lab, c in [(tot, tot_se, "total P&L", "C0"),
                          (sp, sp_se, "spread capture", "C2"),
                          (vh, vh_se, "vol / hedging P&L", "C3")]:
        ax.plot(gap, y, "-o", color=c, label=lab, ms=4)
        ax.fill_between(gap, y - ye, y + ye, color=c, alpha=0.2)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("realised vol  -  implied vol")
    ax.set_ylabel("mean P&L per option (across paths)")
    ax.set_title("Market-maker P&L vs realised-implied vol\n(net-short desk absorbing client buy flow)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/mm_pnl_vs_vol.png", dpi=130)
    plt.close(fig)

    # sample path: inventory + underlying
    s = sweep["sample"]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    steps = np.arange(len(s["inv_sample"]))
    ax1.plot(steps, s["inv_sample"], color="C1", label="inventory (one path)")
    ax1.axhline(0, color="k", lw=0.6); ax1.set_ylabel("option inventory (contracts)", color="C1")
    ax1.set_xlabel("hedge step")
    ax2 = ax1.twinx()
    ax2.plot(steps, s["S_track"], color="C4", alpha=0.7, label="underlying (one path)")
    ax2.set_ylabel("underlying price", color="C4")
    ax1.set_title("Sample path: inventory vs underlying")
    fig.tight_layout()
    fig.savefig(f"{outdir}/sample_inventory_path.png", dpi=130)
    plt.close(fig)


def _plot_adverse_selection(adv_rows, spread_grid, params, plt, outdir):
    tox = [r["toxicity"] for r in adv_rows]
    lag0 = np.array([r["lag0_total"] for r in adv_rows])
    lag1 = np.array([r["lag1_total"] for r in adv_rows])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: instantaneous vs lagged hedging; the gap is adverse selection
    axA.plot(tox, lag0, "-o", color="C2", label="hedge before move (instant)")
    axA.plot(tox, lag1, "-o", color="C3", label="hedge after move (latency)")
    axA.fill_between(tox, lag0, lag1, color="C3", alpha=0.15, label="adverse-selection cost")
    axA.axhline(0, color="k", lw=0.6)
    axA.set_xlabel("flow toxicity (informed fraction)")
    axA.set_ylabel("mean total P&L per option")
    axA.set_title("Toxic flow: cost is realised through hedge latency")
    axA.legend(fontsize=8)

    # Panel B: a wider spread buys tolerance to toxicity
    for hs, totals in spread_grid.items():
        axB.plot(tox, totals, "-o", ms=4, label=f"half-spread = {hs:.2f}")
    axB.axhline(0, color="k", lw=0.6)
    axB.set_xlabel("flow toxicity (informed fraction)")
    axB.set_ylabel("mean total P&L per option (hedge latency)")
    axB.set_title("Widening the quote buys tolerance to toxic flow")
    axB.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/adverse_selection.png", dpi=130)
    plt.close(fig)


def _plot_vol_informed(vi_rows, defence_rows, plt, outdir):
    tox = [r["toxicity"] for r in vi_rows]
    dir_tot = np.array([r["dir_total"] for r in vi_rows])
    vol_tot = np.array([r["vol_total"] for r in vi_rows])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: with instant hedging, direction-informed flow costs ~nothing;
    # vol-informed flow still bleeds - the unhedgeable kind of toxicity.
    axA.plot(tox, dir_tot, "-o", color="C2", label="direction-informed flow (hedged instantly)")
    axA.plot(tox, vol_tot, "-o", color="C3", label="vol-informed flow (hedged instantly)")
    axA.fill_between(tox, dir_tot, vol_tot, color="C3", alpha=0.15,
                     label="unhedgeable vega adverse selection")
    axA.axhline(0, color="k", lw=0.6)
    axA.set_xlabel("informed fraction of flow")
    axA.set_ylabel("mean total P&L per option")
    axA.set_title("Instant hedging kills directional toxicity;\nvol toxicity survives it")
    axA.legend(fontsize=8)

    # Panel B: the defence is priced in vol space - it removes the vega loss
    # (residual -> 0) but costs volume, so it only pays against toxic flow.
    vs = [r["vol_spread"] for r in defence_rows]
    axB.plot(vs, [r["total"] for r in defence_rows], "-o", color="C0",
             label="total P&L (vol-toxic flow)")
    axB.plot(vs, [r["clean_total"] for r in defence_rows], "-o", color="C2", ms=4,
             label="total P&L (uninformed flow)")
    axB.plot(vs, [r["resid"] for r in defence_rows], "--o", color="C3", ms=4,
             label="vega adverse-selection residual")
    axB.axhline(0, color="k", lw=0.6)
    axB.set_xlabel("quoted half-spread in vol space")
    axB.set_ylabel("mean P&L per option")
    axB.set_title("The vol markup removes the vega loss but costs volume:\n"
                  "an interior optimum against toxic flow, pure cost against clean")
    axB.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/vol_informed_flow.png", dpi=130)
    plt.close(fig)


def _print_table(title, rows, cols, fmts):
    print(f"\n{title}")
    print("  " + "".join(f"{c:>16}" for c in cols))
    for r in rows:
        print("  " + "".join(f"{fmts[c](r[c]):>16}" for c in cols))


def main():
    import os
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    params = MMParams(flow_imbalance=0.30)
    n_sims = 4000
    sig_grid = np.round(np.linspace(0.12, 0.30, 13), 4)

    print("Options market-making simulator")
    print(f"  underlying S0={params.S0}, K={params.K}, T={params.T}y, "
          f"implied vol={params.sigma_impl}, steps={params.n_steps}, sims/grid={n_sims}")

    val = experiment_hedging_validation(params, sig_grid, n_sims, seed=0)
    _print_table(
        "Experiment A - static short option, delta-hedged (sim vs BS gamma-P&L theory)",
        val["rows"], ["sigma_real", "sim_mean", "sim_se", "theory_mean"],
        {"sigma_real": lambda x: f"{x:.3f}", "sim_mean": lambda x: f"{x:+.4f}",
         "sim_se": lambda x: f"{x:.4f}", "theory_mean": lambda x: f"{x:+.4f}"})

    sweep = experiment_mm_vol_sweep(params, sig_grid, n_sims, seed=1)
    _print_table(
        "Experiment B - full MM P&L decomposition vs realised vol",
        sweep["rows"], ["sigma_real", "total_mean", "spread_mean", "vol_mean", "avg_fills", "avg_final_inv"],
        {"sigma_real": lambda x: f"{x:.3f}", "total_mean": lambda x: f"{x:+.3f}",
         "spread_mean": lambda x: f"{x:+.3f}", "vol_mean": lambda x: f"{x:+.3f}",
         "avg_fills": lambda x: f"{x:.1f}", "avg_final_inv": lambda x: f"{x:+.2f}"})

    # Experiment C - adverse selection / toxic flow
    adv_base = MMParams(flow_imbalance=0.0)  # symmetric flow, realised == implied
    tox_grid = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
    adv_rows = experiment_adverse_selection(adv_base, tox_grid, n_sims, seed=2)
    _print_table(
        "Experiment C - adverse selection (realised = implied vol, symmetric flow)",
        adv_rows, ["toxicity", "avg_fills", "lag1_spread", "lag1_resid", "lag0_total", "lag1_total"],
        {"toxicity": lambda x: f"{x:.2f}", "avg_fills": lambda x: f"{x:.1f}",
         "lag1_spread": lambda x: f"{x:+.3f}", "lag1_resid": lambda x: f"{x:+.3f}",
         "lag0_total": lambda x: f"{x:+.3f}", "lag1_total": lambda x: f"{x:+.3f}"})
    print("  lag1_resid ~ the adverse-selection cost: ~0 with no toxicity, strongly")
    print("  negative as informed flow rises. lag0 (instant hedge) avoids it.")
    spread_grid = experiment_toxic_spread(adv_base, tox_grid, [0.10, 0.15, 0.25], n_sims, seed=3)

    # Experiment D - vol-informed (vega-toxic) flow
    vi_rows = experiment_vol_informed_flow(adv_base, tox_grid, vol_shock=0.06,
                                           n_sims=n_sims, seed=4)
    _print_table(
        "Experiment D - vol-informed flow vs direction-informed flow (both hedged INSTANTLY)",
        vi_rows, ["toxicity", "dir_total", "vol_total", "vol_spreadcap", "vol_resid"],
        {"toxicity": lambda x: f"{x:.2f}", "dir_total": lambda x: f"{x:+.3f}",
         "vol_total": lambda x: f"{x:+.3f}", "vol_spreadcap": lambda x: f"{x:+.3f}",
         "vol_resid": lambda x: f"{x:+.3f}"})
    print("  Instant hedging neutralises direction-informed flow (dir_total ~ flat) but")
    print("  NOT vol-informed flow: vol_resid goes negative as vega-toxic flow selects")
    print("  which vol regime each side of the book rides. The defence is price, not speed:")
    defence_rows = experiment_vol_spread_defence(adv_base, [0.0, 0.002, 0.005, 0.01, 0.02],
                                                 tox=0.5, vol_shock=0.06,
                                                 n_sims=n_sims, seed=5)
    _print_table(
        "Experiment D2 - defending with a vol-space half-spread (vol toxicity = 0.5)",
        defence_rows, ["vol_spread", "total", "clean_total", "resid", "avg_fills"],
        {"vol_spread": lambda x: f"{x:.3f}", "total": lambda x: f"{x:+.3f}",
         "clean_total": lambda x: f"{x:+.3f}", "resid": lambda x: f"{x:+.3f}",
         "avg_fills": lambda x: f"{x:.1f}"})
    print("  The markup shrinks the vega loss (resid -> 0) but costs volume: against")
    print("  toxic flow the optimum is interior; against clean flow it is pure cost.")

    plt = _try_matplotlib()
    if plt is None:
        print("\n[matplotlib not available - skipped figures; numeric tables above are the result]")
    else:
        _plot_validation(val, params, plt, outdir)
        _plot_sweep(sweep, params, plt, outdir)
        _plot_adverse_selection(adv_rows, spread_grid, params, plt, outdir)
        _plot_vol_informed(vi_rows, defence_rows, plt, outdir)
        print(f"\nFigures written to {outdir}/:")
        print("  hedging_validation.png, mm_pnl_vs_vol.png, sample_inventory_path.png,")
        print("  adverse_selection.png, vol_informed_flow.png")


if __name__ == "__main__":
    main()
