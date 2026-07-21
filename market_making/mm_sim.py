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
from typing import Optional

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

    # online toxicity estimation: the desk watches its own fill markouts (did
    # the underlying move with the client on the next bar?) and keeps an EWMA
    # of the agreement rate. Informed flow trades one side only, so the
    # informed share of FILLS is f = tox/(2-tox); the agreement rate is
    # 0.5 + f/2, giving tox_hat = 2*f_hat/(1+f_hat) with f_hat = 2*agree - 1.
    # With adaptive_spread on, the desk widens its quote by
    # spread_slope * tox_hat - using only information available at quote time.
    adaptive_spread: bool = False
    tox_ewma_alpha: float = 0.08    # EWMA step per fill observation
    spread_slope: float = 0.0       # extra half-spread at tox_hat = 1

    # optional per-step DIRECTIONAL toxicity schedule (length n_steps); when
    # set it overrides the scalar `toxicity`, letting toxicity switch regime
    # mid-simulation.
    toxicity_schedule: Optional[np.ndarray] = None

    # online VOL-toxicity estimation: a vega-space markout. After clients buy
    # options, did the next bar realise more variance than implied predicts?
    # Per fill the observation is side * (r^2/(sigma_impl^2 dt) - 1), side=+1
    # when the client bought. Uninformed flow nets to ~0 (both sides trade in
    # both regimes); vol-informed flow buys exactly before high realised
    # variance, so the EWMA of this markout measures the vega edge the flow is
    # extracting per fill, in relative-variance units. With
    # adaptive_vol_spread on, the desk quotes vol_spread_slope * max(EWMA, 0)
    # (capped) of vol-space markup - the estimate is clipped at zero so noise
    # cannot rectify into phantom markup faster than evidence accumulates.
    adaptive_vol_spread: bool = False
    volmark_ewma_alpha: float = 0.08
    volmark_horizon: int = 5        # markout window: bars of realised var per fill
    vol_spread_slope: float = 0.0   # vol markup per unit of markout excess
    vol_spread_cap: float = 0.008   # ceiling on the adaptive vol markup
    # Null threshold: the clipped markout EWMA has a positive noise floor even
    # on clean flow (zero-mean noise rectified by the clip). Only the excess
    # above this floor triggers markup, so clean flow is not taxed by phantom
    # toxicity. A real desk calibrates it from its own null - e.g. bootstrap
    # the same statistic with fill sides shuffled.
    volmark_deadband: float = 0.10
    vol_toxicity_schedule: Optional[np.ndarray] = None  # per-step vol toxicity

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

    ``sigma_real`` may be a scalar (one realised vol for all paths), an array
    of shape (n_sims,) giving each path its own realised vol, or a matrix of
    shape (n_sims, n_steps) giving each path a vol *path* (regimes that switch
    mid-simulation). Per-path/per-step vols are what make vol-informed
    ("vega-toxic") flow expressible: informed clients condition on the vol
    regime currently in force.

    Returns a dict of per-path arrays (shape (n_sims,)) plus a couple of
    representative time series for plotting a single path.
    """
    p = params
    max_tox = (float(np.max(p.toxicity_schedule)) if p.toxicity_schedule is not None
               else p.toxicity)
    max_vtox = (float(np.max(p.vol_toxicity_schedule))
                if p.vol_toxicity_schedule is not None else p.vol_toxicity)
    if max_tox + max_vtox > 1.0:
        raise ValueError("toxicity + vol_toxicity must be <= 1 (they partition the flow)")
    for sched in (p.toxicity_schedule, p.vol_toxicity_schedule):
        if sched is not None and len(sched) != p.n_steps:
            raise ValueError("toxicity schedules must have length n_steps")
    m = p.contract_multiplier
    dt = p.T / p.n_steps
    sqrt_dt = np.sqrt(dt)
    sigma_real = np.asarray(sigma_real, dtype=float)
    if sigma_real.ndim <= 1:
        SR = np.broadcast_to(sigma_real.reshape(-1, 1) if sigma_real.ndim == 1
                             else sigma_real, (n_sims, p.n_steps))
    else:
        if sigma_real.shape != (n_sims, p.n_steps):
            raise ValueError("2-D sigma_real must have shape (n_sims, n_steps)")
        SR = sigma_real

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

    # Online toxicity estimator state (per path): EWMA of fill/move agreement,
    # with the remaining weight of the 0.5 prior tracked in ``ewma_decay`` so
    # the estimate can be bias-corrected (Adam-style) - otherwise the warm-up
    # drags every estimate toward "clean" for most of a 126-step book.
    agree = np.full(n_sims, 0.5)
    ewma_decay = np.ones(n_sims)             # (1-step)^(#observations), per path
    tox_hat_track = np.zeros(p.n_steps)      # mean tox_hat across paths, per step

    # Vega-markout estimator state (per path). Zero-initialised, so clipping
    # at zero is itself the conservative prior - no debiasing needed. Each
    # fill is scored against the realised variance of the next
    # ``volmark_horizon`` bars (a single squared return is chi-square noisy;
    # a K-bar window cuts the noise by ~sqrt(K)), via ring buffers so the
    # observation arrives K bars after the fill - late, but causal.
    K_mark = max(int(p.volmark_horizon), 1)
    volmark = np.zeros(n_sims)
    side_buf = np.zeros((n_sims, K_mark))
    vr_buf = np.zeros((n_sims, K_mark))
    vr_sum = np.zeros(n_sims)
    volmark_track = np.zeros(p.n_steps)      # mean clipped estimate, per step

    def _tox_hat_from_state():
        # Bias-corrected agreement, then estimate x confidence: with few
        # observations the corrected estimate is noisy and the clip at zero
        # rectifies that noise into phantom toxicity, so it is shrunk by the
        # evidence weight (1 - remaining prior mass).
        conf = 1.0 - ewma_decay
        denom = np.maximum(conf, 1e-12)
        a_hat = np.where(conf > 0.0, (agree - 0.5 * ewma_decay) / denom, 0.5)
        f_hat = conf * np.clip(2.0 * a_hat - 1.0, 0.0, 1.0)
        return 2.0 * f_hat / (1.0 + f_hat)

    var_gap = p.sigma_impl**2 - SR[:, 0]**2

    for t in range(p.n_steps):
        sig_t = SR[:, t]
        tau = p.T - t * dt
        theo = bs_price(S, p.K, tau, p.r, p.sigma_impl, p.q, p.otype)

        # Draw this step's underlying shock up front so informed ("toxic") flow
        # can arrive on the side the imminent move will favour.
        z = rng.standard_normal(n_sims)

        if quoting:
            # Optional vol-space half-spread: price the ask at a marked-up vol
            # and the bid at a marked-down vol. Near expiry vega -> 0 and the
            # vol spread collapses naturally, as it should. With
            # adaptive_vol_spread on, the markup follows the (causal, clipped)
            # vega-markout estimate per path.
            volmark_hat = np.maximum(volmark, 0.0)
            volmark_track[t] = float(volmark_hat.mean())
            vs_eff = p.vol_spread
            if p.adaptive_vol_spread:
                excess = np.maximum(volmark_hat - p.volmark_deadband, 0.0)
                vs_eff = vs_eff + np.minimum(p.vol_spread_slope * excess,
                                             p.vol_spread_cap)
            if p.adaptive_vol_spread or p.vol_spread > 0.0:
                ask_base = bs_price(S, p.K, tau, p.r, p.sigma_impl + vs_eff, p.q, p.otype)
                bid_base = bs_price(S, p.K, tau, p.r,
                                    np.maximum(p.sigma_impl - vs_eff, 1e-4), p.q, p.otype)
            else:
                ask_base = bid_base = theo

            # Online toxicity estimate (causal: uses fills/moves up to t-1).
            tox_hat = _tox_hat_from_state()
            tox_hat_track[t] = float(tox_hat.mean())
            half_spread = p.half_spread + (p.spread_slope * tox_hat
                                           if p.adaptive_spread else 0.0)

            skew_shift = p.skew_coef * q_inv
            bid = bid_base - skew_shift - half_spread
            ask = ask_base - skew_shift + half_spread
            d_bid = theo - bid          # distance from fair (drives fill intensity)
            d_ask = ask - theo

            lam_bid = A_bid * np.exp(-p.k * d_bid)
            lam_ask = A_ask * np.exp(-p.k * d_ask)
            prob_bid = np.clip(1.0 - np.exp(-lam_bid * dt), 0.0, 1.0)
            prob_ask = np.clip(1.0 - np.exp(-lam_ask * dt), 0.0, 1.0)

            tox = (float(p.toxicity_schedule[t]) if p.toxicity_schedule is not None
                   else p.toxicity)
            vtox = (float(p.vol_toxicity_schedule[t])
                    if p.vol_toxicity_schedule is not None else p.vol_toxicity)
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
            hi_vol = sig_t > p.sigma_impl
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
        vol_theory += 0.5 * q_inv * gamma_opt * S**2 * (sig_t**2 - p.sigma_impl**2) * dt * m

        # evolve the underlying with the REALISED vol using the pre-drawn shock
        log_ret = (p.r - p.q - 0.5 * sig_t**2) * dt + sig_t * sqrt_dt * z
        S = S * np.exp(log_ret)

        # Update the online toxicity estimates from this step's fills and the
        # move that just resolved (one-bar markouts). Only now is the move
        # observable, so the estimates used for quoting next step stay causal.
        if quoting:
            # Directional: did the underlying move the client's way?
            n_fills_step = fill_bid.astype(float) + fill_ask.astype(float)
            agree_count = (fill_ask & up).astype(float) + (fill_bid & ~up).astype(float)
            has_fill = n_fills_step > 0
            obs = np.where(has_fill, agree_count / np.maximum(n_fills_step, 1.0), 0.0)
            step_sz = np.clip(p.tox_ewma_alpha * n_fills_step, 0.0, 1.0)
            agree = np.where(has_fill, agree + step_sz * (obs - agree), agree)
            ewma_decay = np.where(has_fill, ewma_decay * (1.0 - step_sz), ewma_decay)

            # Vega-space: did realised variance beat implied over the K bars
            # after a net client buy (and undershoot after a net sell)?
            # Netted per step so a both-sides print (no information)
            # contributes nothing. The fill K bars ago is scored against the
            # variance window that just completed.
            side_net = fill_ask.astype(float) - fill_bid.astype(float)
            var_ratio = log_ret**2 / (p.sigma_impl**2 * dt)
            slot = t % K_mark
            vr_sum = vr_sum + var_ratio - vr_buf[:, slot]
            if t >= K_mark - 1:
                side_old = side_buf[:, (t - K_mark + 1) % K_mark] if K_mark > 1 else side_net
                has_side = side_old != 0.0
                vobs = side_old * (vr_sum / K_mark - 1.0)
                volmark = np.where(has_side,
                                   volmark + p.volmark_ewma_alpha * (vobs - volmark),
                                   volmark)
            vr_buf[:, slot] = var_ratio
            side_buf[:, slot] = side_net

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
        "tox_hat_track": tox_hat_track,
        "tox_hat_final": _tox_hat_from_state(),
        "volmark_track": volmark_track,
        "volmark_final": np.maximum(volmark, 0.0),
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
# Experiment E - online toxicity estimation and adaptive quoting               #
# --------------------------------------------------------------------------- #
def experiment_online_toxicity(base: MMParams, tox_hi=0.6, spread_slope=0.25,
                               n_sims=4000, seed=6):
    """Can the desk INFER toxicity from its own fills and defend itself?

    Three desks - static (base spread), oracle-wide (base + slope*tox_hi,
    i.e. permanently sized for the toxic regime), and adaptive (widens by
    slope * tox_hat, where tox_hat comes from the online markout estimator) -
    each run through three flows: clean, toxic, and a regime switch (clean
    first half, toxic second). All hedge with a one-bar lag, where directional
    toxicity actually costs. The adaptive desk should track the truth closely
    enough to defend in the toxic regimes without paying the oracle-wide
    desk's volume cost in the clean one.
    """
    n = base.n_steps
    schedules = {
        "clean": np.zeros(n),
        "toxic": np.full(n, tox_hi),
        "regime switch": np.where(np.arange(n) < n // 2, 0.0, tox_hi),
    }
    desks = {
        "static": dict(adaptive_spread=False),
        "oracle-wide": dict(adaptive_spread=False,
                            half_spread=base.half_spread + spread_slope * tox_hi),
        "adaptive": dict(adaptive_spread=True, spread_slope=spread_slope),
    }
    totals = {}
    tracks = {}
    for sname, sched in schedules.items():
        for dname, kw in desks.items():
            p = replace(base, toxicity_schedule=sched, hedge_lag=1, **kw)
            r = simulate_paths(p, base.sigma_impl, n_sims,
                               np.random.default_rng(seed), quoting=True)
            totals[(sname, dname)] = _summ(r["total_pnl"])[0]
            if dname == "adaptive":
                tracks[sname] = r["tox_hat_track"]
    return {"totals": totals, "tracks": tracks, "schedules": schedules,
            "tox_hi": tox_hi}


# --------------------------------------------------------------------------- #
# Experiment F - online VOL-toxicity estimation and adaptive vol markup        #
# --------------------------------------------------------------------------- #
def _regime_vol_paths(base: MMParams, vol_shock: float, n_sims: int,
                      block: int = 42, seed: int = 0) -> np.ndarray:
    """Per-path vol PATHS: hi/lo regimes (sigma_impl +/- vol_shock, p=1/2 each)
    redrawn every ``block`` steps.

    Regime *variation within a path* is what makes online vol-toxicity
    inference possible at all: with one fixed regime per path, a clean desk
    that happens to sit in the high-vol regime is statistically identical to
    a vega-picked-off one, and no estimator can tell them apart. With
    regimes that turn over (~monthly here), informed flow re-aligns with each
    new regime while uninformed flow stays symmetric, and the flow-vs-realised
    -variance markout becomes identifiable.
    """
    rng = np.random.default_rng(seed)
    n_blocks = -(-base.n_steps // block)
    hi = rng.random((n_sims, n_blocks)) < 0.5
    sig = np.where(hi, base.sigma_impl + vol_shock, base.sigma_impl - vol_shock)
    return np.repeat(sig, block, axis=1)[:, :base.n_steps]


def experiment_online_vol_toxicity(base: MMParams, vtox_hi=0.6, vol_shock=0.06,
                                   vol_spread_slope=0.06, oracle_vol_spread=0.005,
                                   n_sims=4000, seed=7):
    """Close the loop on Experiment D: detect vega-toxic flow online and price
    the vol markup adaptively.

    Three desks - static (no markup), oracle-markup (a fixed vol_spread sized
    for the toxic regime), adaptive (vol_spread_slope * clipped vega-markout
    EWMA) - run through clean, vol-toxic, and regime-switching flow. All hedge
    instantly (lag 0), so every P&L difference is vega adverse selection and
    quote width, never hedge latency. Vol regimes redraw every ~21 bars (see
    _regime_vol_paths - the identifiability requirement).
    """
    n = base.n_steps
    schedules = {
        "clean": np.zeros(n),
        "vol-toxic": np.full(n, vtox_hi),
        "regime switch": np.where(np.arange(n) < n // 2, 0.0, vtox_hi),
    }
    desks = {
        "static": dict(adaptive_vol_spread=False),
        "oracle-markup": dict(adaptive_vol_spread=False, vol_spread=oracle_vol_spread),
        "adaptive": dict(adaptive_vol_spread=True, vol_spread_slope=vol_spread_slope),
    }
    totals = {}
    tracks = {}
    sig_paths = _regime_vol_paths(base, vol_shock, n_sims, seed=seed)
    for sname, sched in schedules.items():
        for dname, kw in desks.items():
            p = replace(base, vol_toxicity_schedule=sched, hedge_lag=0, **kw)
            r = simulate_paths(p, sig_paths, n_sims,
                               np.random.default_rng(seed + 1), quoting=True)
            totals[(sname, dname)] = _summ(r["total_pnl"])[0]
            if dname == "adaptive":
                tracks[sname] = r["volmark_track"]
    return {"totals": totals, "tracks": tracks, "schedules": schedules,
            "vtox_hi": vtox_hi, "vol_shock": vol_shock}


# --------------------------------------------------------------------------- #
# Plotting (guarded - matplotlib optional) and CLI                            #
# --------------------------------------------------------------------------- #
def _try_matplotlib():
    """Return (plt, plotstyle) with the repo's shared chart style applied, or None."""
    try:
        import os
        import sys

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import plotstyle as ps
        ps.apply_style()
        return plt, ps
    except Exception:
        return None


def _plot_validation(val, params, plt, ps, outdir):
    rows = val["rows"]
    sr = [r["sigma_real"] for r in rows]
    sim = [r["sim_mean"] for r in rows]
    sim_se = [r["sim_se"] for r in rows]
    th = [r["theory_mean"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].errorbar(sr, sim, yerr=sim_se, fmt="o", color=ps.series_color(0),
                   label="simulated hedged P&L", capsize=3)
    ax[0].plot(sr, th, "-", color=ps.series_color(1), label="BS gamma-P&L theory")
    ax[0].axvline(params.sigma_impl, color=ps.MUTED, ls="--", lw=1, label="implied vol")
    ax[0].axhline(0, color=ps.BASELINE, lw=0.8)
    ax[0].set_xlabel("realised vol"); ax[0].set_ylabel("P&L (short 1 option, hedged)")
    ax[0].set_title("Static short, delta-hedged: sim vs theory"); ax[0].legend(fontsize=8)
    if len(val["scatter_sim"]):
        s, t = val["scatter_theory"], val["scatter_sim"]
        ax[1].scatter(s, t, s=6, alpha=0.25, color=ps.series_color(0))
        lo, hi = min(s.min(), t.min()), max(s.max(), t.max())
        ax[1].plot([lo, hi], [lo, hi], color=ps.INK, lw=1, label="y = x")
        ax[1].set_xlabel("analytic gamma P&L (per path)")
        ax[1].set_ylabel("simulated P&L (per path)")
        ax[1].set_title("Per-path identity (discrete-hedge noise)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/hedging_validation.png", dpi=130)
    plt.close(fig)


def _plot_sweep(sweep, params, plt, ps, outdir):
    rows = sweep["rows"]
    gap = [r["sigma_real"] - params.sigma_impl for r in rows]
    tot = np.array([r["total_mean"] for r in rows]); tot_se = np.array([r["total_se"] for r in rows])
    sp = np.array([r["spread_mean"] for r in rows]); sp_se = np.array([r["spread_se"] for r in rows])
    vh = np.array([r["vol_mean"] for r in rows]); vh_se = np.array([r["vol_se"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for y, ye, lab, c in [(tot, tot_se, "total P&L", ps.series_color(0)),
                          (sp, sp_se, "spread capture", ps.series_color(2)),
                          (vh, vh_se, "vol / hedging P&L", ps.series_color(1))]:
        ax.plot(gap, y, "-o", color=c, label=lab, ms=4)
        ax.fill_between(gap, y - ye, y + ye, color=c, alpha=0.2)
    ax.axvline(0, color=ps.MUTED, ls="--", lw=1)
    ax.axhline(0, color=ps.BASELINE, lw=0.8)
    ax.set_xlabel("realised vol  -  implied vol")
    ax.set_ylabel("mean P&L per option (across paths)")
    ax.set_title("Market-maker P&L vs realised-implied vol\n(net-short desk absorbing client buy flow)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/mm_pnl_vs_vol.png", dpi=130)
    plt.close(fig)

    # sample path: inventory and underlying as stacked panels sharing the time
    # axis (one measure per axis - never a twin/dual axis).
    s = sweep["sample"]
    steps = np.arange(len(s["inv_sample"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    ax1.step(steps, s["inv_sample"], where="post", color=ps.series_color(6),
             label="inventory (one path)")
    ax1.axhline(0, color=ps.BASELINE, lw=0.8)
    ax1.set_ylabel("option inventory (contracts)")
    ax1.set_title("Sample path: inventory vs underlying")
    ax1.legend(fontsize=8)
    ax2.plot(steps, s["S_track"], color=ps.series_color(0), label="underlying (same path)")
    ax2.set_ylabel("underlying price")
    ax2.set_xlabel("hedge step")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/sample_inventory_path.png", dpi=130)
    plt.close(fig)


def _plot_adverse_selection(adv_rows, spread_grid, params, plt, ps, outdir):
    tox = [r["toxicity"] for r in adv_rows]
    lag0 = np.array([r["lag0_total"] for r in adv_rows])
    lag1 = np.array([r["lag1_total"] for r in adv_rows])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: instantaneous vs lagged hedging; the gap is adverse selection
    axA.plot(tox, lag0, "-o", color=ps.series_color(0), label="hedge before move (instant)")
    axA.plot(tox, lag1, "-o", color=ps.series_color(1), label="hedge after move (latency)")
    axA.fill_between(tox, lag0, lag1, color=ps.CRITICAL, alpha=0.12,
                     label="adverse-selection cost")
    axA.axhline(0, color=ps.BASELINE, lw=0.8)
    axA.set_xlabel("flow toxicity (informed fraction)")
    axA.set_ylabel("mean total P&L per option")
    axA.set_title("Toxic flow: cost is realised through hedge latency")
    axA.legend(fontsize=8)

    # Panel B: a wider spread buys tolerance to toxicity
    for i, (hs, totals) in enumerate(spread_grid.items()):
        axB.plot(tox, totals, "-o", ms=4, color=ps.series_color(i),
                 label=f"half-spread = {hs:.2f}")
    axB.axhline(0, color=ps.BASELINE, lw=0.8)
    axB.set_xlabel("flow toxicity (informed fraction)")
    axB.set_ylabel("mean total P&L per option (hedge latency)")
    axB.set_title("Widening the quote buys tolerance to toxic flow")
    axB.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/adverse_selection.png", dpi=130)
    plt.close(fig)


def _plot_vol_informed(vi_rows, defence_rows, plt, ps, outdir):
    tox = [r["toxicity"] for r in vi_rows]
    dir_tot = np.array([r["dir_total"] for r in vi_rows])
    vol_tot = np.array([r["vol_total"] for r in vi_rows])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: with instant hedging, direction-informed flow costs ~nothing;
    # vol-informed flow still bleeds - the unhedgeable kind of toxicity.
    axA.plot(tox, dir_tot, "-o", color=ps.series_color(0),
             label="direction-informed flow (hedged instantly)")
    axA.plot(tox, vol_tot, "-o", color=ps.series_color(1),
             label="vol-informed flow (hedged instantly)")
    axA.fill_between(tox, dir_tot, vol_tot, color=ps.CRITICAL, alpha=0.12,
                     label="unhedgeable vega adverse selection")
    axA.axhline(0, color=ps.BASELINE, lw=0.8)
    axA.set_xlabel("informed fraction of flow")
    axA.set_ylabel("mean total P&L per option")
    axA.set_title("Instant hedging kills directional toxicity;\nvol toxicity survives it")
    axA.legend(fontsize=8)

    # Panel B: the defence is priced in vol space - it removes the vega loss
    # (residual -> 0) but costs volume, so it only pays against toxic flow.
    vs = [r["vol_spread"] for r in defence_rows]
    axB.plot(vs, [r["total"] for r in defence_rows], "-o", color=ps.series_color(0),
             label="total P&L (vol-toxic flow)")
    axB.plot(vs, [r["clean_total"] for r in defence_rows], "-o", color=ps.series_color(2),
             ms=4, label="total P&L (uninformed flow)")
    axB.plot(vs, [r["resid"] for r in defence_rows], "--o", color=ps.series_color(1),
             ms=4, label="vega adverse-selection residual")
    axB.axhline(0, color=ps.BASELINE, lw=0.8)
    axB.set_xlabel("quoted half-spread in vol space")
    axB.set_ylabel("mean P&L per option")
    axB.set_title("The vol markup removes the vega loss but costs volume:\n"
                  "an interior optimum against toxic flow, pure cost against clean")
    axB.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/vol_informed_flow.png", dpi=130)
    plt.close(fig)


def _plot_online_toxicity(online, params, plt, ps, outdir):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: the estimator tracking a regime switch, causally.
    track = online["tracks"]["regime switch"]
    sched = online["schedules"]["regime switch"]
    steps = np.arange(len(track))
    axA.plot(steps, sched, color=ps.INK, ls="--", lw=1.5, label="true toxicity")
    axA.plot(steps, track, color=ps.series_color(0), lw=2,
             label="online estimate (mean tox_hat)")
    axA.set_xlabel("quote/hedge step")
    axA.set_ylabel("directional toxicity")
    axA.set_ylim(-0.05, 1.0)
    axA.set_title("The markout estimator tracks a toxicity\nregime switch it was never told about")
    axA.legend(fontsize=8)

    # Panel B: desk comparison across flow scenarios (grouped bars).
    scenarios = list(online["schedules"].keys())
    desks = ["static", "oracle-wide", "adaptive"]
    x = np.arange(len(scenarios))
    width = 0.26
    for i, d in enumerate(desks):
        vals = [online["totals"][(s, d)] for s in scenarios]
        bars = axB.bar(x + (i - 1) * width, vals, width * 0.92,
                       color=ps.series_color(i), label=d)
        axB.bar_label(bars, fmt="%+.1f", fontsize=8, color=ps.INK_2, padding=2)
    axB.axhline(0, color=ps.BASELINE, lw=0.8)
    axB.set_xticks(x); axB.set_xticklabels(scenarios)
    axB.set_ylabel("mean total P&L per option (hedge latency)")
    axB.set_title("Adaptive quoting defends when toxic\nwithout paying the wide quote when clean")
    axB.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/online_toxicity.png", dpi=130)
    plt.close(fig)


def _plot_online_vol_toxicity(online, params, plt, ps, outdir):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: the vega-markout estimate under the vol-toxicity regime switch.
    track = online["tracks"]["regime switch"]
    steps = np.arange(len(track))
    n = len(track)
    axA.axvline(n // 2, color=ps.INK, ls="--", lw=1.2,
                label="vol-toxic flow switches on")
    axA.plot(steps, track, color=ps.series_color(0), lw=2,
             label="vega markout estimate (mean)")
    axA.axhline(0.10, color=ps.MUTED, ls=":", lw=1.2,
                label="calibrated null threshold")
    axA.set_xlabel("quote/hedge step")
    axA.set_ylabel("vega markout (relative-variance units)")
    axA.set_title("The vega-markout estimator detects vol-toxic flow\n"
                  "(noisier and slower than the directional markout - honestly so)")
    axA.legend(fontsize=8)

    # Panel B: desk comparison across flow scenarios (grouped bars).
    scenarios = list(online["schedules"].keys())
    desks = ["static", "oracle-markup", "adaptive"]
    x = np.arange(len(scenarios))
    width = 0.26
    for i, d in enumerate(desks):
        vals = [online["totals"][(s, d)] for s in scenarios]
        bars = axB.bar(x + (i - 1) * width, vals, width * 0.92,
                       color=ps.series_color(i), label=d)
        axB.bar_label(bars, fmt="%+.2f", fontsize=8, color=ps.INK_2, padding=2)
    axB.axhline(0, color=ps.BASELINE, lw=0.8)
    axB.set_xticks(x); axB.set_xticklabels(scenarios)
    axB.set_ylabel("mean total P&L per option (instant hedging)")
    axB.set_title("Adaptive vol markup: near-static when clean, part of the\n"
                  "oracle's edge when toxic - one book's markout is noisy")
    axB.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/online_vol_toxicity.png", dpi=130)
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

    # Experiment E - online toxicity estimation and adaptive quoting
    online = experiment_online_toxicity(adv_base, tox_hi=0.6, spread_slope=0.25,
                                        n_sims=n_sims, seed=6)
    print("\nExperiment E - online toxicity estimation (markout EWMA), hedge latency on")
    print(f"  {'scenario':>16} {'static':>10} {'oracle-wide':>12} {'adaptive':>10}")
    for s in online["schedules"]:
        print(f"  {s:>16} {online['totals'][(s, 'static')]:>+10.3f} "
              f"{online['totals'][(s, 'oracle-wide')]:>+12.3f} "
              f"{online['totals'][(s, 'adaptive')]:>+10.3f}")
    print("  The adaptive desk infers toxicity from its own fill markouts: it defends")
    print("  like the oracle-wide desk in toxic flow without paying that desk's volume")
    print("  cost in clean flow, and it re-widens on a mid-sim regime switch unaided.")

    # Experiment F - online VOL-toxicity estimation and adaptive vol markup
    online_v = experiment_online_vol_toxicity(adv_base, n_sims=n_sims, seed=7)
    print("\nExperiment F - online vol-toxicity estimation (vega markout), instant hedging")
    print(f"  {'scenario':>16} {'static':>10} {'oracle-markup':>14} {'adaptive':>10}")
    for s in online_v["schedules"]:
        print(f"  {s:>16} {online_v['totals'][(s, 'static')]:>+10.3f} "
              f"{online_v['totals'][(s, 'oracle-markup')]:>+14.3f} "
              f"{online_v['totals'][(s, 'adaptive')]:>+10.3f}")
    print("  The vega markout detects vol-toxic flow (fills scored against the next")
    print("  bars' realised variance) and prices a markup on the excess over its null")
    print("  threshold. Honest asymmetry vs Experiment E: one book's vega markout is")
    print("  chi-square noisy, so the adaptive desk recovers only part of the oracle's")
    print("  edge when toxicity is stationary - but skips the oracle's clean-flow tax")
    print("  and beats it when toxicity is time-varying. Detection is cheap; per-book")
    print("  repricing is not.")

    res = _try_matplotlib()
    if res is None:
        print("\n[matplotlib not available - skipped figures; numeric tables above are the result]")
    else:
        plt, ps = res
        _plot_validation(val, params, plt, ps, outdir)
        _plot_sweep(sweep, params, plt, ps, outdir)
        _plot_adverse_selection(adv_rows, spread_grid, params, plt, ps, outdir)
        _plot_vol_informed(vi_rows, defence_rows, plt, ps, outdir)
        _plot_online_toxicity(online, params, plt, ps, outdir)
        _plot_online_vol_toxicity(online_v, params, plt, ps, outdir)
        print(f"\nFigures written to {outdir}/:")
        print("  hedging_validation.png, mm_pnl_vs_vol.png, sample_inventory_path.png,")
        print("  adverse_selection.png, vol_informed_flow.png, online_toxicity.png,")
        print("  online_vol_toxicity.png")


if __name__ == "__main__":
    main()
