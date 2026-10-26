"""Offline tests for the IV-skew scanner's own-IV pipeline.

The scanner now inverts implied vols from bid-ask mid prices with the repo's
Brent inverter instead of trusting yfinance's ``impliedVolatility`` field.
These tests build synthetic chains whose PRICES come from a known smile while
the vendor IV column is filled with garbage - the pipeline must recover the
true smile from the prices alone. No network access anywhere.

Run:  python -m pytest tests/test_iv_skew.py -q
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from skew_bubble_indicator import IV_skew as X
from pricing_and_vol_surface.vol_surface import bs_call

S, T, R = 100.0, 30.0 / 365.0, 0.03


def true_smile(K: float) -> float:
    """A put-skewed smile: vol rises as strikes fall below spot."""
    return 0.20 + 0.8 * max(0.0, (0.95 * S - K) / S)


def make_chain(otype: str, strikes=None) -> pd.DataFrame:
    """A synthetic chain priced off the true smile; vendor IV is GARBAGE.

    Half-spreads are proportional to price (~6% total spread) as in a real
    chain - a fixed absolute spread would trip the spread-percentage gate on
    every cheap OTM quote and silently empty the wings.
    """
    if strikes is None:
        strikes = np.arange(80.0, 121.0, 2.5)
    rows = []
    for K in strikes:
        sigma = true_smile(K)
        call = bs_call(S, K, T, R, sigma)
        price = call if otype == "call" else call - S + K * np.exp(-R * T)
        half = 0.03 * price
        rows.append({
            "strike": K,
            "bid": price - half,
            "ask": price + half,
            "volume": 500,
            "openInterest": 1000,
            "lastPrice": price,
            "impliedVolatility": 9.99,   # vendor value is nonsense on purpose
        })
    return pd.DataFrame(rows)


def test_add_own_iv_recovers_true_smile_from_prices():
    for otype in ("put", "call"):
        df = X.add_own_iv(make_chain(otype), S, T, R, otype)
        assert (df["yf_iv"] == 9.99).all()   # vendor column preserved as diagnostic
        err = np.abs(df["impliedVolatility"].values
                     - np.array([true_smile(k) for k in df["strike"]]))
        assert np.nanmax(err) < 0.01         # recovered from prices, not vendor IV


def test_no_arb_violating_mid_inverts_to_nan():
    df = make_chain("call", strikes=np.array([90.0]))
    df.loc[0, "bid"] = 5.0                   # deep below intrinsic (S-K=10)
    df.loc[0, "ask"] = 5.2
    out = X.add_own_iv(df, S, T, R, "call")
    assert np.isnan(out["impliedVolatility"].iloc[0])


def test_missing_two_sided_market_inverts_to_nan():
    df = make_chain("call", strikes=np.array([100.0]))
    df.loc[0, "bid"] = 0.0                   # no bid -> no usable mid
    out = X.add_own_iv(df, S, T, R, "call")
    assert np.isnan(out["impliedVolatility"].iloc[0])


def test_validation_pipeline_runs_on_own_iv():
    """validate_option_data must keep good rows (own IV in the sane band)
    even though the vendor IV column is out of range, and drop rows whose
    price cannot be inverted."""
    df = make_chain("put")
    # A put quoted ABOVE its no-arbitrage upper bound (K e^-rT ~ 89.8): the
    # tight spread passes the spread gate, but the mid cannot be inverted.
    bad = pd.DataFrame([{
        "strike": 90.0, "bid": 92.0, "ask": 93.0,
        "volume": 500, "openInterest": 1000, "lastPrice": 92.5,
        "impliedVolatility": 9.99,
    }])
    bad_iv = X.add_own_iv(bad, S, T, R, "put")["impliedVolatility"].iloc[0]
    assert not np.isfinite(bad_iv)
    out = X.validate_option_data(pd.concat([df, bad], ignore_index=True),
                                 S, T, R, "put")
    # good rows survive on OWN IV despite garbage vendor IV...
    assert len(out) >= len(df) - 1
    assert out["impliedVolatility"].between(X.MIN_IV, X.MAX_IV).all()
    # ...and the uninvertible quote is gone
    assert not (out["bid"] >= 92.0).any()


def test_skew_metric_reads_the_price_implied_skew():
    """End-to-end without network: put IV at the OTM put strike must exceed
    call IV at the OTM call strike, matching the smile the PRICES encode."""
    puts = X.validate_option_data(make_chain("put"), S, T, R, "put")
    calls = X.validate_option_data(make_chain("call"), S, T, R, "call")
    put_iv = X.find_mean_iv_at_strike(puts, 0.90 * S)
    call_iv = X.find_mean_iv_at_strike(calls, 1.10 * S)
    assert np.isfinite(put_iv) and np.isfinite(call_iv)
    assert put_iv - call_iv > 0.02           # the built-in put skew
    assert abs(put_iv - true_smile(0.90 * S)) < 0.02


# ---------------------------------------------------------------------------
# Delta-anchored strike selection
# ---------------------------------------------------------------------------
def flat_iv_chain(sigma: float = 0.20, strikes=None) -> pd.DataFrame:
    """A validated-shape chain with KNOWN flat IVs, so target deltas map to
    analytically checkable strikes (the selector reads strike + IV only)."""
    if strikes is None:
        strikes = np.arange(80.0, 121.0, 2.5)
    return pd.DataFrame({"strike": strikes, "impliedVolatility": sigma})


def expected_delta_strike(strikes, sigma, target, otype):
    """Independent Black-Scholes delta computed in the test (scipy), so the
    selector is checked against math it does not share code with."""
    from scipy.stats import norm
    strikes = np.asarray(strikes, dtype=float)
    d1 = (np.log(S / strikes) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) if otype == "call" else norm.cdf(d1) - 1.0
    return float(strikes[np.argmin(np.abs(delta - target))])


def test_delta_anchor_picks_known_strikes_on_flat_vol():
    sigma = 0.20
    puts, calls = flat_iv_chain(sigma), flat_iv_chain(sigma)
    got = X.get_delta_anchored_strike_targets(puts, calls, S, T, R)
    assert got is not None
    put_k, call_k = got
    otm_puts = puts["strike"][puts["strike"] <= S]
    otm_calls = calls["strike"][calls["strike"] >= S]
    assert put_k == expected_delta_strike(otm_puts, sigma, -0.25, "put")
    assert call_k == expected_delta_strike(otm_calls, sigma, 0.25, "call")
    assert put_k < S < call_k                # anchors are OTM by construction


def test_delta_targets_are_configurable_and_signed_or_not():
    puts, calls = flat_iv_chain(), flat_iv_chain()
    k25 = X.get_delta_anchored_strike_targets(puts, calls, S, T, R)
    k10 = X.get_delta_anchored_strike_targets(
        puts, calls, S, T, R, put_delta=-0.10, call_delta=0.10)
    assert k10[0] < k25[0] and k10[1] > k25[1]   # 10-delta sits further OTM
    # an unsigned put target means the same thing as the signed one
    unsigned = X.get_delta_anchored_strike_targets(
        puts, calls, S, T, R, put_delta=0.10, call_delta=0.10)
    assert unsigned == k10


def test_delta_target_constants_plumb_through(monkeypatch):
    """Module-level targets (what the CLI flags set) are read at call time."""
    puts, calls = flat_iv_chain(), flat_iv_chain()
    k25 = X.get_delta_anchored_strike_targets(puts, calls, S, T, R)
    monkeypatch.setattr(X, "TARGET_PUT_DELTA", -0.10)
    monkeypatch.setattr(X, "TARGET_CALL_DELTA", 0.10)
    k10 = X.get_delta_anchored_strike_targets(puts, calls, S, T, R)
    assert k10[0] < k25[0] and k10[1] > k25[1]


def test_delta_anchor_unavailable_falls_back_to_none():
    good = flat_iv_chain()
    # no finite IVs on one side -> whole selection unavailable
    bad = flat_iv_chain(sigma=np.nan)
    assert X.get_delta_anchored_strike_targets(bad, good, S, T, R) is None
    assert X.get_delta_anchored_strike_targets(good, bad, S, T, R) is None
    # ATM-only strikes: nearest delta is beyond the tolerance -> unavailable
    atm_only = flat_iv_chain(strikes=np.array([99.0, 100.0, 101.0]))
    assert X.get_delta_anchored_strike_targets(atm_only, atm_only, S, T, R) is None
    # degenerate inputs
    assert X.get_delta_anchored_strike_targets(good, good, S, 0.0, R) is None
    assert X.get_delta_anchored_strike_targets(good, good, np.nan, T, R) is None


def test_delta_anchored_skew_reads_the_smile_end_to_end():
    """Full pipeline on the synthetic put-skewed smile: prices -> own IVs ->
    delta anchors -> positive skew, without any moneyness fallback. The smile
    is flat above 0.95*S, where the 25-delta put sits, so the 10-delta pair -
    whose put anchor is inside the skewed wing - is what has signal here."""
    puts = X.validate_option_data(make_chain("put"), S, T, R, "put")
    calls = X.validate_option_data(make_chain("call"), S, T, R, "call")
    got = X.get_delta_anchored_strike_targets(
        puts, calls, S, T, R, put_delta=-0.10, call_delta=0.25)
    assert got is not None
    assert got[0] < 0.95 * S                 # the anchor reached the wing
    put_iv = X.find_mean_iv_at_strike(puts, got[0])
    call_iv = X.find_mean_iv_at_strike(calls, got[1])
    assert np.isfinite(put_iv) and np.isfinite(call_iv)
    assert put_iv - call_iv > 0.005          # the built-in put skew, delta-anchored


# --------------------------------------------------------------------------- #
# Persistence: idempotent snapshot, quality-gated summary columns             #
# --------------------------------------------------------------------------- #
def test_snapshot_save_is_idempotent_per_day(tmp_path, monkeypatch):
    """A same-day re-run must REPLACE that day's rows rather than append a
    second copy of the same delayed session (which silently doubled days in
    every downstream average); earlier days are kept."""
    csv = str(tmp_path / "snap.csv")
    monkeypatch.setattr(X, "DAILY_SNAPSHOT_CSV", csv)
    X.save_iv_skew_snapshot(pd.DataFrame({"ticker": ["AAA", "BBB"],
                                          "iv_skew": [0.10, -0.05]}))
    X.save_iv_skew_snapshot(pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"],
                                          "iv_skew": [0.11, -0.06, 0.02]}))
    out = pd.read_csv(csv)
    assert len(out) == 3 and set(out["ticker"]) == {"AAA", "BBB", "CCC"}

    stale = pd.DataFrame({"ticker": ["OLD"], "iv_skew": [0.0],
                          "timestamp": [datetime.now() - timedelta(days=1)]})
    stale.to_csv(csv, index=False)
    X.save_iv_skew_snapshot(pd.DataFrame({"ticker": ["NEW"], "iv_skew": [0.1]}))
    out = pd.read_csv(csv)
    assert set(out["ticker"]) == {"OLD", "NEW"}


def test_bubble_summary_records_reliable_only_companion_columns(tmp_path, monkeypatch):
    """The summary CSV carries the quality-gated companion statistics next to
    the headline ones, and the ALERT fires on the reliable-only rate when
    enough reliable rows exist (recorded as alert_basis). Here only 2 of 4
    rows are reliable - below MIN_RELIABLE_FOR_ALERT - so the alert falls
    back to the all-valid basis."""
    csv = str(tmp_path / "bubble.csv")
    monkeypatch.setattr(X, "BUBBLE_SUMMARY_CSV", csv)
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "index": ["tech"] * 5,
        "index_name": ["Tech"] * 5,
        "iv_skew": [0.10, -0.05, np.nan, -0.02, 0.30],
        "is_reliable": [True, True, True, False, False],
    })
    X.save_bubble_summary_by_index(df)
    out = pd.read_csv(csv)
    row = out[out["index"] == "tech"].iloc[0]
    assert row["valid_measurements"] == 4          # NaN row excluded
    assert row["inverted_count"] == 2              # B and D are negative
    assert row["reliable_measurements"] == 2       # A and B
    assert row["inverted_reliable"] == 1           # B only
    assert row["inversion_pct_reliable"] == 50.0
    assert row["alert_basis"] == "all"             # too few reliable rows
    assert row["alert_level"] == "WARNING"         # 2/4 = 50% on all-valid
    grow = out[out["index"] == "GLOBAL"].iloc[0]
    assert grow["reliable_measurements"] == 2
    assert grow["inversion_pct_reliable"] == 50.0
    assert grow["alert_basis"] == "all"


def test_alert_fires_on_reliable_basis_when_supported():
    """With enough reliable rows, the alert level follows the reliable-only
    rate even when the all-valid rate sits in a different band; below the
    minimum it falls back to all-valid."""
    def frame(skews, reliable):
        return pd.DataFrame({"iv_skew": skews, "is_reliable": reliable})

    # 6 reliable rows, 4 of them inverted (66.7% > 60 -> CRITICAL); the 4
    # unreliable rows pull the all-valid rate to 50% (WARNING band).
    df = frame([0.1, -0.05, -0.02, 0.3, -0.01, -0.03] + [0.2, -0.04, 0.25, 0.15],
               [True] * 6 + [False] * 4)
    pct, basis = X.alert_inversion_rate(df)
    assert (pct, basis) == (4 / 6 * 100, "reliable")

    # 3 reliable rows only (all clean) - below MIN_RELIABLE_FOR_ALERT, so
    # the alert uses the all-valid rate (5/10 inverted).
    df = frame([-0.05, -0.02, -0.01, -0.03, -0.04] + [0.1, 0.2, 0.3, 0.25, 0.15],
               [True, True, True] + [False] * 7)
    pct, basis = X.alert_inversion_rate(df)
    assert (pct, basis) == (5 / 10 * 100, "all")


def test_parity_implied_q_recovers_a_known_dividend():
    """Chains priced under parity with a 2% dividend yield must give the
    yield back; absurd implied carry (stale quotes) must refuse to 0.0."""
    q_true = 0.02
    calls, puts = [], []
    for K in np.arange(90.0, 111.0, 2.5):
        C = max(0.5 * (S - K) + 5.0, 0.4)          # arbitrary but sane mids
        P = C - S * np.exp(-q_true * T) + K * np.exp(-R * T)
        calls.append({"strike": K, "bid": C - 0.05, "ask": C + 0.05})
        puts.append({"strike": K, "bid": P - 0.05, "ask": P + 0.05})
    q_hat = X.parity_implied_q(pd.DataFrame(calls), pd.DataFrame(puts), S, T, R)
    assert abs(q_hat - q_true) < 1e-6

    # a huge C-P gap implies absurd carry -> refused to 0.0
    bad = pd.DataFrame([{"strike": K, "bid": P + 5.0, "ask": P + 5.1}
                        for K in (97.5, 100.0, 102.5)])
    good_calls = pd.DataFrame(calls)
    assert X.parity_implied_q(good_calls, bad, S, T, R) == 0.0

    # fewer than two matched pairs -> 0.0
    one = pd.DataFrame([calls[4]])
    assert X.parity_implied_q(one, pd.DataFrame([puts[4]]), S, T, R) == 0.0
