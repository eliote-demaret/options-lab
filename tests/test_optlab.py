import numpy as np
import pytest

from optlab import (bs_price, bs_greeks, binomial_crr, monte_carlo, implied_vol,
                    simulate_gbm, delta_hedge_short_call, beta_hedge_contracts)

S, K, T, r, sig = 42, 40, 0.5, 0.10, 0.20


def test_hull_reference_values():
    # Hull, Options Futures & Other Derivatives, BSM worked example
    assert bs_price(S, K, T, r, sig, kind="call") == pytest.approx(4.76, abs=0.01)
    assert bs_price(S, K, T, r, sig, kind="put") == pytest.approx(0.81, abs=0.01)


def test_put_call_parity_with_dividends():
    q = 0.03
    c = bs_price(S, K, T, r, sig, q, "call")
    p = bs_price(S, K, T, r, sig, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_no_arbitrage_bounds():
    c = bs_price(S, K, T, r, sig, kind="call")
    assert max(S - K * np.exp(-r * T), 0) <= c <= S


def test_binomial_converges_to_bs():
    bs = float(bs_price(S, K, T, r, sig, kind="call"))
    assert binomial_crr(S, K, T, r, sig, steps=2000) == pytest.approx(bs, abs=2e-3)


def test_hull_two_step_american_put():
    # Hull two-step tree: u=1.2, d=0.8 -> sigma = ln(1.2)/sqrt(1) is not exact CRR,
    # so check the generic properties instead: American put >= European put
    eu = binomial_crr(50, 52, 2, 0.05, 0.3, steps=500, kind="put")
    am = binomial_crr(50, 52, 2, 0.05, 0.3, steps=500, kind="put", american=True)
    assert am > eu


def test_american_call_no_dividend_equals_european():
    eu = binomial_crr(S, K, T, r, sig, steps=500)
    am = binomial_crr(S, K, T, r, sig, steps=500, american=True)
    assert am == pytest.approx(eu, abs=1e-10)


def test_monte_carlo_within_3_std_errors():
    price, se = monte_carlo(S, K, T, r, sig, n_paths=400_000)
    assert abs(price - float(bs_price(S, K, T, r, sig))) < 3 * se


def test_implied_vol_round_trip():
    for vol in (0.05, 0.2, 0.6, 1.2):
        px = float(bs_price(100, 110, 1, 0.03, vol))
        assert implied_vol(px, 100, 110, 1, 0.03) == pytest.approx(vol, abs=1e-6)


def test_greeks_match_finite_differences():
    g = bs_greeks(S, K, T, r, sig)
    h = 1e-4
    fd_delta = (bs_price(S + h, K, T, r, sig) - bs_price(S - h, K, T, r, sig)) / (2 * h)
    fd_vega = (bs_price(S, K, T, r, sig + h) - bs_price(S, K, T, r, sig - h)) / (2 * h)
    assert g["delta"] == pytest.approx(float(fd_delta), abs=1e-6)
    assert g["vega"] == pytest.approx(float(fd_vega), abs=1e-5)
    assert g["delta"] == pytest.approx(0.779, abs=1e-3)


def test_hedging_error_shrinks_with_frequency():
    errs = []
    for n in (13, 52, 252):
        paths = simulate_gbm(100, 0.08, 0.2, 1.0, n, 4000, seed=1)
        errs.append(delta_hedge_short_call(paths, 100, 1.0, 0.02, 0.2).std())
    assert errs[0] > errs[1] > errs[2]
    # error std scales roughly like 1/sqrt(N): 252 vs 52 rebalances -> ratio ~ 2.2
    assert 1.6 < errs[1] / errs[2] < 2.8


def test_beta_hedge():
    assert beta_hedge_contracts(1.2, 0.0, 10e6, 5000, 10) == pytest.approx(-240)
