import numpy as np
import pytest

from optlab.pricing import bs_price
from optlab.solutions import (fx_forward, usd_put, usd_call, zero_cost_collar_cap,
                              exporter_outcomes, capital_protected_note,
                              capital_protected_payoff, reverse_convertible,
                              reverse_convertible_payoff)

X0, T, R_EUR, R_USD, VOL = 1 / 1.10, 0.5, 0.02, 0.04, 0.08


def test_forward_is_covered_interest_parity():
    assert fx_forward(X0, R_EUR, R_USD, T) == pytest.approx(X0 * np.exp(-0.01))


def test_fx_put_call_parity():
    K = 0.90
    c, p = usd_call(X0, K, T, R_EUR, R_USD, VOL), usd_put(X0, K, T, R_EUR, R_USD, VOL)
    assert c - p == pytest.approx(X0 * np.exp(-R_USD * T) - K * np.exp(-R_EUR * T), abs=1e-12)


def test_collar_is_zero_cost_and_cap_above_forward():
    F = fx_forward(X0, R_EUR, R_USD, T)
    k_floor = 0.97 * F
    k_cap = zero_cost_collar_cap(X0, k_floor, T, R_EUR, R_USD, VOL)
    assert usd_call(X0, k_cap, T, R_EUR, R_USD, VOL) == pytest.approx(
        usd_put(X0, k_floor, T, R_EUR, R_USD, VOL), abs=1e-10)
    assert k_floor < F < k_cap


def test_collar_outcomes_bounded():
    F = fx_forward(X0, R_EUR, R_USD, T)
    k_cap = zero_cost_collar_cap(X0, 0.97 * F, T, R_EUR, R_USD, VOL)
    out = exporter_outcomes(1e7, np.linspace(0.7, 1.1, 50), F, F, 0.01, 0.97 * F, k_cap, R_EUR, T)
    assert out["Zero-cost collar"].min() == pytest.approx(1e7 * 0.97 * F)
    assert out["Zero-cost collar"].max() == pytest.approx(1e7 * k_cap)


def test_capital_protected_note_budget_identity():
    n = capital_protected_note(100, 5, 0.03, 0.03, 0.18, fees=0.01)
    assert n["zero_coupon"] + n["fees"] + n["participation"] * n["option_price"] == pytest.approx(1)
    assert capital_protected_payoff(-0.4, n["participation"]) == pytest.approx(1)  # capital protected


def test_cap_increases_participation():
    plain = capital_protected_note(100, 5, 0.03, 0.03, 0.18)["participation"]
    capped = capital_protected_note(100, 5, 0.03, 0.03, 0.18, cap=0.5)["participation"]
    assert capped > plain


def test_reverse_convertible_is_bond_plus_short_put():
    S0, Tm, r, q, vol = 100, 1, 0.03, 0.02, 0.25
    rc = reverse_convertible(S0, Tm, r, q, vol, strike_pct=1.0, fees=0.0)
    # fair product (no fees): PV of expected redemption under Q = 100% of notional
    put = float(bs_price(S0, S0, Tm, r, vol, q, "put")) / S0
    pv = np.exp(-r * Tm) * (1 + rc["coupon"]) - put
    assert pv == pytest.approx(1, abs=1e-12)
    assert reverse_convertible_payoff(-0.3, rc["coupon"]) == pytest.approx(0.7 + rc["coupon"])
