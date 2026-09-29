"""Client solutions built on the pricing library: what a sales desk would pitch.

1. Corporate FX hedging: exporter receiving USD, comparing forward / option / zero-cost collar.
2. Investor products: capital-protected note and reverse convertible.

FX convention: the underlying X is the EUR value of 1 USD (X = 1 / EURUSD).
Options on USD priced in EUR with Garman-Kohlhagen = Black-Scholes with q = r_usd.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .pricing import bs_price


# ----------------------------------------------------------------------------- FX
def fx_forward(x0, r_dom, r_for, T):
    """Covered interest parity: F = X0 * exp((r_dom - r_for) T)."""
    return x0 * np.exp((r_dom - r_for) * T)


def usd_put(x0, K, T, r_eur, r_usd, vol):
    """EUR price of a put on 1 USD (right to sell 1 USD at K EUR)."""
    return float(bs_price(x0, K, T, r_eur, vol, q=r_usd, kind="put"))


def usd_call(x0, K, T, r_eur, r_usd, vol):
    return float(bs_price(x0, K, T, r_eur, vol, q=r_usd, kind="call"))


def zero_cost_collar_cap(x0, K_floor, T, r_eur, r_usd, vol):
    """Strike of the USD call sold so that its premium pays exactly for the USD put bought."""
    target = usd_put(x0, K_floor, T, r_eur, r_usd, vol)
    return brentq(lambda k: usd_call(x0, k, T, r_eur, r_usd, vol) - target, x0 * 0.8, x0 * 2.0)


def exporter_outcomes(notional_usd, x_T, fwd, K_put, put_prem, K_floor, K_cap, r_eur, T):
    """EUR received at maturity for each hedging strategy (premiums carried to T)."""
    x_T = np.asarray(x_T, float)
    carry = np.exp(r_eur * T)
    return {
        "Unhedged": notional_usd * x_T,
        "Forward": np.full_like(x_T, notional_usd * fwd),
        "Buy USD put": notional_usd * (np.maximum(x_T, K_put) - put_prem * carry),
        "Zero-cost collar": notional_usd * np.clip(x_T, K_floor, K_cap),
    }


# ----------------------------------------------------------------- investor products
def capital_protected_note(S0, T, r, q, vol, fees=0.01, cap=None):
    """100% capital protection + participation in the index upside.

    Budget = 1 - zero-coupon - fees, spent on ATM calls (or an ATM/cap call spread).
    Returns participation and the building blocks, all in % of notional.
    """
    zc = np.exp(-r * T)
    budget = 1 - zc - fees
    call = float(bs_price(S0, S0, T, r, vol, q, "call")) / S0
    if cap is not None:
        call -= float(bs_price(S0, S0 * (1 + cap), T, r, vol, q, "call")) / S0
    return {"zero_coupon": zc, "fees": fees, "option_budget": budget,
            "option_price": call, "participation": budget / call}


def capital_protected_payoff(perf, participation, cap=None):
    """Redemption in % of notional given index performance S_T/S0 - 1."""
    up = np.maximum(np.asarray(perf, float), 0)
    if cap is not None:
        up = np.minimum(up, cap)
    return 1 + participation * up


def reverse_convertible(S0, T, r, q, vol, strike_pct=1.0, fees=0.01):
    """Bond + short put (strike = strike_pct * S0). Returns the coupon paid at maturity.

    Investor pays 100%. Bank invests it at r and sells the investor's put.
    Coupon (in % of notional, paid at T) = interest + put premium carried to T - fees.
    """
    K = strike_pct * S0
    put = float(bs_price(S0, K, T, r, vol, q, "put")) / K  # per 1 of notional (N/K puts)
    coupon = (np.exp(r * T) - 1) + (put - fees) * np.exp(r * T)
    return {"put_premium": put, "coupon": coupon, "strike_pct": strike_pct}


def reverse_convertible_payoff(perf, coupon, strike_pct=1.0):
    """Redemption: 100% + coupon if S_T >= K, else (S_T/K) + coupon (paid in shares)."""
    ratio = (1 + np.asarray(perf, float)) / strike_pct
    return np.minimum(ratio, 1) + coupon
