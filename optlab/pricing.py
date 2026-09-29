"""Option pricing engines: Black-Scholes-Merton, CRR binomial tree, Monte Carlo.

Conventions: S spot, K strike, T maturity in years, r continuously-compounded
risk-free rate, q continuous dividend yield, sigma annualised volatility.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, q, sigma):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton price of a European option (vectorised)."""
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    # at expiry the price is the payoff
    payoff = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    T_safe = np.where(T > 0, T, 1e-12)
    d1, d2 = _d1_d2(S, K, T_safe, r, q, sigma)
    df_r, df_q = np.exp(-r * T_safe), np.exp(-q * T_safe)
    if kind == "call":
        price = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif kind == "put":
        price = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return np.where(T > 0, price, payoff)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Analytical Greeks. Theta is per year, vega and rho per 1.00 change."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf = norm.pdf(d1)
    gamma = df_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * df_q * pdf * np.sqrt(T)
    common_theta = -S * df_q * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = df_q * norm.cdf(d1)
        theta = common_theta - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = df_q * (norm.cdf(d1) - 1)
        theta = common_theta + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
        rho = -K * T * df_r * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def binomial_crr(S, K, T, r, sigma, steps=500, q=0.0, kind="call", american=False):
    """Cox-Ross-Rubinstein tree with backward induction (European or American)."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("No-arbitrage condition d < e^((r-q)dt) < u violated")
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    ST = S * u ** (steps - j) * d**j
    values = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            Si = S * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
            exercise = np.maximum(Si - K, 0) if kind == "call" else np.maximum(K - Si, 0)
            values = np.maximum(values, exercise)
    return float(values[0])


def monte_carlo(S, K, T, r, sigma, q=0.0, kind="call", n_paths=200_000, seed=0):
    """Risk-neutral Monte Carlo with antithetic variates. Returns (price, std error)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    z = np.concatenate([z, -z])
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    disc = np.exp(-r * T) * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(n_paths))


def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-8, max_iter=100):
    """Implied volatility: Newton-Raphson on vega, bisection fallback."""
    lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if kind == "call" \
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if price <= lower:
        raise ValueError("Price below no-arbitrage lower bound")
    sigma = 0.2
    for _ in range(max_iter):
        diff = float(bs_price(S, K, T, r, sigma, q, kind)) - price
        if abs(diff) < tol:
            return sigma
        vega = float(bs_greeks(S, K, T, r, sigma, q, kind)["vega"])
        if vega < 1e-10:
            break
        sigma -= diff / vega
        if not 1e-4 < sigma < 5:
            break
    lo, hi = 1e-4, 5.0  # bisection fallback
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(bs_price(S, K, T, r, mid, q, kind)) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
