"""Hedging simulations: dynamic delta hedging of an option, and futures hedge ratios."""
from __future__ import annotations

import numpy as np

from .pricing import bs_price, bs_greeks


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=0):
    """Paths of dS = mu S dt + sigma S dz (exact log-normal discretisation)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = S0 * np.exp(np.cumsum(log_inc, axis=1))
    return np.hstack([np.full((n_paths, 1), S0), paths])


def delta_hedge_short_call(paths, K, T, r, sigma_implied, q=0.0):
    """Sell one call at implied vol, delta-hedge at each path date with shares + cash.

    Returns the final hedged P&L per path (in currency, at maturity).
    Cash account accrues at r; shares receive dividend yield q.
    """
    n_paths, n_dates = paths.shape
    n_steps = n_dates - 1
    dt = T / n_steps
    S0 = paths[:, 0]
    premium = bs_price(S0, K, T, r, sigma_implied, q, "call")
    delta = bs_greeks(S0, K, T, r, sigma_implied, q, "call")["delta"]
    cash = premium - delta * S0  # receive premium, buy delta shares
    for i in range(1, n_steps):
        tau = T - i * dt
        S = paths[:, i]
        cash = cash * np.exp(r * dt) + delta * S * (np.exp(q * dt) - 1)
        new_delta = bs_greeks(S, K, tau, r, sigma_implied, q, "call")["delta"]
        cash -= (new_delta - delta) * S  # rebalance
        delta = new_delta
    ST = paths[:, -1]
    cash = cash * np.exp(r * dt) + delta * ST * (np.exp(q * dt) - 1)
    return cash + delta * ST - np.maximum(ST - K, 0)


def min_variance_hedge_ratio(dS, dF):
    """h* = rho * sigma_S / sigma_F, plus hedge effectiveness rho^2."""
    dS, dF = np.asarray(dS), np.asarray(dF)
    rho = np.corrcoef(dS, dF)[0, 1]
    h = rho * dS.std(ddof=1) / dF.std(ddof=1)
    return {"h_star": h, "rho": rho, "effectiveness": rho**2}


def n_contracts(h_star, exposure_value, futures_price, contract_size):
    """Number of futures contracts with tailing: N* = h* V_A / V_F."""
    return h_star * exposure_value / (futures_price * contract_size)


def beta_hedge_contracts(beta_now, beta_target, portfolio_value, index_futures, multiplier):
    """N = (beta* - beta) V_A / V_F  (negative = sell)."""
    return (beta_target - beta_now) * portfolio_value / (index_futures * multiplier)
