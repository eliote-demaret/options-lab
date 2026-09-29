"""Merton (1976) jump-diffusion call price, used to show why implied vol has a skew."""
from __future__ import annotations

from math import exp, factorial, log

from .pricing import bs_price


def merton_call(S, K, T, r, sigma, lam, mu_j, delta_j, n_terms=60):
    """Log-normal jumps: ln(1+J) ~ N(mu_j, delta_j^2), intensity lam per year."""
    k = exp(mu_j + 0.5 * delta_j**2) - 1  # mean relative jump size
    lam_p = lam * (1 + k)
    price = 0.0
    for n in range(n_terms):
        sigma_n = (sigma**2 + n * delta_j**2 / T) ** 0.5
        r_n = r - lam * k + n * log(1 + k) / T
        weight = exp(-lam_p * T) * (lam_p * T) ** n / factorial(n)
        price += weight * float(bs_price(S, K, T, r_n, sigma_n, kind="call"))
    return price
