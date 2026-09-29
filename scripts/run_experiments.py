"""Reproduce every figure and number of the report:  python scripts/run_experiments.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optlab import (bs_price, bs_greeks, binomial_crr, implied_vol, simulate_gbm,
                    delta_hedge_short_call, min_variance_hedge_ratio, n_contracts,
                    beta_hedge_contracts)
from optlab.jumps import merton_call

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, RED, GREY = "#1f4e8c", "#e67e22", "#c0392b", "#8a94a6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
res = {}

# 1. Binomial convergence -------------------------------------------------------
S, K, T, r, sig = 100, 100, 1.0, 0.03, 0.20
bs = float(bs_price(S, K, T, r, sig))
steps = np.arange(5, 401, 1)
tree = [binomial_crr(S, K, T, r, sig, steps=int(n)) for n in steps]
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(steps, tree, color=BLUE, lw=0.9, label="CRR binomial")
ax.axhline(bs, color=RED, ls="--", lw=1.2, label=f"Black-Scholes = {bs:.4f}")
ax.set(xlabel="Number of steps", ylabel="ATM call price", title="Binomial tree converges to Black-Scholes")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "1_binomial_convergence.png", dpi=180)
res["binomial"] = {"bs": bs, "n50": binomial_crr(S, K, T, r, sig, 50),
                   "n400": binomial_crr(S, K, T, r, sig, 400)}
am = binomial_crr(S, 110, T, r, sig, 1000, kind="put", american=True)
eu = binomial_crr(S, 110, T, r, sig, 1000, kind="put")
res["early_exercise_premium_put_K110"] = {"american": am, "european": eu, "premium": am - eu}

# 2. Delta hedging error vs rebalancing frequency ------------------------------
mu, n_paths = 0.08, 20_000
freqs = [4, 12, 52, 252, 1000]
stds, pnls = [], {}
for n in freqs:
    paths = simulate_gbm(S, mu, sig, T, n, n_paths, seed=42)
    pnl = delta_hedge_short_call(paths, K, T, r, sig)
    stds.append(pnl.std()); pnls[n] = pnl
premium = bs
res["hedging_error"] = {str(n): {"mean": float(pnls[n].mean()), "std": float(s),
                                 "std_pct_premium": float(s / premium * 100)} for n, s in zip(freqs, stds)}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
for n, col in [(12, GREY), (52, ORANGE), (252, BLUE)]:
    axs[0].hist(pnls[n], bins=120, range=(-4, 4), histtype="stepfilled", alpha=.45, color=col,
                label=f"{n} rebalances (σ = {pnls[n].std():.2f})")
axs[0].set(xlabel="Hedged P&L at maturity", ylabel="Paths", title="Short ATM call, delta-hedged")
axs[0].legend(frameon=False, fontsize=7.5)
axs[1].loglog(freqs, stds, "o-", color=BLUE, label="Simulated std of P&L")
ref = stds[2] * np.sqrt(freqs[2] / np.array(freqs))
axs[1].loglog(freqs, ref, "--", color=RED, label="∝ 1/√N")
axs[1].set(xlabel="Rebalances per year (log)", ylabel="Hedging error std (log)", title="Hedging error shrinks like 1/√N")
axs[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "2_hedging_error.png", dpi=180)

# 3. Realised vs implied volatility --------------------------------------------
implied = 0.20
real_vols = np.linspace(0.10, 0.30, 9)
sim_mean, theo = [], []
for rv in real_vols:
    paths = simulate_gbm(S, mu, rv, T, 252, 10_000, seed=7)
    sim_mean.append(delta_hedge_short_call(paths, K, T, r, implied).mean())
    # seller keeps the premium difference, carried to maturity
    theo.append((bs - float(bs_price(S, K, T, r, rv))) * np.exp(r * T))
res["vol_pnl"] = {f"{rv:.3f}": {"simulated": float(m), "bs_difference": float(t)}
                  for rv, m, t in zip(real_vols, sim_mean, theo)}
fig, ax = plt.subplots(figsize=(6, 3))
ax.axhline(0, color=GREY, lw=0.7); ax.axvline(implied * 100, color=GREY, lw=0.7, ls=":")
ax.plot(real_vols * 100, sim_mean, "o", color=BLUE, label="Simulated mean hedged P&L (option seller)")
ax.plot(real_vols * 100, theo, "-", color=RED, lw=1.2, label="BS(σ implied) − BS(σ realised)")
ax.set(xlabel="Realised volatility (%) — call sold at 20% implied", ylabel="P&L at maturity",
       title="A delta-hedged option is a bet on realised vs implied vol")
ax.legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(FIG / "3_vol_pnl.png", dpi=180)

# 4. Futures cross-hedge (Hull jet-fuel set-up) ---------------------------------
rng = np.random.default_rng(3)
rho_true, sS, sF, n_months = 0.928, 0.0263, 0.0313, 120
z1, z2 = rng.standard_normal((2, n_months))
dF = sF * z1
dS = sS * (rho_true * z1 + np.sqrt(1 - rho_true**2) * z2)
est = min_variance_hedge_ratio(dS, dF)
N_tail = n_contracts(est["h_star"], 1.94 * 2_000_000, 1.99, 42_000)
# out-of-sample check on fresh data
z1, z2 = rng.standard_normal((2, 10_000))
dF_o = sF * z1; dS_o = sS * (rho_true * z1 + np.sqrt(1 - rho_true**2) * z2)
hs = np.linspace(0, 1.5, 61)
var_red = [1 - np.var(dS_o - h * dF_o) / np.var(dS_o) for h in hs]
res["futures_hedge"] = {"h_star_estimated": float(est["h_star"]), "h_star_true": rho_true * sS / sF,
                        "rho_est": float(est["rho"]), "effectiveness": float(est["effectiveness"]),
                        "contracts_tailed": float(N_tail),
                        "oos_variance_reduction_at_h_star": float(1 - np.var(dS_o - est['h_star'] * dF_o) / np.var(dS_o)),
                        "oos_variance_reduction_at_h1": float(1 - np.var(dS_o - dF_o) / np.var(dS_o))}
res["beta_hedge_contracts"] = beta_hedge_contracts(1.2, 0.6, 10e6, 5000, 10)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(hs, np.array(var_red) * 100, color=BLUE)
ax.axvline(est["h_star"], color=RED, ls="--", lw=1, label=f"h* estimated = {est['h_star']:.3f}")
ax.axvline(1, color=GREY, ls=":", lw=1, label="naive h = 1")
ax.set(xlabel="Hedge ratio h", ylabel="Variance removed (%)", title="Minimum-variance cross-hedge (out-of-sample)")
ax.legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(FIG / "4_futures_hedge.png", dpi=180)

# 5. Jumps create an implied-volatility skew -------------------------------------
strikes = np.linspace(70, 130, 25)
jump = dict(sigma=0.15, lam=0.5, mu_j=-0.10, delta_j=0.10)
ivs = [implied_vol(merton_call(S, k, 0.5, r, **jump), S, k, 0.5, r) for k in strikes]
res["skew"] = {"iv_K70": ivs[0], "iv_K100": ivs[12], "iv_K130": ivs[-1]}
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(strikes, np.array(ivs) * 100, "o-", color=BLUE, ms=3, label="Implied vol of jump-diffusion prices")
ax.axhline(jump["sigma"] * 100, color=GREY, ls=":", label="Diffusion vol (15%)")
ax.set(xlabel="Strike (spot = 100, T = 6 months)", ylabel="Implied vol (%)",
       title="Negative jumps → equity-style volatility skew")
ax.legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(FIG / "5_skew.png", dpi=180)

(ROOT / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps(res, indent=2, default=float))
