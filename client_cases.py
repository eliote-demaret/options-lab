"""Client solutions case studies:  python scripts/client_cases.py
Market parameters are illustrative, not live quotes."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optlab.solutions import (fx_forward, usd_put, zero_cost_collar_cap, exporter_outcomes,
                              capital_protected_note, capital_protected_payoff,
                              reverse_convertible, reverse_convertible_payoff)

FIG = ROOT / "figures"
BLUE, ORANGE, RED, GREEN, GREY = "#1f4e8c", "#e67e22", "#c0392b", "#2e8b57", "#8a94a6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
out = {}

# ---------------------------------------------------------------- Case 1: exporter
N_USD, EURUSD0, T, R_EUR, R_USD, VOL = 10_000_000, 1.10, 0.5, 0.02, 0.04, 0.08
x0 = 1 / EURUSD0
F = fx_forward(x0, R_EUR, R_USD, T)
put_prem = usd_put(x0, F, T, R_EUR, R_USD, VOL)                  # ATM-forward USD put
k_floor = 0.97 * F
k_cap = zero_cost_collar_cap(x0, k_floor, T, R_EUR, R_USD, VOL)
eurusd_T = np.linspace(0.98, 1.26, 300)
res = exporter_outcomes(N_USD, 1 / eurusd_T, F, F, put_prem, k_floor, k_cap, R_EUR, T)
scen = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25])
tab = exporter_outcomes(N_USD, 1 / scen, F, F, put_prem, k_floor, k_cap, R_EUR, T)
out["fx"] = {"eurusd_spot": EURUSD0, "eurusd_forward": 1 / F, "put_premium_eur": put_prem * N_USD,
             "put_premium_pct": put_prem / x0 * 100, "put_strike_eurusd": 1 / F,
             "collar_worst_eurusd": 1 / k_floor, "collar_best_eurusd": 1 / k_cap,
             "collar_floor_eur": N_USD * k_floor, "collar_cap_eur": N_USD * k_cap,
             "scenarios": {f"{s:.2f}": {k: float(v[i]) for k, v in tab.items()} for i, s in enumerate(scen)}}

fig, ax = plt.subplots(figsize=(7, 3.4))
for (k, v), c, ls in zip(res.items(), [GREY, BLUE, GREEN, ORANGE], [":", "-", "-", "-"]):
    ax.plot(eurusd_T, v / 1e6, color=c, ls=ls, lw=1.8, label=k)
ax.axvline(EURUSD0, color="#ccc", lw=.7)
ax.set(xlabel="EUR/USD in 6 months", ylabel="EUR received (millions)",
       title="Exporter receiving USD 10m in 6 months: EUR received by strategy")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "6_fx_hedging.png", dpi=180)

# --------------------------------------------------- Case 2: capital-protected note
S0, T5, R, Q, VOL_IDX = 100, 5, 0.025, 0.03, 0.18
plain = capital_protected_note(S0, T5, R, Q, VOL_IDX, fees=0.01)
capped = capital_protected_note(S0, T5, R, Q, VOL_IDX, fees=0.01, cap=0.50)
out["cpn"] = {"plain": plain, "capped_50": capped}
perf = np.linspace(-0.6, 1.0, 300)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(perf * 100, (1 + perf) * 100, color=GREY, ls=":", label="Direct index investment")
ax.plot(perf * 100, capital_protected_payoff(perf, plain["participation"]) * 100, color=BLUE, lw=1.8,
        label=f"Protected note, {plain['participation']*100:.0f}% participation")
ax.plot(perf * 100, capital_protected_payoff(perf, capped["participation"], cap=0.5) * 100, color=ORANGE, lw=1.8,
        label=f"Protected note, {capped['participation']*100:.0f}% participation, 50% cap")
ax.set(xlabel="Index performance over 5 years (%)", ylabel="Redemption (% of capital)",
       title="Capital-protected note (5 years)")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "7_capital_protected.png", dpi=180)

# --------------------------------------------------------- Case 3: reverse convertible
S_STOCK, T1, R1, Q1, VOL_STOCK = 100, 1.0, 0.025, 0.02, 0.30
rc100 = reverse_convertible(S_STOCK, T1, R1, Q1, VOL_STOCK, strike_pct=1.00)
rc90 = reverse_convertible(S_STOCK, T1, R1, Q1, VOL_STOCK, strike_pct=0.90)
out["rc"] = {"strike_100": rc100, "strike_90": rc90}
perf1 = np.linspace(-0.6, 0.5, 300)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(perf1 * 100, (1 + perf1) * 100, color=GREY, ls=":", label="Holding the stock")
ax.plot(perf1 * 100, reverse_convertible_payoff(perf1, rc100["coupon"], 1.0) * 100, color=RED, lw=1.8,
        label=f"Reverse convertible, strike 100%, coupon {rc100['coupon']*100:.1f}%")
ax.plot(perf1 * 100, reverse_convertible_payoff(perf1, rc90["coupon"], 0.9) * 100, color=BLUE, lw=1.8,
        label=f"Reverse convertible, strike 90%, coupon {rc90['coupon']*100:.1f}%")
ax.axhline(100, color="#ccc", lw=.7)
ax.set(xlabel="Stock performance over 1 year (%)", ylabel="Redemption (% of capital)",
       title="Reverse convertible (1 year): high coupon, capital at risk")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "8_reverse_convertible.png", dpi=180)

(ROOT / "client_cases.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))
