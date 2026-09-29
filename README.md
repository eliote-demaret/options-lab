# Options Lab — pricing, Greeks & hedging from scratch

A small Python library and set of experiments that implement the core of derivatives pricing
(Hull, *Options, Futures and Other Derivatives*) and test **when the theory works in practice**. It then uses the same tools the way a
sales desk would: to **structure and pitch hedging and investment solutions to clients**.

<p align="center">
  <img src="figures/2_hedging_error.png" width="85%">
</p>
<p align="center">
  <img src="figures/6_fx_hedging.png" width="48%">
  <img src="figures/8_reverse_convertible.png" width="48%">
</p>

📄 **Write-ups:** [Pricing & hedging report](Options_Lab_Report.pdf) · [Client solutions (FX hedging, capital-protected note, reverse convertible)](Client_Solutions.pdf)

| Module | What it does |
|---|---|
| `optlab/pricing.py` | Black-Scholes-Merton (with dividend yield), analytical Greeks, CRR binomial tree (European & American), Monte Carlo with antithetic variates, implied volatility (Newton-Raphson + bisection fallback) |
| `optlab/hedging.py` | GBM path simulation, discrete delta-hedging of a short call, minimum-variance futures hedge ratio, beta hedging with index futures |
| `optlab/solutions.py` | Client solutions: FX forward / option / zero-cost collar for an exporter, capital-protected note (participation, cap), reverse convertible (enhanced coupon) |
| `optlab/jumps.py` | Merton jump-diffusion pricer, used to generate an implied-volatility skew |
| `tests/` | 18 unit tests: Hull reference values, put-call parity, no-arbitrage bounds, tree→BS convergence, American vs European, MC error, IV round-trip, Greeks vs finite differences, hedging error, FX parity, zero-cost collar, product budget identities |
| `scripts/run_experiments.py` | Reproduces figures 1-5 and `results.json` |
| `scripts/client_cases.py` | Reproduces the client case studies (figures 6-8, `client_cases.json`) |

## Key findings

1. **Binomial → Black-Scholes.** The CRR price of an ATM call oscillates around the BSM value
   (9.4134) and converges: error −0.04 at 50 steps, −0.005 at 400 steps.
   An American put (K = 110) is worth 0.68 more than its European twin: that's the early-exercise premium.
2. **Discrete delta hedging.** For a short ATM 1-year call hedged under BSM assumptions, the std of
   the hedged P&L falls from 21% of the premium (monthly rebalancing) to 4.5% (daily) and
   2.3% (1,000×/year). The error scales like **1/√N**, as theory predicts.
3. **A hedged option is a volatility trade.** Selling the call at 20% implied vol and delta-hedging
   gives a mean P&L that matches BS(σ<sub>implied</sub>) − BS(σ<sub>realised</sub>):
   about +1.9 if realised vol is 15%, about −2.0 if it is 25%.
4. **Cross-hedging with futures.** Replicating Hull's jet-fuel example, the regression hedge ratio
   (h* ≈ 0.75 estimated vs 0.78 true) removes 86% of the variance out-of-sample, against 79% for a naive h = 1.
5. **Why there is a skew.** Prices from a jump-diffusion with negative jumps, inverted through
   Black-Scholes, give an implied vol of 22.5% at K = 70 vs 17.4% ATM: the equity skew comes from crash risk.

## Client solutions (sales perspective)

*Illustrative market parameters, not live quotes.*

6. **Exporter receiving USD 10m in 6 months.** Forward at 1.1111 locks EUR 9.00m for free. An ATM-forward
   USD put costs EUR 0.20m (2.2%) but keeps the upside. A **zero-cost collar** guarantees EUR 8.73m–9.29m
   (EUR/USD 1.1454 / 1.0765) with no premium. Includes a "which solution for which client" matrix and a 3-sentence pitch.
7. **5-year capital-protected note on Euro Stoxx 50.** Zero-coupon 88.2% + 1% fees leaves a 10.8% option budget,
   giving **84% participation**, or 110% with a 50% cap. Shows how rates, vol and dividends drive participation.
8. **1-year reverse convertible (bond + short put).** Coupon of **13.2%** with a 100% strike or 9.1% with a 90% strike:
   the investor earns the volatility premium and is short vol.

## Run it

```bash
pip install numpy scipy matplotlib pytest
pytest -q                          # 18 tests
python scripts/run_experiments.py  # figures 1-5 + results.json (~20 s)
python scripts/client_cases.py     # figures 6-8 + client_cases.json
```

## Limitations / next steps
- Constant volatility and rates in the hedging simulations; no transaction costs yet
  (adding a bid-ask cost would show the trade-off between hedging error and cost).
- Next: calibrate implied vols to real option quotes (e.g. Euro Stoxx 50 or S&P 500 chains)
  and backtest a delta-hedged straddle on historical data.
