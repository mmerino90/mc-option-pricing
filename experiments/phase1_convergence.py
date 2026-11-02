"""Phase 1: Monte Carlo convergence to Black–Scholes.

Produces:
  results/phase1_convergence.png  (error vs N on log-log + estimate with 95% CI)
  results/phase1_table.md         (errors against Black–Scholes)
Run from the repo root:  python -m experiments.phase1_convergence
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcpricer import bs_price, mc_price

sys.stdout.reconfigure(encoding="utf-8")  # the Windows console can't print "σ" by default
P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
OUT = Path("results")
OUT.mkdir(exist_ok=True)

exact = bs_price(**P)
Ns = np.unique(np.logspace(2, 6, 17).astype(int))

# 1) Typical error (RMSE) averaged over many repetitions: a single error is too noisy
REPS = 100
rmse, mean_se, coverage = [], [], []
for N in Ns:
    runs = [mc_price(**P, n=N, seed=1000 * N + k) for k in range(REPS)]
    prices, ses = np.array(runs).T
    rmse.append(np.sqrt(np.mean((prices - exact) ** 2)))
    mean_se.append(ses.mean())
    coverage.append(np.mean(np.abs(prices - exact) <= 1.96 * ses))

slope = np.polyfit(np.log(Ns), np.log(rmse), 1)[0]

# 2) A single sequence of estimates with its 95% CI
single = np.array([mc_price(**P, n=N, seed=7) for N in Ns])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.loglog(Ns, rmse, "o-", label=f"actual RMSE ({REPS} repetitions)")
ax1.loglog(Ns, mean_se, "s--", label="estimated standard error σ̂/√N")
ax1.loglog(Ns, rmse[0] * np.sqrt(Ns[0] / Ns), "k:", label="reference ∝ 1/√N")
ax1.set(xlabel="N simulations", ylabel="error", title=f"Convergence (slope = {slope:.3f})")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

ax2.semilogx(Ns, single[:, 0], "o-", label="MC estimate")
ax2.fill_between(Ns, single[:, 0] - 1.96 * single[:, 1], single[:, 0] + 1.96 * single[:, 1],
                 alpha=0.25, label="95% CI")
ax2.axhline(exact, color="k", ls="--", label=f"Black–Scholes = {exact:.4f}")
ax2.set(xlabel="N simulations", ylabel="call price", title="Estimate and confidence interval")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase1_convergence.png", dpi=150)

# 3) Error table against Black–Scholes for several strikes
N = 1_000_000
rows = ["| Type | K | Black–Scholes | Monte Carlo | Error | Std. error | \\|error\\|/SE |",
        "|:--|--:|--:|--:|--:|--:|--:|"]
for kind in ("call", "put"):
    for K in (80, 90, 100, 110, 120):
        p = {**P, "K": float(K)}
        bs = bs_price(**p, kind=kind)
        mc, se = mc_price(**p, n=N, kind=kind, seed=K)
        rows.append(f"| {kind} | {K} | {bs:.4f} | {mc:.4f} | {mc - bs:+.4f} | {se:.4f} | {abs(mc - bs) / se:.2f} |")
table = f"N = {N:,} simulations, S0=100, T=1, r=5%, σ=20%\n\n" + "\n".join(rows)
(OUT / "phase1_table.md").write_text(table + "\n", encoding="utf-8")

print(f"Black–Scholes call = {exact:.4f}")
print(f"Log-log slope of the RMSE = {slope:.3f} (theory: -0.5)")
print(f"Average 95% CI coverage = {np.mean(coverage):.1%} (theory: 95%)")
print(table)
