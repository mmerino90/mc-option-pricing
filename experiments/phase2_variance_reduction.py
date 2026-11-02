"""Phase 2: variance reduction (antithetic variates, control variate, Sobol).

Produces:
  results/phase2_variance_reduction.png  (convergence per method + reduction factor by strike)
  results/phase2_table.md                (variance, time and efficiency of each method)
Run from the repo root:  python -m experiments.phase2_variance_reduction
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcpricer import bs_price, mc_antithetic, mc_control_variate, mc_price, mc_sobol

sys.stdout.reconfigure(encoding="utf-8")
P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
OUT = Path("results")
OUT.mkdir(exist_ok=True)

METHODS = {
    "Plain Monte Carlo": mc_price,
    "Antithetic": mc_antithetic,
    "Control variate": mc_control_variate,
    "Sobol (QMC)": mc_sobol,
}
exact = bs_price(**P)

# 1) Convergence: RMSE against Black–Scholes, 50 repetitions per N
Ns = 2 ** np.arange(8, 19)
REPS = 50
rmse = {}
for name, f in METHODS.items():
    extra = {"replicates": 1} if f is mc_sobol else {}  # here we measure the actual error, not the SE
    rmse[name] = [
        np.sqrt(np.mean([(f(**P, n=N, seed=1000 * N + k, **extra)[0] - exact) ** 2 for k in range(REPS)]))
        for N in Ns
    ]
slopes = {name: float(np.polyfit(np.log(Ns), np.log(e), 1)[0]) for name, e in rmse.items()}

# 2) Variance, time and efficiency at equal N
N = 2**18


def best_time(f, repeats=7):
    times = []
    for k in range(repeats):
        t0 = time.perf_counter()
        f(**P, n=N, seed=k)
        times.append(time.perf_counter() - t0)
    return min(times)  # the minimum is the measurement least polluted by other processes


rows = []
for name, f in METHODS.items():
    price, se = f(**P, n=N, seed=3)
    rows.append((name, price, se, se**2 * N, best_time(f)))  # se²·N = variance per simulation
var0, t0 = rows[0][3], rows[0][4]
lines = [
    f"At-the-money call, N = {N:,} simulated prices, Black–Scholes = {exact:.4f}\n",
    "| Method | Price | Std. error | Variance reduction | Time (ms) | Relative cost | Efficiency gain |",
    "|:--|--:|--:|--:|--:|--:|--:|",
]
for name, price, se, var, t in rows:
    lines.append(f"| {name} | {price:.4f} | {se:.5f} | {var0 / var:.1f}× | {1000 * t:.1f} | {t / t0:.2f}× | "
                 f"{(var0 * t0) / (var * t):.1f}× |")
lines.append("\nNote: Sobol's error falls like 1/N rather than 1/√N, so its \"variance reduction\" "
             "is not a fixed factor: it grows with N. This figure only holds for this N.")

# 3) Variance reduction factor by strike (where each technique works)
Ks = np.arange(60, 145, 5, dtype=float)
factor = {"Antithetic": [], "Control variate": []}
for K in Ks:
    p = {**P, "K": K}
    _, se0 = mc_price(**p, n=N, seed=9)
    factor["Antithetic"].append((se0 / mc_antithetic(**p, n=N, seed=9)[1]) ** 2)
    factor["Control variate"].append((se0 / mc_control_variate(**p, n=N, seed=9)[1]) ** 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for name, e in rmse.items():
    ax1.loglog(Ns, e, "o-", ms=4, label=f"{name} (slope {slopes[name]:.2f})")
ax1.set(xlabel="N simulated prices", ylabel="RMSE against Black–Scholes",
        title=f"Convergence ({REPS} repetitions per point)")
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)

for name, fs in factor.items():
    ax2.semilogy(Ks, fs, "o-", ms=4, label=name)
ax2.axvline(P["S0"], color="k", ls=":", lw=1)
ax2.axhline(1, color="gray", lw=1)
ax2.set(xlabel="strike K  (S0 = 100)", ylabel="plain variance / method variance",
        title="Variance reduction by strike (call)")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase2_variance_reduction.png", dpi=150)

table = "\n".join(lines)
(OUT / "phase2_table.md").write_text(table + "\n", encoding="utf-8")
print(table)
print("\nLog-log slopes:", {k: round(v, 3) for k, v in slopes.items()})
print("Factor by strike:")
for K, a, c in zip(Ks, factor["Antithetic"], factor["Control variate"]):
    print(f"  K={K:5.0f}  antithetic {a:6.2f}×   control {c:8.2f}×")
