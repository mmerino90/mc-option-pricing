"""Fase 1: convergencia de Monte Carlo hacia Black–Scholes.

Genera:
  results/phase1_convergence.png  (error vs N en log-log + estimación con IC 95%)
  results/phase1_table.md         (tabla de errores frente a Black–Scholes)
Ejecutar desde la raíz del repo:  python -m experiments.phase1_convergence
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcpricer import bs_price, mc_price

sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows no imprime "σ" por defecto
P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
OUT = Path("results")
OUT.mkdir(exist_ok=True)

exact = bs_price(**P)
Ns = np.unique(np.logspace(2, 6, 17).astype(int))

# 1) Error típico (RMSE) promediando muchas repeticiones: un solo error es muy ruidoso
REPS = 100
rmse, mean_se, coverage = [], [], []
for N in Ns:
    runs = [mc_price(**P, n=N, seed=1000 * N + k) for k in range(REPS)]
    prices, ses = np.array(runs).T
    rmse.append(np.sqrt(np.mean((prices - exact) ** 2)))
    mean_se.append(ses.mean())
    coverage.append(np.mean(np.abs(prices - exact) <= 1.96 * ses))

slope = np.polyfit(np.log(Ns), np.log(rmse), 1)[0]

# 2) Una sola secuencia de estimaciones con su IC al 95%
single = np.array([mc_price(**P, n=N, seed=7) for N in Ns])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.loglog(Ns, rmse, "o-", label=f"RMSE real ({REPS} repeticiones)")
ax1.loglog(Ns, mean_se, "s--", label="error estándar estimado σ̂/√N")
ax1.loglog(Ns, rmse[0] * np.sqrt(Ns[0] / Ns), "k:", label="referencia ∝ 1/√N")
ax1.set(xlabel="N simulaciones", ylabel="error", title=f"Convergencia (pendiente = {slope:.3f})")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

ax2.semilogx(Ns, single[:, 0], "o-", label="estimación MC")
ax2.fill_between(Ns, single[:, 0] - 1.96 * single[:, 1], single[:, 0] + 1.96 * single[:, 1],
                 alpha=0.25, label="IC 95%")
ax2.axhline(exact, color="k", ls="--", label=f"Black–Scholes = {exact:.4f}")
ax2.set(xlabel="N simulaciones", ylabel="precio call", title="Estimación e intervalo de confianza")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase1_convergence.png", dpi=150)

# 3) Tabla de errores frente a Black–Scholes para varios strikes
N = 1_000_000
rows = ["| Tipo | K | Black–Scholes | Monte Carlo | Error | Error estándar | |error|/SE |",
        "|---|---|---|---|---|---|---|"]
for kind in ("call", "put"):
    for K in (80, 90, 100, 110, 120):
        p = {**P, "K": float(K)}
        bs = bs_price(**p, kind=kind)
        mc, se = mc_price(**p, n=N, kind=kind, seed=K)
        rows.append(f"| {kind} | {K} | {bs:.4f} | {mc:.4f} | {mc - bs:+.4f} | {se:.4f} | {abs(mc - bs) / se:.2f} |")
table = f"N = {N:,} simulaciones, S0=100, T=1, r=5%, σ=20%\n\n" + "\n".join(rows)
(OUT / "phase1_table.md").write_text(table + "\n", encoding="utf-8")

print(f"Black–Scholes call = {exact:.4f}")
print(f"Pendiente log-log del RMSE = {slope:.3f} (teoría: -0.5)")
print(f"Cobertura media del IC 95% = {np.mean(coverage):.1%} (teoría: 95%)")
print(table)
