"""Fase 2: reducción de varianza (antitéticas, variable de control, Sobol).

Genera:
  results/phase2_variance_reduction.png  (convergencia por método + factor de reducción por strike)
  results/phase2_table.md                (varianza, tiempo y eficiencia de cada método)
Ejecutar desde la raíz del repo:  python -m experiments.phase2_variance_reduction
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
    "Monte Carlo simple": mc_price,
    "Antitéticas": mc_antithetic,
    "Variable de control": mc_control_variate,
    "Sobol (QMC)": mc_sobol,
}
exact = bs_price(**P)

# 1) Convergencia: RMSE frente a Black–Scholes, 50 repeticiones por N
Ns = 2 ** np.arange(8, 19)
REPS = 50
rmse = {}
for name, f in METHODS.items():
    extra = {"replicates": 1} if f is mc_sobol else {}  # aquí medimos el error real, no el SE
    rmse[name] = [
        np.sqrt(np.mean([(f(**P, n=N, seed=1000 * N + k, **extra)[0] - exact) ** 2 for k in range(REPS)]))
        for N in Ns
    ]
slopes = {name: float(np.polyfit(np.log(Ns), np.log(e), 1)[0]) for name, e in rmse.items()}

# 2) Varianza, tiempo y eficiencia a igual N
N = 2**18


def best_time(f, repeats=7):
    times = []
    for k in range(repeats):
        t0 = time.perf_counter()
        f(**P, n=N, seed=k)
        times.append(time.perf_counter() - t0)
    return min(times)  # el mínimo es la medida menos contaminada por otros procesos


rows = []
for name, f in METHODS.items():
    price, se = f(**P, n=N, seed=3)
    rows.append((name, price, se, se**2 * N, best_time(f)))  # se²·N = varianza por simulación
var0, t0 = rows[0][3], rows[0][4]
lines = [
    f"Call at-the-money, N = {N:,} precios simulados, Black–Scholes = {exact:.4f}\n",
    "| Método | Precio | Error estándar | Reducción de varianza | Tiempo (ms) | Coste relativo | Ganancia de eficiencia |",
    "|:--|--:|--:|--:|--:|--:|--:|",
]
for name, price, se, var, t in rows:
    lines.append(f"| {name} | {price:.4f} | {se:.5f} | {var0 / var:.1f}× | {1000 * t:.1f} | {t / t0:.2f}× | "
                 f"{(var0 * t0) / (var * t):.1f}× |")

# 3) Factor de reducción de varianza según el strike (dónde funciona cada técnica)
Ks = np.arange(60, 145, 5, dtype=float)
factor = {"Antitéticas": [], "Variable de control": []}
for K in Ks:
    p = {**P, "K": K}
    _, se0 = mc_price(**p, n=N, seed=9)
    factor["Antitéticas"].append((se0 / mc_antithetic(**p, n=N, seed=9)[1]) ** 2)
    factor["Variable de control"].append((se0 / mc_control_variate(**p, n=N, seed=9)[1]) ** 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for name, e in rmse.items():
    ax1.loglog(Ns, e, "o-", ms=4, label=f"{name} (pendiente {slopes[name]:.2f})")
ax1.set(xlabel="N precios simulados", ylabel="RMSE frente a Black–Scholes",
        title=f"Convergencia ({REPS} repeticiones por punto)")
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)

for name, fs in factor.items():
    ax2.semilogy(Ks, fs, "o-", ms=4, label=name)
ax2.axvline(P["S0"], color="k", ls=":", lw=1)
ax2.axhline(1, color="gray", lw=1)
ax2.set(xlabel="strike K  (S0 = 100)", ylabel="varianza simple / varianza del método",
        title="Reducción de varianza según el strike (call)")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase2_variance_reduction.png", dpi=150)

lines.append("\nOjo: el error de Sobol baja como 1/N y no como 1/√N, así que su \"reducción de varianza\" "
             "no es un factor fijo: crece con N. Esta cifra solo vale para este N.")
table = "\n".join(lines)
(OUT / "phase2_table.md").write_text(table + "\n", encoding="utf-8")
print(table)
print("\nPendientes log-log:", {k: round(v, 3) for k, v in slopes.items()})
print("Factor por strike:")
for K, a, c in zip(Ks, factor["Antitéticas"], factor["Variable de control"]):
    print(f"  K={K:5.0f}  antitéticas {a:6.2f}×   control {c:8.2f}×")
