"""Phase 4: Asian, barrier and American options.

Produces:
  results/phase4_exotics.png  (Asian payoff correlation, barrier monitoring bias, exercise boundary)
  results/phase4_table.md     (one table per option)
Run from the repo root:  python -m experiments.phase4_exotics
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcpricer import (binomial_put, bs_price, down_and_out_call_price, geometric_asian_call_price,
                      mc_american_put_lsm, mc_asian_call, mc_down_and_out_call, simulate_paths)

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
lines = []

# ---------- 1) Asian call: geometric average as control variate for the arithmetic one ----------
P = dict(S0=100.0, T=1.0, r=0.05, sigma=0.2)
M_ASIAN, N = 12, 100_000
lines += [
    f"## Asian call (arithmetic average of {M_ASIAN} monthly fixings)\n",
    f"S0=100, T=1, r=5%, σ=20%, N = {N:,} paths. Each cell is estimate ± standard error.\n",
    "| K | European (BS) | Geometric Asian (exact) | Arithmetic, plain MC | Arithmetic, control variate | Variance reduction |",
    "|--:|--:|--:|--:|--:|--:|",
]
for K in (90.0, 100.0, 110.0):
    plain, se_p = mc_asian_call(**P, K=K, n=N, m=M_ASIAN, seed=1, control=False)
    cv, se_c = mc_asian_call(**P, K=K, n=N, m=M_ASIAN, seed=1)
    lines.append(f"| {K:.0f} | {bs_price(**P, K=K):.4f} | {geometric_asian_call_price(**P, K=K, m=M_ASIAN):.4f} | "
                 f"{plain:.4f} ± {se_p:.4f} | {cv:.5f} ± {se_c:.5f} | {(se_p / se_c) ** 2:,.0f}× |")

paths = simulate_paths(**P, n=2000, m=M_ASIAN, rng=np.random.default_rng(2))
disc = np.exp(-P["r"] * P["T"])
arith_payoff = disc * np.maximum(paths.mean(axis=1) - 100, 0)
geo_payoff = disc * np.maximum(np.exp(np.log(paths).mean(axis=1)) - 100, 0)
corr = np.corrcoef(arith_payoff, geo_payoff)[0, 1]

# ---------- 2) Down-and-out call: discrete monitoring bias ----------
B = dict(S0=100.0, K=100.0, H=90.0, T=1.0, r=0.05, sigma=0.2)
exact_barrier = down_and_out_call_price(**B)
ms = [4, 12, 52, 252, 1000]
BETA = 0.5826  # Broadie–Glasserman–Kou constant: -zeta(1/2) / sqrt(2 pi)
lines += [
    f"\n## Down-and-out call (S0=100, K=100, barrier H=90)\n",
    f"Continuous-monitoring closed form = {exact_barrier:.4f}. N = {N:,} paths.\n",
    "| Monitoring dates | Discrete MC | BGK-shifted formula | MC + Brownian bridge |",
    "|--:|--:|--:|--:|",
]
discrete, bridge, bgk = [], [], []
for m in ms:
    discrete.append(mc_down_and_out_call(**B, n=N, m=m, seed=3))
    bridge.append(mc_down_and_out_call(**B, n=N, m=m, seed=3, bridge=True))
    H_shift = B["H"] * np.exp(-BETA * B["sigma"] * np.sqrt(B["T"] / m))
    bgk.append(down_and_out_call_price(**{**B, "H": H_shift}))
    lines.append(f"| {m} | {discrete[-1][0]:.4f} ± {discrete[-1][1]:.4f} | {bgk[-1]:.4f} | "
                 f"{bridge[-1][0]:.4f} ± {bridge[-1][1]:.4f} |")
discrete, bridge = np.array(discrete), np.array(bridge)

# ---------- 3) American put: Longstaff–Schwartz vs binomial tree ----------
A = dict(K=40.0, T=1.0, r=0.06, sigma=0.2)
M_EX = 50
lines += [
    f"\n## American put (K=40, T=1, r=6%, σ=20%) — the Longstaff–Schwartz (2001) example\n",
    f"LSM: {M_EX} exercise dates, cubic polynomial basis, {N:,} training + {N:,} pricing paths. "
    f"Trees: 5,000 steps; the Bermudan tree only allows exercise on the same {M_EX} dates.\n",
    "| S0 | European (BS) | American (tree) | Bermudan, 50 dates (tree) | LSM | LSM − Bermudan | Early-exercise premium |",
    "|--:|--:|--:|--:|--:|--:|--:|",
]
for S0 in (36.0, 38.0, 40.0, 42.0, 44.0):
    euro = bs_price(S0=S0, **A, kind="put")
    amer = binomial_put(S0=S0, **A, steps=5000)
    berm = binomial_put(S0=S0, **A, steps=5000, exercise_every=100)
    lsm, se = mc_american_put_lsm(S0=S0, **A, n=N, m=M_EX, seed=4)
    lines.append(f"| {S0:.0f} | {euro:.4f} | {amer:.4f} | {berm:.4f} | {lsm:.4f} ± {se:.4f} | "
                 f"{lsm - berm:+.4f} | {amer - euro:.4f} |")

berm36 = binomial_put(S0=36.0, **A, steps=5000, exercise_every=100)
lines += [
    f"\n### Effect of the regression basis (S0 = 36, Bermudan tree = {berm36:.4f})\n",
    "| Polynomial degree | LSM price | Gap to tree |",
    "|--:|--:|--:|",
]
for degree in (1, 2, 3, 4, 5):
    lsm, se = mc_american_put_lsm(S0=36.0, **A, n=N, m=M_EX, seed=4, degree=degree)
    lines.append(f"| {degree} | {lsm:.4f} ± {se:.4f} | {lsm - berm36:+.4f} |")

_, tree_boundary = binomial_put(S0=36.0, **A, steps=5000, return_boundary=True)
_, _, lsm_boundary = mc_american_put_lsm(S0=36.0, **A, n=N, m=M_EX, seed=4, return_boundary=True)

# ---------- Figure ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ax = axes[0]
ax.scatter(geo_payoff, arith_payoff, s=6, alpha=0.5)
lim = arith_payoff.max() * 1.05
ax.plot([0, lim], [0, lim], "k:", lw=1, label="y = x")
ax.set(xlabel="geometric Asian payoff", ylabel="arithmetic Asian payoff",
       title=f"Asian call, same paths: correlation {corr:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.errorbar(ms, discrete[:, 0], yerr=1.96 * discrete[:, 1], fmt="o-", ms=4, capsize=3, label="discrete MC")
ax.plot(ms, bgk, "s--", ms=4, label="BGK-shifted formula")
ax.errorbar(ms, bridge[:, 0], yerr=1.96 * bridge[:, 1], fmt="^", ms=5, capsize=3, label="MC + Brownian bridge")
ax.axhline(exact_barrier, color="k", ls="--", label=f"continuous formula = {exact_barrier:.3f}")
ax.set_xscale("log")
ax.set(xlabel="monitoring dates m", ylabel="down-and-out call price", title="Barrier: discrete monitoring bias")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
ax.plot(np.arange(5000) / 5000, tree_boundary, lw=1.5, label="binomial tree (continuous exercise)")
ax.plot(np.arange(1, M_EX + 1) / M_EX, lsm_boundary, "o", ms=4, label="Longstaff–Schwartz (50 dates)")
ax.axhline(A["K"], color="gray", lw=1)
ax.set(xlabel="time t (years)", ylabel="exercise if S below",
       title="American put (S0=36, K=40): exercise boundary")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase4_exotics.png", dpi=150)

table = "\n".join(lines)
(OUT / "phase4_table.md").write_text(table + "\n", encoding="utf-8")
print(table)
print(f"\nAsian payoff correlation (2000 paths, K=100): {corr:.5f}")
