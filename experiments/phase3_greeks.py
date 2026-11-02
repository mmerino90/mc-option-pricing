"""Phase 3: delta and vega by pathwise and finite differences, against Black–Scholes.

Produces:
  results/phase3_greeks.png  (Greeks vs strike + error of finite differences vs bump size)
  results/phase3_table.md    (all estimators against the analytical values)
Run from the repo root:  python -m experiments.phase3_greeks
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcpricer import bs_delta, bs_vega, fd_greeks, pathwise_greeks

sys.stdout.reconfigure(encoding="utf-8")
P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
OUT = Path("results")
OUT.mkdir(exist_ok=True)
ANALYTIC = {"delta": bs_delta, "vega": bs_vega}

# 1) Table: every estimator against the analytical Greek, N = 100 000
N = 100_000
lines = [
    f"N = {N:,} simulations, S0=100, T=1, r=5%, σ=20%. Finite differences use h = 1 (spot) and h = 0.01 (vol). "
    "Each cell is estimate ± standard error.\n",
    "| Greek | Type | K | Black–Scholes | Pathwise | FD, common numbers | FD, independent numbers |",
    "|:--|:--|--:|--:|--:|--:|--:|",
]
for greek in ("delta", "vega"):
    for kind in ("call", "put"):
        for K in (80.0, 100.0, 120.0):
            p = {**P, "K": K}
            cells = [pathwise_greeks(**p, n=N, kind=kind, seed=1)[greek],
                     fd_greeks(**p, n=N, kind=kind, seed=1, common=True)[greek],
                     fd_greeks(**p, n=N, kind=kind, seed=1, common=False)[greek]]
            row = " | ".join(f"{e:.4f} ± {s:.4f}" for e, s in cells)
            lines.append(f"| {greek} | {kind} | {K:.0f} | {ANALYTIC[greek](**p, kind=kind):.4f} | {row} |")

# 2) Greeks across strikes: analytical curve vs pathwise estimates
Ks = np.linspace(60, 140, 17)
Ks_fine = np.linspace(60, 140, 200)

# 3) Error of finite-difference delta vs bump size h (ATM call)
hs = np.logspace(-3, 1.5, 16)
N_SWEEP, REPS = 10_000, 200
exact_delta = bs_delta(**P)


def rmse_delta(**kw):
    ests = [fd_greeks(**P, n=N_SWEEP, seed=k, **kw)["delta"][0] for k in range(REPS)]
    return np.sqrt(np.mean((np.array(ests) - exact_delta) ** 2))


rmse_common = [rmse_delta(h_spot=h, common=True) for h in hs]
rmse_indep = [rmse_delta(h_spot=h, common=False) for h in hs]
pw = [pathwise_greeks(**P, n=N_SWEEP, seed=k)["delta"][0] for k in range(REPS)]
rmse_pw = np.sqrt(np.mean((np.array(pw) - exact_delta) ** 2))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, greek in zip(axes[:2], ("delta", "vega")):
    for kind, color in (("call", "C0"), ("put", "C1")):
        if greek == "vega" and kind == "put":
            continue  # identical to the call
        ax.plot(Ks_fine, ANALYTIC[greek](P["S0"], Ks_fine, P["T"], P["r"], P["sigma"], kind=kind),
                color=color, label=f"Black–Scholes ({kind})" if greek == "delta" else "Black–Scholes")
        est = np.array([pathwise_greeks(**{**P, "K": K}, n=N, kind=kind, seed=2)[greek] for K in Ks])
        ax.errorbar(Ks, est[:, 0], yerr=1.96 * est[:, 1], fmt="o", ms=4, color=color, mfc="white",
                    label=f"pathwise ({kind})" if greek == "delta" else "pathwise")
    ax.axvline(P["S0"], color="k", ls=":", lw=1)
    ax.set(xlabel="strike K  (S0 = 100)", ylabel=greek, title=f"{greek.capitalize()} across strikes")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax = axes[2]
ax.loglog(hs, rmse_indep, "o-", ms=4, label="FD, independent numbers")
ax.loglog(hs, rmse_common, "o-", ms=4, label="FD, common numbers")
ax.axhline(rmse_pw, color="k", ls="--", label="pathwise (no h)")
ax.set(xlabel="bump size h  (S0 = 100)", ylabel="RMSE of delta",
       title=f"Finite-difference delta vs bump (N = {N_SWEEP:,}, {REPS} reps)")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase3_greeks.png", dpi=150)

table = "\n".join(lines)
(OUT / "phase3_table.md").write_text(table + "\n", encoding="utf-8")
print(table)
print(f"\nDelta RMSE (N={N_SWEEP:,}): pathwise {rmse_pw:.5f}")
for h, c, i in zip(hs, rmse_common, rmse_indep):
    print(f"  h={h:8.4f}  common {c:.5f}   independent {i:.5f}")
