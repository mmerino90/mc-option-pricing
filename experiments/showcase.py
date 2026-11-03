"""One-image summary of the project (1080 × 1350, LinkedIn portrait format).

Produces results/showcase.png. Speedup numbers are read from results/phase5_table.md,
so run phase5_speedup first.  Run from the repo root:  python -m experiments.showcase
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcpricer import binomial_put, bs_price, mc_american_put_lsm, mc_control_variate, mc_price, mc_sobol

OUT = Path("results")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4, fixed order

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID, "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "legend.frameon": False,
})

# ---------- data ----------
rows = [line for line in (OUT / "phase5_table.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("| ") and " s (" in line]
options = [row.split("|")[1].strip() for row in rows]
times = np.array([[float(x) for x in re.findall(r"([\d,.]+) s \(", row)][:4] for row in rows])
impls = ["Pure Python", "NumPy", "C++, 1 thread", "C++, 12 threads"]

P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
exact = bs_price(**P)
Ns = 2 ** np.arange(8, 17)
REPS = 30
conv = {}
for name, f, extra in (("Plain Monte Carlo   ∝ 1/√N", mc_price, {}),
                       ("Control variate", mc_control_variate, {}),
                       ("Sobol quasi-MC   ∝ 1/N", mc_sobol, {"replicates": 1})):
    conv[name] = [np.sqrt(np.mean([(f(**P, n=N, seed=1000 * N + k, **extra)[0] - exact) ** 2
                                   for k in range(REPS)])) for N in Ns]

A = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2)
_, tree_boundary = binomial_put(**A, steps=2000, return_boundary=True)
_, _, lsm_boundary = mc_american_put_lsm(**A, n=50_000, m=50, seed=4, return_boundary=True)

# ---------- layout ----------
fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
fig.text(0.06, 0.955, "Monte Carlo option pricing engine", fontsize=30, weight="bold")
fig.text(0.06, 0.928, "Python + a multithreaded C++ core, every result checked against a closed form or a tree",
         fontsize=13.5, color=INK_2)

stats = [("86×", "faster than pure Python\n(C++ core, 12 threads)"),
         ("1,800×", "less variance on Asian options\n(geometric control variate)"),
         ("0.0004", "American put: Longstaff–Schwartz\nvs binomial tree")]
for i, (value, label) in enumerate(stats):
    x = 0.06 + i * 0.31
    fig.text(x, 0.855, value, fontsize=40, weight="bold", color=BLUE)
    fig.text(x, 0.838, label, fontsize=12, color=INK_2, va="top", linespacing=1.4)

# speedup: one panel, log scale, one colour per implementation
ax = fig.add_axes([0.1, 0.44, 0.84, 0.31])
width = 0.2
x = np.arange(len(options))
for i, (name, color) in enumerate(zip(impls, ("#d6d4cd", "#9c9a92", ORANGE, BLUE))):
    vals = times[:, i]
    bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.9, color=color, label=name, zorder=3)
    ax.bar_label(bars, labels=[f"{v:.2g}s" for v in vals], fontsize=10, color=INK_2, padding=3)
ax.set_yscale("log")
ax.set_xticks(x, [o.replace(" (", "\n(") for o in options])
ax.tick_params(axis="x", length=0)
ax.set_ylabel("seconds per million paths (log)")
ax.set_title("Same simulation, four implementations", loc="left", fontsize=15, weight="bold", pad=14)
ax.legend(ncol=4, loc="upper left", fontsize=11, bbox_to_anchor=(0, 1.0))
ax.set_ylim(top=times.max() * 12)
ax.grid(axis="x", visible=False)

# convergence
ax = fig.add_axes([0.1, 0.08, 0.37, 0.27])
for (name, err), color in zip(conv.items(), (INK_2, ORANGE, BLUE)):
    ax.loglog(Ns, err, "o-", ms=5, lw=2, color=color, label=name)
ax.set(xlabel="simulated paths N", ylabel="pricing error (RMSE)")
ax.set_title("Error vs simulations", loc="left", fontsize=15, weight="bold", pad=10)
ax.legend(fontsize=10.5, loc="lower left")

# exercise boundary
ax = fig.add_axes([0.58, 0.08, 0.36, 0.27])
ax.plot(np.arange(2000) / 2000, tree_boundary, color=INK_2, lw=2, label="binomial tree")
ax.plot(np.arange(1, 51) / 50, lsm_boundary, "o", ms=6, color=BLUE, mec=SURFACE, mew=1, label="learned by regression")
ax.set(xlabel="time (years)", ylabel="exercise when stock is below")
ax.set_title("American put: when to exercise", loc="left", fontsize=15, weight="bold", pad=10)
ax.legend(fontsize=10.5, loc="upper left")

fig.text(0.06, 0.018, "github.com/mmerino90/mc-option-pricing", fontsize=12, color=INK_2)
fig.savefig(OUT / "showcase.png", dpi=100)
print("saved", OUT / "showcase.png")
