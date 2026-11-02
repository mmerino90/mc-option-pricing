"""Phase 5: how much faster is the C++ core?

Times four implementations of the same three simulations:
  pure Python loops · NumPy (the mcpricer package) · C++ single thread · C++ all cores
Produces:
  results/phase5_speedup.png  (time per million paths, log scale)
  results/phase5_table.md
Build the extension first:  python setup.py build_ext --inplace
Run from the repo root:     python -m experiments.phase5_speedup
"""
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mcpricer_cpp as cpp
from mcpricer import mc_asian_call, mc_down_and_out_call, mc_price

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
S0, K, H, T, r, sigma = 100.0, 100.0, 90.0, 1.0, 0.05, 0.2
M_ASIAN, M_BARRIER = 52, 252  # weekly averaging, daily barrier checks
CORES = os.cpu_count()


# ---------- pure-Python baselines: one path at a time, like the C++ code ----------

def py_european(n):
    drift, vol, total = (r - 0.5 * sigma**2) * T, sigma * math.sqrt(T), 0.0
    for _ in range(n):
        total += max(S0 * math.exp(drift + vol * random.gauss(0, 1)) - K, 0.0)
    return math.exp(-r * T) * total / n


def py_asian(n):
    dt = T / M_ASIAN
    drift, vol, total = (r - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), 0.0
    for _ in range(n):
        S, acc = S0, 0.0
        for _ in range(M_ASIAN):
            S *= math.exp(drift + vol * random.gauss(0, 1))
            acc += S
        total += max(acc / M_ASIAN - K, 0.0)
    return math.exp(-r * T) * total / n


def py_barrier(n):
    dt = T / M_BARRIER
    drift, vol, total = (r - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), 0.0
    for _ in range(n):
        S = S0
        for _ in range(M_BARRIER):
            S *= math.exp(drift + vol * random.gauss(0, 1))
            if S <= H:
                S = 0.0
                break
        total += max(S - K, 0.0)
    return math.exp(-r * T) * total / n


# (option name, {implementation: (function of n, n to time it with)})
CASES = [
    ("European call", {
        "Pure Python": (py_european, 50_000),
        "NumPy": (lambda n: mc_price(S0, K, T, r, sigma, n, seed=1), 2_000_000),
        "C++, 1 thread": (lambda n: cpp.european(S0, K, T, r, sigma, n, seed=1, threads=1), 2_000_000),
        f"C++, {CORES} threads": (lambda n: cpp.european(S0, K, T, r, sigma, n, seed=1), 2_000_000),
    }),
    (f"Asian call ({M_ASIAN} fixings)", {
        "Pure Python": (py_asian, 2_000),
        "NumPy": (lambda n: mc_asian_call(S0, K, T, r, sigma, n, M_ASIAN, seed=1, control=False), 200_000),
        "C++, 1 thread": (lambda n: cpp.asian_call(S0, K, T, r, sigma, n, M_ASIAN, seed=1, threads=1), 200_000),
        f"C++, {CORES} threads": (lambda n: cpp.asian_call(S0, K, T, r, sigma, n, M_ASIAN, seed=1), 200_000),
    }),
    (f"Barrier call ({M_BARRIER} checks)", {
        "Pure Python": (py_barrier, 1_000),
        "NumPy": (lambda n: mc_down_and_out_call(S0, K, H, T, r, sigma, n, M_BARRIER, seed=1), 200_000),
        "C++, 1 thread": (lambda n: cpp.down_and_out_call(S0, K, H, T, r, sigma, n, M_BARRIER, seed=1, threads=1), 200_000),
        f"C++, {CORES} threads": (lambda n: cpp.down_and_out_call(S0, K, H, T, r, sigma, n, M_BARRIER, seed=1), 200_000),
    }),
]


def seconds_per_million(f, n, repeats=3):
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        f(n)
        best = min(best, time.perf_counter() - t0)  # the minimum is the least disturbed measurement
    return best / n * 1_000_000


results = {}
for option, impls in CASES:
    results[option] = {name: seconds_per_million(f, n) for name, (f, n) in impls.items()}
    print(option, {k: f"{v:.3f}s" for k, v in results[option].items()})

impl_names = list(CASES[0][1])
lines = [
    f"Seconds to simulate one million paths (best of 3). {CORES} logical CPU cores. "
    "Speedups are relative to the NumPy version.\n",
    "| Option | " + " | ".join(impl_names) + " | Fastest vs pure Python |",
    "|:--|" + "--:|" * (len(impl_names) + 1),
]
for option, times in results.items():
    base = times["NumPy"]
    cells = [f"{t:,.3f} s ({base / t:,.1f}×)" for t in times.values()]
    cells.append(f"{times['Pure Python'] / min(times.values()):,.0f}×")
    lines.append(f"| {option} | " + " | ".join(cells) + " |")

fig, ax = plt.subplots(figsize=(11, 4.8))
width = 0.8 / len(impl_names)
x = np.arange(len(results))
for i, name in enumerate(impl_names):
    vals = [results[o][name] for o in results]
    bars = ax.bar(x + (i - (len(impl_names) - 1) / 2) * width, vals, width, label=name)
    ax.bar_label(bars, labels=[f"{v:.2g}s" for v in vals], fontsize=8, padding=2)
ax.set_yscale("log")
ax.set_xticks(x, list(results))
ax.set(ylabel="seconds per million paths (log scale)", title="Same simulation, four implementations")
ax.legend()
ax.grid(True, axis="y", which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase5_speedup.png", dpi=150)

table = "\n".join(lines)
(OUT / "phase5_table.md").write_text(table + "\n", encoding="utf-8")
print(table)
