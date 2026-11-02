# Monte Carlo option pricing engine

A computational finance learning project: pricing options by Monte Carlo in Python (and later in C++ via pybind11), always checked against analytical formulas.

## Phases

| Phase | Content | Status |
|---|---|---|
| 1 | Black–Scholes, geometric Brownian motion, 1/√N convergence | ✅ |
| 2 | Variance reduction (antithetic variates, control variate, Sobol) | ✅ |
| 3 | Greeks: pathwise vs finite differences | ⏳ |
| 4 | Exotics: Asian, barrier, American (Longstaff–Schwartz) | ⏳ |
| 5 | C++ core + pybind11, speedup | ⏳ |

## Results

### Phase 1: convergence

![convergence](results/phase1_convergence.png)

The log-log slope of the error is **−0.511** (theory: −0.5), and the 95% confidence interval contains the Black–Scholes price **94.4%** of the time. Full error table in [results/phase1_table.md](results/phase1_table.md).

### Phase 2: variance reduction

![variance reduction](results/phase2_variance_reduction.png)

For an at-the-money call with the same number of simulations:
- antithetic variates cut the variance **2×**;
- the control variate cuts it **7×**;
- Sobol (quasi-Monte Carlo) converges like **1/N** (slope −1.01) instead of 1/√N.

Both antithetics and the control variate work far better for in-the-money calls. Table with timings and efficiency in [results/phase2_table.md](results/phase2_table.md).

## Usage

```bash
pip install -r requirements.txt
python -m pytest
python -m experiments.phase1_convergence
python -m experiments.phase2_variance_reduction
```
