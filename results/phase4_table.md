## Asian call (arithmetic average of 12 monthly fixings)

S0=100, T=1, r=5%, σ=20%, N = 100,000 paths. Each cell is estimate ± standard error.

| K | European (BS) | Geometric Asian (exact) | Arithmetic, plain MC | Arithmetic, control variate | Variance reduction |
|--:|--:|--:|--:|--:|--:|
| 90 | 16.6994 | 12.6430 | 12.8894 ± 0.0345 | 12.91957 ± 0.00082 | 1,758× |
| 100 | 10.4506 | 5.9402 | 6.1258 ± 0.0267 | 6.15584 ± 0.00074 | 1,318× |
| 110 | 6.0401 | 2.1443 | 2.2650 ± 0.0169 | 2.29098 ± 0.00068 | 620× |

## Down-and-out call (S0=100, K=100, barrier H=90)

Continuous-monitoring closed form = 8.6655. N = 100,000 paths.

| Monitoring dates | Discrete MC | BGK-shifted formula | MC + Brownian bridge |
|--:|--:|--:|--:|
| 4 | 10.0101 ± 0.0469 | 9.9626 | 8.6973 ± 0.0447 |
| 12 | 9.5996 ± 0.0468 | 9.5802 | 8.7044 ± 0.0455 |
| 52 | 9.0991 ± 0.0463 | 9.1736 | 8.5761 ± 0.0456 |
| 252 | 8.9620 ± 0.0466 | 8.9139 | 8.7080 ± 0.0463 |
| 1000 | 8.8135 ± 0.0463 | 8.7941 | 8.6862 ± 0.0461 |

## American put (K=40, T=1, r=6%, σ=20%) — the Longstaff–Schwartz (2001) example

LSM: 50 exercise dates, cubic polynomial basis, 100,000 training + 100,000 pricing paths. Trees: 5,000 steps; the Bermudan tree only allows exercise on the same 50 dates.

| S0 | European (BS) | American (tree) | Bermudan, 50 dates (tree) | LSM | LSM − Bermudan | Early-exercise premium |
|--:|--:|--:|--:|--:|--:|--:|
| 36 | 3.8443 | 4.4867 | 4.4779 | 4.4775 ± 0.0092 | -0.0005 | 0.6424 |
| 38 | 2.8519 | 3.2572 | 3.2502 | 3.2539 ± 0.0094 | +0.0037 | 0.4052 |
| 40 | 2.0664 | 2.3195 | 2.3140 | 2.3185 ± 0.0086 | +0.0045 | 0.2531 |
| 42 | 1.4645 | 1.6213 | 1.6171 | 1.6172 ± 0.0075 | +0.0001 | 0.1567 |
| 44 | 1.0169 | 1.1130 | 1.1099 | 1.1044 ± 0.0065 | -0.0055 | 0.0961 |

### Effect of the regression basis (S0 = 36, Bermudan tree = 4.4779)

| Polynomial degree | LSM price | Gap to tree |
|--:|--:|--:|
| 1 | 4.4224 ± 0.0099 | -0.0555 |
| 2 | 4.4697 ± 0.0092 | -0.0082 |
| 3 | 4.4775 ± 0.0092 | -0.0005 |
| 4 | 4.4785 ± 0.0091 | +0.0005 |
| 5 | 4.4789 ± 0.0091 | +0.0010 |
