At-the-money call, N = 262,144 simulated prices, Black–Scholes = 10.4506

| Method | Price | Std. error | Variance reduction | Time (ms) | Relative cost | Efficiency gain |
|:--|--:|--:|--:|--:|--:|--:|
| Plain Monte Carlo | 10.4622 | 0.02875 | 1.0× | 21.7 | 1.00× | 1.0× |
| Antithetic | 10.4443 | 0.02028 | 2.0× | 20.7 | 0.96× | 2.1× |
| Control variate | 10.4471 | 0.01094 | 6.9× | 35.4 | 1.63× | 4.2× |
| Sobol (QMC) | 10.4505 | 0.00009 | 108049.6× | 46.9 | 2.16× | 50014.1× |

Note: Sobol's error falls like 1/N rather than 1/√N, so its "variance reduction" is not a fixed factor: it grows with N. This figure only holds for this N.
