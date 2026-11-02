Call at-the-money, N = 262,144 precios simulados, Black–Scholes = 10.4506

| Método | Precio | Error estándar | Reducción de varianza | Tiempo (ms) | Coste relativo | Ganancia de eficiencia |
|:--|--:|--:|--:|--:|--:|--:|
| Monte Carlo simple | 10.4622 | 0.02875 | 1.0× | 6.9 | 1.00× | 1.0× |
| Antitéticas | 10.4443 | 0.02028 | 2.0× | 6.7 | 0.98× | 2.1× |
| Variable de control | 10.4471 | 0.01094 | 6.9× | 34.2 | 4.97× | 1.4× |
| Sobol (QMC) | 10.4505 | 0.00009 | 108049.6× | 46.2 | 6.72× | 16089.4× |

Ojo: el error de Sobol baja como 1/N y no como 1/√N, así que su "reducción de varianza" no es un factor fijo: crece con N. Esta cifra solo vale para este N.
