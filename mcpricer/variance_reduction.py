"""Técnicas de reducción de varianza para la opción europea.

Todas devuelven (precio, error estándar) igual que mc_price, y n es siempre el
número total de precios S_T simulados, para que la comparación sea justa.
"""
import numpy as np
from scipy.stats import norm, qmc

from .monte_carlo import payoff, terminal_price


def mc_antithetic(S0, K, T, r, sigma, n, kind="call", seed=None):
    """Variables antitéticas: cada normal Z se usa también como -Z.

    Se generan n/2 normales y se simulan n precios. Las dos mitades de un par
    NO son independientes, así que el error estándar se calcula sobre la media
    de cada par, (Y(Z) + Y(-Z)) / 2, que sí son independientes entre pares.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n // 2)
    disc = np.exp(-r * T)
    y_plus = disc * payoff(terminal_price(S0, T, r, sigma, Z), K, kind)
    y_minus = disc * payoff(terminal_price(S0, T, r, sigma, -Z), K, kind)
    pairs = 0.5 * (y_plus + y_minus)
    return pairs.mean(), pairs.std(ddof=1) / np.sqrt(len(pairs))


def mc_control_variate(S0, K, T, r, sigma, n, kind="call", seed=None):
    """Variable de control: el propio activo descontado, X = e^{-rT} S_T.

    Sabemos que E[X] = S0 exactamente. Estimamos
        Y_cv = Y - b (X - S0),   con b = Cov(Y, X) / Var(X)
    que es la pendiente de la regresión lineal de Y sobre X.
    """
    # ponytail: b se estima con la misma muestra -> sesgo de orden 1/n, despreciable aquí;
    # una muestra piloto separada lo elimina si hiciera falta.
    rng = np.random.default_rng(seed)
    ST = terminal_price(S0, T, r, sigma, rng.standard_normal(n))
    disc = np.exp(-r * T)
    Y = disc * payoff(ST, K, kind)
    X = disc * ST
    cov = np.cov(Y, X)
    b = cov[0, 1] / cov[1, 1]
    y_cv = Y - b * (X - S0)
    return y_cv.mean(), y_cv.std(ddof=1) / np.sqrt(n)


def mc_sobol(S0, K, T, r, sigma, n, kind="call", seed=None, replicates=16):
    """Quasi-Monte Carlo con secuencias de Sobol aleatorizadas (scrambled).

    Los puntos de Sobol no son independientes, así que la fórmula sigma/sqrt(n)
    no vale. Truco estándar: repetir con `replicates` aleatorizaciones
    independientes de n/replicates puntos cada una y medir la dispersión de
    sus medias. n/replicates debe ser potencia de 2 (Sobol se equilibra así).
    """
    per = n // replicates
    m = int(np.log2(per))
    if 2**m != per or per * replicates != n:
        raise ValueError("n / replicates debe ser una potencia de 2 exacta")
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    means = []
    for _ in range(replicates):
        u = qmc.Sobol(d=1, scramble=True, seed=rng).random_base2(m)[:, 0]
        Z = norm.ppf(u)  # uniforme -> normal invirtiendo la función de distribución
        means.append((disc * payoff(terminal_price(S0, T, r, sigma, Z), K, kind)).mean())
    se = np.std(means, ddof=1) / np.sqrt(replicates) if replicates > 1 else np.nan
    return np.mean(means), se
