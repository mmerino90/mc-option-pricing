"""Fórmula cerrada de Black–Scholes para opciones europeas.

Es nuestra "verdad": todo lo que calculemos con Monte Carlo lo compararemos
contra estos números.
"""
import numpy as np
from scipy.stats import norm


def bs_price(S0, K, T, r, sigma, kind="call"):
    """Precio Black–Scholes de una call o put europea.

    S0: precio actual del activo, K: strike, T: vencimiento en años,
    r: tipo de interés libre de riesgo (continuo), sigma: volatilidad anual.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError(f"kind debe ser 'call' o 'put', no {kind!r}")
