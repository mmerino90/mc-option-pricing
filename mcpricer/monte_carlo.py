"""Valoración Monte Carlo de opciones europeas bajo movimiento browniano geométrico."""
import numpy as np


def terminal_price(S0, T, r, sigma, Z):
    """Precio del activo en T a partir de normales estándar Z ya generadas.

    Solución exacta del GBM dS = r S dt + sigma S dW:
        S_T = S0 * exp((r - sigma²/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)
    Como es exacta, no hace falta dividir [0, T] en pasos para una europea.
    Recibe Z (en vez de generarlas) para poder usar Z y -Z, o normales de Sobol.
    """
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_terminal(S0, T, r, sigma, n, rng):
    """Simula n precios del activo en T bajo la medida neutral al riesgo."""
    # ponytail: solo el valor final; las trayectorias completas llegan con las exóticas (fase 4)
    return terminal_price(S0, T, r, sigma, rng.standard_normal(n))


def payoff(ST, K, kind="call"):
    if kind == "call":
        return np.maximum(ST - K, 0.0)
    if kind == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError(f"kind debe ser 'call' o 'put', no {kind!r}")


def mc_price(S0, K, T, r, sigma, n, kind="call", seed=None):
    """Precio Monte Carlo y su error estándar.

    precio = e^{-rT} * media(payoff), error estándar = desv_típica / sqrt(n).
    Intervalo de confianza al 95%: precio ± 1.96 * error estándar.
    """
    rng = np.random.default_rng(seed)
    ST = simulate_terminal(S0, T, r, sigma, n, rng)
    discounted = np.exp(-r * T) * payoff(ST, K, kind)
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(n)
