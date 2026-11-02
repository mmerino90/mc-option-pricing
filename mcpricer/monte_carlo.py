"""Monte Carlo pricing of European options under geometric Brownian motion."""
import numpy as np


def terminal_price(S0, T, r, sigma, Z):
    """Asset price at T from already-drawn standard normals Z.

    Exact solution of the GBM dS = r S dt + sigma S dW:
        S_T = S0 * exp((r - sigma²/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)
    Because it is exact, a European option needs no time steps on [0, T].
    Takes Z (instead of drawing it) so we can reuse Z and -Z, or Sobol normals.
    """
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_terminal(S0, T, r, sigma, n, rng):
    """Simulate n asset prices at T under the risk-neutral measure."""
    # ponytail: terminal value only; full paths arrive with the exotics (phase 4)
    return terminal_price(S0, T, r, sigma, rng.standard_normal(n))


def payoff(ST, K, kind="call"):
    if kind == "call":
        return np.maximum(ST - K, 0.0)
    if kind == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")


def mc_price(S0, K, T, r, sigma, n, kind="call", seed=None):
    """Monte Carlo price and its standard error.

    price = e^{-rT} * mean(payoff), standard error = std / sqrt(n).
    95% confidence interval: price ± 1.96 * standard error.
    """
    rng = np.random.default_rng(seed)
    ST = simulate_terminal(S0, T, r, sigma, n, rng)
    discounted = np.exp(-r * T) * payoff(ST, K, kind)
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(n)
