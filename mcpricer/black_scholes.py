"""Closed-form Black–Scholes prices for European options.

This is our ground truth: every Monte Carlo result is compared against it.
"""
import numpy as np
from scipy.stats import norm


def _d1(S0, K, T, r, sigma):
    return (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_price(S0, K, T, r, sigma, kind="call"):
    """Black–Scholes price of a European call or put.

    S0: spot price, K: strike, T: maturity in years,
    r: continuously compounded risk-free rate, sigma: annual volatility.
    """
    d1 = _d1(S0, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")


def bs_delta(S0, K, T, r, sigma, kind="call"):
    """dV/dS0: how much the option price moves per unit move in the spot."""
    d1 = _d1(S0, K, T, r, sigma)
    if kind == "call":
        return norm.cdf(d1)
    if kind == "put":
        return norm.cdf(d1) - 1.0
    raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")


def bs_vega(S0, K, T, r, sigma, kind="call"):
    """dV/dsigma: same for call and put (put-call parity has no sigma in it)."""
    return S0 * norm.pdf(_d1(S0, K, T, r, sigma)) * np.sqrt(T)
