"""Closed-form Black–Scholes prices for European options.

This is our ground truth: every Monte Carlo result is compared against it.
"""
import numpy as np
from scipy.stats import norm


def bs_price(S0, K, T, r, sigma, kind="call"):
    """Black–Scholes price of a European call or put.

    S0: spot price, K: strike, T: maturity in years,
    r: continuously compounded risk-free rate, sigma: annual volatility.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")
