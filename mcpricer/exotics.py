"""Path-dependent options: Asian (average price) and barrier.

Both need the asset at intermediate dates, not just at T, so we simulate paths.
"""
import numpy as np
from scipy.stats import norm

from .black_scholes import bs_price


def simulate_paths(S0, T, r, sigma, n, m, rng):
    """(n, m) array of asset prices at t_k = k T / m, k = 1..m (S0 itself not included).

    Each step uses the exact GBM increment, so there is no discretisation error
    in the prices themselves: log S_{k+1} = log S_k + (r - sigma²/2) dt + sigma sqrt(dt) Z.
    """
    dt = T / m
    Z = rng.standard_normal((n, m))
    return S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))


# ---------- Asian call on the average of the m fixings ----------

def geometric_asian_call_price(S0, K, T, r, sigma, m):
    """Closed form for the call on the GEOMETRIC average G = (S_1 ... S_m)^(1/m).

    log G is an average of normals, so it is normal itself, with
        mean = log S0 + (r - sigma²/2) T (m+1) / (2m)
        var  = sigma² T (m+1)(2m+1) / (6 m²)
    and the price is a Black–Scholes-style formula with those two numbers.
    """
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    v = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m**2)
    d1 = (mu - np.log(K) + v) / np.sqrt(v)
    d2 = d1 - np.sqrt(v)
    return np.exp(-r * T) * (np.exp(mu + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))


def mc_asian_call(S0, K, T, r, sigma, n, m, seed=None, control=True):
    """Call on the ARITHMETIC average of m fixings (no closed form).

    control=True uses the geometric Asian call as control variate: same paths,
    payoffs almost identical (correlation ~0.999), and its exact price is known.
    """
    rng = np.random.default_rng(seed)
    paths = simulate_paths(S0, T, r, sigma, n, m, rng)
    disc = np.exp(-r * T)
    Y = disc * np.maximum(paths.mean(axis=1) - K, 0.0)
    if control:
        X = disc * np.maximum(np.exp(np.log(paths).mean(axis=1)) - K, 0.0)
        cov = np.cov(Y, X)
        Y = Y - cov[0, 1] / cov[1, 1] * (X - geometric_asian_call_price(S0, K, T, r, sigma, m))
    return Y.mean(), Y.std(ddof=1) / np.sqrt(n)


# ---------- Down-and-out call: dies if the asset ever falls to the barrier H < S0 ----------

def down_and_out_call_price(S0, K, H, T, r, sigma):
    """Closed form with CONTINUOUS monitoring, for H <= K (Merton / Reiner–Rubinstein).

    down-and-out = vanilla call - down-and-in call.
    """
    lam = (r + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (S0 * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    down_in = (S0 * (H / S0) ** (2 * lam) * norm.cdf(y)
               - K * np.exp(-r * T) * (H / S0) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))
    return bs_price(S0, K, T, r, sigma) - down_in


def mc_down_and_out_call(S0, K, H, T, r, sigma, n, m, seed=None, bridge=False):
    """Barrier checked at m dates. bridge=True also accounts for crossings between dates.

    Brownian bridge: given log S at both ends of a step and both above log H, the
    probability that the path dipped below H in between is
        p = exp(-2 log(S_k / H) log(S_{k+1} / H) / (sigma² dt))
    so each path is weighted by its probability of surviving every step.
    """
    rng = np.random.default_rng(seed)
    dt = T / m
    S = np.full(n, float(S0))
    alive = np.ones(n)  # survival weight: 0/1 without the bridge, a probability with it
    for _ in range(m):  # step by step: memory stays O(n) even with thousands of dates
        S_next = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n))
        alive *= S_next > H
        if bridge:
            # clip at 0 so dead paths (log < 0) can't overflow the exponential
            a, b = np.maximum(np.log(S / H), 0), np.maximum(np.log(S_next / H), 0)
            alive *= 1 - np.exp(-2 * a * b / (sigma**2 * dt))
        S = S_next
    Y = np.exp(-r * T) * alive * np.maximum(S - K, 0.0)
    return Y.mean(), Y.std(ddof=1) / np.sqrt(n)
