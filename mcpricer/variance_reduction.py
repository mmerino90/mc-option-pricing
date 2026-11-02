"""Variance reduction techniques for the European option.

All return (price, standard error) like mc_price, and n is always the total
number of simulated prices S_T, so comparisons are fair.
"""
import numpy as np
from scipy.stats import norm, qmc

from .monte_carlo import payoff, terminal_price


def mc_antithetic(S0, K, T, r, sigma, n, kind="call", seed=None):
    """Antithetic variates: every normal Z is also used as -Z.

    Draws n/2 normals and simulates n prices. The two halves of a pair are NOT
    independent, so the standard error is computed on the pair averages
    (Y(Z) + Y(-Z)) / 2, which are independent across pairs.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n // 2)
    disc = np.exp(-r * T)
    y_plus = disc * payoff(terminal_price(S0, T, r, sigma, Z), K, kind)
    y_minus = disc * payoff(terminal_price(S0, T, r, sigma, -Z), K, kind)
    pairs = 0.5 * (y_plus + y_minus)
    return pairs.mean(), pairs.std(ddof=1) / np.sqrt(len(pairs))


def mc_control_variate(S0, K, T, r, sigma, n, kind="call", seed=None):
    """Control variate: the discounted asset itself, X = e^{-rT} S_T.

    We know E[X] = S0 exactly. We estimate
        Y_cv = Y - b (X - S0),   with b = Cov(Y, X) / Var(X)
    which is the slope of the linear regression of Y on X.
    """
    # ponytail: b is estimated on the same sample -> O(1/n) bias, negligible here;
    # a separate pilot sample removes it if it ever matters.
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
    """Quasi-Monte Carlo with randomized (scrambled) Sobol sequences.

    Sobol points are not independent, so sigma/sqrt(n) does not apply.
    Standard trick: repeat with `replicates` independent scramblings of
    n/replicates points each and measure the spread of their means.
    n/replicates must be a power of 2 (that is where Sobol is balanced).
    """
    per = n // replicates
    m = int(np.log2(per))
    if 2**m != per or per * replicates != n:
        raise ValueError("n / replicates must be an exact power of 2")
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    means = []
    for _ in range(replicates):
        u = qmc.Sobol(d=1, scramble=True, seed=rng).random_base2(m)[:, 0]
        Z = norm.ppf(u)  # uniform -> normal by inverting the CDF
        means.append((disc * payoff(terminal_price(S0, T, r, sigma, Z), K, kind)).mean())
    se = np.std(means, ddof=1) / np.sqrt(replicates) if replicates > 1 else np.nan
    return np.mean(means), se
