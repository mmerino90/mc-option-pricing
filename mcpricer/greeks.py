"""Monte Carlo Greeks (delta and vega): pathwise vs finite differences.

Both functions return {"delta": (estimate, std_error), "vega": (estimate, std_error)}.
"""
import numpy as np

from .monte_carlo import payoff, terminal_price


def _mean_se(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def pathwise_greeks(S0, K, T, r, sigma, n, kind="call", seed=None):
    """Pathwise method: differentiate the discounted payoff inside the expectation.

    dV/dθ = e^{-rT} E[ payoff'(S_T) * dS_T/dθ ], with
        payoff'(S_T) = 1{S_T > K} for a call, -1{S_T < K} for a put
        dS_T/dS0     = S_T / S0
        dS_T/dsigma  = S_T (sqrt(T) Z - sigma T)
    Valid because the payoff is continuous (it fails for digitals, or for gamma).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    ST = terminal_price(S0, T, r, sigma, Z)
    if kind == "call":
        dpayoff = (ST > K).astype(float)
    elif kind == "put":
        dpayoff = -(ST < K).astype(float)
    else:
        raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")
    disc_dpayoff = np.exp(-r * T) * dpayoff
    return {
        "delta": _mean_se(disc_dpayoff * ST / S0),
        "vega": _mean_se(disc_dpayoff * ST * (np.sqrt(T) * Z - sigma * T)),
    }


def fd_greeks(S0, K, T, r, sigma, n, kind="call", seed=None, h_spot=1.0, h_vol=0.01, common=True):
    """Central finite differences: (V(θ + h) - V(θ - h)) / 2h, each V priced by Monte Carlo.

    common=True reuses the same normals Z for both bumped prices (common random
    numbers), so the noise cancels in the difference. common=False draws fresh
    normals for each bumped price, to show why that is a bad idea.
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    def diff(S0_up, S0_dn, sig_up, sig_dn, h):
        Z_up = rng.standard_normal(n)
        Z_dn = Z_up if common else rng.standard_normal(n)
        y_up = disc * payoff(terminal_price(S0_up, T, r, sig_up, Z_up), K, kind)
        y_dn = disc * payoff(terminal_price(S0_dn, T, r, sig_dn, Z_dn), K, kind)
        return _mean_se((y_up - y_dn) / (2 * h))

    return {
        "delta": diff(S0 + h_spot, S0 - h_spot, sigma, sigma, h_spot),
        "vega": diff(S0, S0, sigma + h_vol, sigma - h_vol, h_vol),
    }
