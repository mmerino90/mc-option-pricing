"""American put: binomial tree (reference) and Longstaff–Schwartz Monte Carlo."""
import numpy as np

from .exotics import simulate_paths


def binomial_put(S0, K, T, r, sigma, steps, american=True, exercise_every=1, return_boundary=False):
    """Cox–Ross–Rubinstein tree. Our reference price, since there is no closed form.

    Backward induction: at each node the value is the discounted risk-neutral
    average of the two children, or the exercise value K - S if that is larger.
    The exercise boundary at step i is the highest node price where exercising wins.
    exercise_every=k only allows exercise every k steps (a Bermudan option), so the
    tree can match the m exercise dates that Longstaff–Schwartz uses.
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral probability of an up move
    disc = np.exp(-r * dt)
    S = S0 * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)  # maturity nodes, top first
    V = np.maximum(K - S, 0.0)
    boundary = np.full(steps, np.nan)
    for i in range(steps - 1, -1, -1):
        S = S[:-1] * d
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american and i % exercise_every == 0:
            exercise = K - S > V
            V = np.where(exercise, K - S, V)
            if exercise.any():
                boundary[i] = S[exercise].max()
    return (V[0], boundary) if return_boundary else V[0]


def _lsm_pass(paths, K, r, dt, degree, betas=None):
    """One backward pass. Fits the regressions if betas is None, otherwise reuses them."""
    n, m = paths.shape
    step_disc = np.exp(-r * dt)
    cash = np.maximum(K - paths[:, -1], 0.0)  # value of following the policy, seen from each date
    fitted = {}
    boundary = np.full(m, np.nan)
    boundary[-1] = K
    for k in range(m - 2, -1, -1):
        cash *= step_disc  # bring tomorrow's cash flow back to date k
        S = paths[:, k]
        intrinsic = np.maximum(K - S, 0.0)
        itm = intrinsic > 0  # only in-the-money paths matter for the decision
        if itm.sum() <= degree:
            continue
        X = np.vander(S[itm] / K, degree + 1)  # basis 1, x, x², ... with x = S/K (keeps numbers ~1)
        beta = np.linalg.lstsq(X, cash[itm], rcond=None)[0] if betas is None else betas[k]
        fitted[k] = beta
        exercise = np.zeros(n, dtype=bool)
        exercise[itm] = intrinsic[itm] > X @ beta  # exercise if it beats the estimated continuation value
        cash[exercise] = intrinsic[exercise]
        if exercise.any():
            boundary[k] = S[exercise].max()
    return cash * step_disc, fitted, boundary  # last discount: from t_1 back to 0


def mc_american_put_lsm(S0, K, T, r, sigma, n, m, seed=None, degree=3, return_boundary=False):
    """Longstaff–Schwartz with m exercise dates.

    1. Training paths: going backwards in time, regress the discounted future cash
       flow on a polynomial of S_k (in-the-money paths only). The fitted value is
       the estimated "value of waiting", and we exercise when K - S_k beats it.
    2. Pricing paths: apply that exercise rule to fresh, independent paths.
    Fitting and pricing on the same paths would let the rule peek at the future;
    with separate paths the price is a slightly LOW estimate (the rule is
    sub-optimal), which is the honest direction to be wrong in.
    """
    rng = np.random.default_rng(seed)
    dt = T / m
    _, betas, _ = _lsm_pass(simulate_paths(S0, T, r, sigma, n, m, rng), K, r, dt, degree)
    cash, _, boundary = _lsm_pass(simulate_paths(S0, T, r, sigma, n, m, rng), K, r, dt, degree, betas)
    price = max(cash.mean(), K - S0)  # also allowed: exercise right now
    se = cash.std(ddof=1) / np.sqrt(n)
    return (price, se, boundary) if return_boundary else (price, se)
