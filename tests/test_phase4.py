import numpy as np
import pytest

from mcpricer import (binomial_put, bs_price, down_and_out_call_price, geometric_asian_call_price,
                      mc_american_put_lsm, mc_asian_call, mc_down_and_out_call, simulate_paths)

P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
LS = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2)  # the classic Longstaff–Schwartz example


def test_paths_have_the_right_mean():
    # E[S_t] = S0 e^{rt} at every date
    paths = simulate_paths(**{k: P[k] for k in ("S0", "T", "r", "sigma")}, n=200_000, m=4,
                           rng=np.random.default_rng(0))
    expected = P["S0"] * np.exp(P["r"] * np.arange(1, 5) / 4)
    assert paths.mean(axis=0) == pytest.approx(expected, rel=2e-3)


# ---------- Asian ----------

def test_geometric_formula_matches_simulation():
    paths = simulate_paths(100.0, 1.0, 0.05, 0.2, 200_000, 12, np.random.default_rng(1))
    Y = np.exp(-0.05) * np.maximum(np.exp(np.log(paths).mean(axis=1)) - 100.0, 0)
    se = Y.std(ddof=1) / np.sqrt(len(Y))
    assert abs(Y.mean() - geometric_asian_call_price(**P, m=12)) < 3 * se


def test_geometric_asian_with_one_fixing_is_european():
    assert geometric_asian_call_price(**P, m=1) == pytest.approx(bs_price(**P))


def test_control_variate_agrees_with_plain_and_is_far_more_precise():
    plain, se_plain = mc_asian_call(**P, n=100_000, m=12, seed=2, control=False)
    cv, se_cv = mc_asian_call(**P, n=100_000, m=12, seed=3)
    assert abs(plain - cv) < 3 * np.hypot(se_plain, se_cv)
    assert se_cv < se_plain / 10


def test_asian_price_ordering():
    # geometric average <= arithmetic average (AM-GM), and averaging lowers volatility
    arithmetic, _ = mc_asian_call(**P, n=100_000, m=12, seed=4)
    assert geometric_asian_call_price(**P, m=12) < arithmetic < bs_price(**P)


# ---------- Barrier ----------

def test_far_away_barrier_is_a_vanilla_call():
    assert down_and_out_call_price(100.0, 100.0, 1e-6, 1.0, 0.05, 0.2) == pytest.approx(bs_price(**P))


def test_brownian_bridge_removes_the_monitoring_bias():
    exact = down_and_out_call_price(100.0, 100.0, 90.0, 1.0, 0.05, 0.2)
    discrete, se_d = mc_down_and_out_call(100.0, 100.0, 90.0, 1.0, 0.05, 0.2, n=100_000, m=12, seed=5)
    bridge, se_b = mc_down_and_out_call(100.0, 100.0, 90.0, 1.0, 0.05, 0.2, n=100_000, m=12, seed=5, bridge=True)
    assert discrete - exact > 5 * se_d  # checking only 12 dates misses crossings -> too expensive
    assert abs(bridge - exact) < 3 * se_b


# ---------- American put ----------

def test_binomial_european_matches_black_scholes():
    assert binomial_put(**LS, steps=2000, american=False) == pytest.approx(bs_price(**LS, kind="put"), abs=1e-3)


def test_binomial_american_reference_value():
    # The accepted value for this example; early exercise is worth ~0.64 over the European
    assert binomial_put(**LS, steps=2000) == pytest.approx(4.4867, abs=1e-3)


def test_lsm_matches_bermudan_tree():
    # 50 exercise dates in both: tree with 5000 steps, exercise every 100
    tree = binomial_put(**LS, steps=5000, exercise_every=100)
    price, se = mc_american_put_lsm(**LS, n=100_000, m=50, seed=6)
    assert abs(price - tree) < 3 * se + 0.005  # + small allowance for LSM's low bias
    assert price > bs_price(**LS, kind="put")
