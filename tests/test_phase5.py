import numpy as np
import pytest

from mcpricer import bs_price, mc_asian_call, mc_down_and_out_call

cpp = pytest.importorskip("mcpricer_cpp", reason="build it first: python setup.py build_ext --inplace")

P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
N = 200_000


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_matches_black_scholes(kind):
    price, se = cpp.european(**P, n=N, kind=kind, seed=1)
    assert abs(price - bs_price(**P, kind=kind)) < 3 * se


def test_asian_matches_python_control_variate():
    # The Python control-variate price is ~40x more precise: use it as the reference
    reference, _ = mc_asian_call(**P, n=N, m=12, seed=2)
    price, se = cpp.asian_call(**P, n=N, m=12, seed=2)
    assert abs(price - reference) < 3 * se


def test_barrier_matches_python():
    py_price, py_se = mc_down_and_out_call(**P, H=90.0, n=N, m=52, seed=3)
    price, se = cpp.down_and_out_call(**P, H=90.0, n=N, m=52, seed=3)
    assert abs(price - py_price) < 3 * np.hypot(se, py_se)


def test_same_seed_and_threads_is_reproducible():
    assert cpp.european(**P, n=10_000, seed=7, threads=4) == cpp.european(**P, n=10_000, seed=7, threads=4)


def test_thread_count_does_not_change_the_answer_statistically():
    one, se1 = cpp.european(**P, n=N, seed=8, threads=1)
    many, se2 = cpp.european(**P, n=N, seed=8, threads=8)
    assert abs(one - many) < 3 * np.hypot(se1, se2)


def test_bad_input_raises_value_error():
    with pytest.raises(ValueError):
        cpp.european(**P, n=1000, kind="straddle")
    with pytest.raises(ValueError):
        cpp.european(**P, n=1)
