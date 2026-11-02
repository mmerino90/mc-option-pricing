import numpy as np
import pytest

from mcpricer import bs_price, mc_price

P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_bs_known_values():
    # Valores de referencia de libro (Hull) para estos parámetros
    assert bs_price(**P, kind="call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(**P, kind="put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    # C - P = S0 - K e^{-rT}, para cualquier strike
    for K in (80.0, 100.0, 120.0):
        p = {**P, "K": K}
        lhs = bs_price(**p, kind="call") - bs_price(**p, kind="put")
        assert lhs == pytest.approx(P["S0"] - K * np.exp(-P["r"] * P["T"]))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_within_confidence_interval(kind):
    price, se = mc_price(**P, n=200_000, kind=kind, seed=1)
    assert abs(price - bs_price(**P, kind=kind)) < 3 * se


def test_standard_error_scales_as_inverse_sqrt_n():
    _, se_small = mc_price(**P, n=10_000, seed=2)
    _, se_big = mc_price(**P, n=1_000_000, seed=3)
    # 100 veces más simulaciones -> error 10 veces menor
    assert se_small / se_big == pytest.approx(10, rel=0.05)
