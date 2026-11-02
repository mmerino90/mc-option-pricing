import pytest

from mcpricer import bs_price, mc_antithetic, mc_control_variate, mc_price, mc_sobol

P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
N = 2**16
METHODS = [mc_antithetic, mc_control_variate, mc_sobol]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_unbiased_within_confidence_interval(method, kind):
    price, se = method(**P, n=N, kind=kind, seed=11)
    assert abs(price - bs_price(**P, kind=kind)) < 3 * se


@pytest.mark.parametrize("method", METHODS)
def test_reduces_standard_error(method):
    # mismo número de precios simulados -> error estándar menor que Monte Carlo simple
    _, se_plain = mc_price(**P, n=N, seed=5)
    _, se = method(**P, n=N, seed=5)
    assert se < se_plain


def test_control_variate_is_exact_for_the_asset_itself():
    # Si el "payoff" fuera el propio activo (call con K=0), la variable de control
    # lo explica entero: varianza cero y precio exacto S0.
    price, se = mc_control_variate(**{**P, "K": 0.0}, n=1000, seed=1)
    assert price == pytest.approx(P["S0"])
    assert se == pytest.approx(0.0, abs=1e-10)


def test_sobol_requires_power_of_two():
    with pytest.raises(ValueError):
        mc_sobol(**P, n=1000)
