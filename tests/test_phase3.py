import pytest

from mcpricer import bs_delta, bs_price, bs_vega, fd_greeks, pathwise_greeks

P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
N = 200_000
ANALYTIC = {"delta": bs_delta, "vega": bs_vega}


def test_analytic_known_values():
    # d1 = 0.35 for these parameters: N(0.35) = 0.6368, 100 * pdf(0.35) = 37.52
    assert bs_delta(**P, kind="call") == pytest.approx(0.6368, abs=1e-4)
    assert bs_delta(**P, kind="put") == pytest.approx(0.6368 - 1, abs=1e-4)
    assert bs_vega(**P) == pytest.approx(37.524, abs=1e-3)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_analytic_matches_bumped_formula(kind):
    # The Greek formulas must be the derivatives of bs_price
    h = 1e-4
    up, dn = bs_price(**{**P, "S0": 100 + h}, kind=kind), bs_price(**{**P, "S0": 100 - h}, kind=kind)
    assert (up - dn) / (2 * h) == pytest.approx(bs_delta(**P, kind=kind), abs=1e-6)
    up, dn = bs_price(**{**P, "sigma": 0.2 + h}, kind=kind), bs_price(**{**P, "sigma": 0.2 - h}, kind=kind)
    assert (up - dn) / (2 * h) == pytest.approx(bs_vega(**P, kind=kind), abs=1e-4)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("greek", ["delta", "vega"])
def test_pathwise_unbiased(kind, greek):
    est, se = pathwise_greeks(**P, n=N, kind=kind, seed=4)[greek]
    assert abs(est - ANALYTIC[greek](**P, kind=kind)) < 3 * se


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("greek", ["delta", "vega"])
def test_finite_differences_with_small_bump(kind, greek):
    # small h -> the O(h²) bias is negligible next to the statistical error
    est, se = fd_greeks(**P, n=N, kind=kind, seed=4, h_spot=0.1, h_vol=0.001)[greek]
    assert abs(est - ANALYTIC[greek](**P, kind=kind)) < 3 * se


def test_common_random_numbers_shrink_the_error():
    _, se_common = fd_greeks(**P, n=N, seed=4, common=True)["delta"]
    _, se_indep = fd_greeks(**P, n=N, seed=4, common=False)["delta"]
    assert se_indep > 10 * se_common
