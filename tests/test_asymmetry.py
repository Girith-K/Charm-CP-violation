# tests for the asymmetry maths, errors, averages and binning

import numpy as np
import pytest

from charm_acp import asymmetry as asy


def test_raw_asymmetry_matches_binomial_error():
    n, nb = 40000.0, 36000.0
    A, sA = asy.raw_asymmetry(n, nb, np.sqrt(n), np.sqrt(nb))
    N = n + nb
    assert A == pytest.approx((n - nb) / N)
    assert sA == pytest.approx(np.sqrt((1 - A * A) / N), rel=1e-9)


def test_raw_asymmetry_counting():
    A, sA = asy.raw_asymmetry_counting(600.0, 400.0)
    assert A == pytest.approx(0.2)
    assert sA == pytest.approx(np.sqrt((1 - 0.04) / 1000.0))


def test_delta_acp_combination():
    d, s = asy.delta_acp(0.001, 0.002, -0.003, 0.004)
    assert d == pytest.approx(0.004)
    assert s == pytest.approx(np.hypot(0.002, 0.004))


def test_polarity_average_equal_weights():
    a, s = asy.polarity_average(-0.0068, 0.0022, -0.0208, 0.0020)
    assert a == pytest.approx(-0.0138)
    assert s == pytest.approx(0.5 * np.hypot(0.0022, 0.0020))


def test_weighted_average_hand_example():
    avg, err = asy.weighted_average([1.0, 3.0], [1.0, 2.0])
    assert avg == pytest.approx(1.4)
    assert err == pytest.approx(np.sqrt(0.8))


def test_weighted_average_rejects_bad_errors():
    with pytest.raises(ValueError):
        asy.weighted_average([1.0], [0.0])
    with pytest.raises(ValueError):
        asy.weighted_average([1.0, 2.0], [1.0, np.nan])


def test_bin_index_2d_keeps_top_edge():
    xe = np.array([0.0, 1.0, 2.0])
    ye = np.array([0.0, 1.0, 2.0])
    assert asy.bin_index_2d([2.0], [2.0], xe, ye)[0] == 3
    assert asy.bin_index_2d([0.0], [0.5], xe, ye)[0] == 0
    assert asy.bin_index_2d([-0.1], [0.5], xe, ye)[0] == -1
    assert asy.bin_index_2d([2.1], [0.5], xe, ye)[0] == -1


def test_binned_delta_acp_masks_empty_bins():
    n = np.array([100.0, 0.0])
    s = np.array([10.0, 0.0])
    d, e = asy.binned_delta_acp(n, n, s, s, n, n, s, s)
    assert d == pytest.approx(0.0)
    assert np.isfinite(e)
