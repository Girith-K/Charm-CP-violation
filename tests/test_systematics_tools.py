# tests for the Barlow check, the fiducial cut and the random candidate pick

import numpy as np
import pytest

from charm_acp import asymmetry as asy
from charm_acp import selection as sel


def test_barlow_sigma_for_a_nested_subset():
    assert asy.barlow_sigma(0.0006348, 0.0006475) == pytest.approx(
        np.sqrt(0.0006475 ** 2 - 0.0006348 ** 2), rel=1e-9)


def test_barlow_sigma_is_symmetric_in_the_two_errors():
    assert asy.barlow_sigma(0.002, 0.003) == pytest.approx(
        asy.barlow_sigma(0.003, 0.002))


def test_barlow_sigma_is_smaller_than_naive_quadrature():
    s_nom, s_var = 0.0006348, 0.0006475
    assert asy.barlow_sigma(s_nom, s_var) < 0.5 * np.hypot(s_nom, s_var)


def test_barlow_significance_none_when_errors_coincide():
    assert asy.barlow_significance(1e-4, 0.00063, 0.00063) is None


def test_barlow_significance_flags_a_real_shift():
    s = asy.barlow_significance(0.000534, 0.0006348, 0.0006437)
    assert s is not None and s > 3.0


def _soft(px_gev, pz_gev):
    n = len(px_gev)
    return {"D0_MM": np.full(n, 1864.84),
            "piplus_0_PX": np.asarray(px_gev, float) * 1000.0,
            "piplus_0_PZ": np.asarray(pz_gev, float) * 1000.0}


def test_fiducial_defaults_reproduce_the_old_hard_coded_region():
    arr = _soft([0.0, 0.0, 1.5, 1.5, 0.0],
                [3.0, 5.0, 5.0, 8.0, 8.0])
    cuts = sel.Cuts(soft_fiducial=True)
    keep = sel.soft_fiducial_mask(arr, "KPi", cuts)
    assert list(keep) == [False, True, False, True, True]


def test_fiducial_boundary_can_be_moved():
    arr = _soft([0.0, 0.0], [3.0, 5.0])
    loose = sel.soft_fiducial_mask(arr, "KPi",
                                   sel.Cuts(soft_fiducial=True, fid_pz_min=2.0))
    tight = sel.soft_fiducial_mask(arr, "KPi",
                                   sel.Cuts(soft_fiducial=True, fid_pz_min=6.0))
    assert list(loose) == [True, True]
    assert list(tight) == [False, False]


def test_fiducial_stays_symmetric_in_px():
    arr = _soft([+1.5, -1.5], [5.0, 5.0])
    keep = sel.soft_fiducial_mask(arr, "KPi", sel.Cuts(soft_fiducial=True))
    assert keep[0] == keep[1]


def test_random_candidate_rule_ignores_vertex_chi2():
    run = np.zeros(6, dtype=np.int64)
    evt = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    good = np.array([9.0, 0.1, 9.0, 9.0, 0.1, 9.0])
    best = sel.single_candidate_pick(run, evt, good, by="vchi2")
    rand = sel.single_candidate_pick(run, evt, good, by="random")
    assert best.sum() == rand.sum() == 2
    assert list(np.flatnonzero(best)) == [1, 4]
    assert list(np.flatnonzero(rand)) != [1, 4]


def test_random_candidate_rule_is_deterministic():
    run = np.zeros(6, dtype=np.int64)
    evt = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    v = np.arange(6.0)
    a = sel.single_candidate_pick(run, evt, v, by="random")
    b = sel.single_candidate_pick(run, evt, v, by="random")
    assert np.array_equal(a, b)


def test_unknown_candidate_rule_raises():
    with pytest.raises(ValueError):
        sel.single_candidate_pick(np.zeros(2), np.arange(2), np.zeros(2),
                                  by="lowest_dm")
