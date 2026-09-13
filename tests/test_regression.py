# regression test, the fit on a fixed fake dataset has to give the same numbers every time

import numpy as np
import pytest

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit

EDGES = np.linspace(140.0, 158.0, 73)
LO, HI = 140.0, 158.0
SEED = 20260716

SHAPE = dict(sigma=0.5819, ratio=2.10, frac=0.72, a=1.15, b=1.60)
A_INJECTED = -0.01364
N_SIGNAL = 300_000
N_BKG = 90_000


def _counts():
    rng = np.random.default_rng(SEED)
    out = {}
    for cat, sign, mu in (("D0", +1, 145.4300), ("D0bar", -1, 145.4295)):
        n_sig = N_SIGNAL * (1 + sign * A_INJECTED) / 2
        cdf = (n_sig * fit._sig_cdf_dgauss(EDGES, mu, SHAPE["sigma"],
                                           SHAPE["ratio"], SHAPE["frac"],
                                           LO, HI)
               + N_BKG * fit._bkg_cdf(EDGES, SHAPE["a"], SHAPE["b"], LO, HI))
        out[cat] = rng.poisson(np.diff(cdf))
    return out


def test_dataset_is_deterministic():
    a, b = _counts(), _counts()
    for cat in a:
        assert np.array_equal(a[cat], b[cat])
    assert int(sum(c.sum() for c in a.values())) == 481_384


NOMINAL_A = -0.014282833
NOMINAL_SIGMA = 0.0019834898


def test_nominal_estimator_reproduces_the_pinned_asymmetry():
    res, comb = fit.fit_categories_binned(_counts(), EDGES, signal="dgauss")
    assert comb.valid
    A, sA = asy.raw_asymmetry(res["D0"].n_sig, res["D0bar"].n_sig,
                              res["D0"].n_sig_err, res["D0bar"].n_sig_err)
    assert A == pytest.approx(NOMINAL_A, abs=1e-7), (
        f"the nominal estimator moved: {A:.9f} vs pinned {NOMINAL_A:.9f}")
    assert sA == pytest.approx(NOMINAL_SIGMA, rel=1e-2)


def test_nominal_estimator_recovers_the_injected_asymmetry():
    res, _comb = fit.fit_categories_binned(_counts(), EDGES, signal="dgauss")
    A, sA = asy.raw_asymmetry(res["D0"].n_sig, res["D0bar"].n_sig,
                              res["D0"].n_sig_err, res["D0bar"].n_sig_err)
    assert abs(A - A_INJECTED) < 3 * sA


def test_fit_quality_of_the_pinned_dataset():
    _res, comb = fit.fit_categories_binned(_counts(), EDGES, signal="dgauss")
    assert comb.ndf == 64
    assert 0.5 < comb.chi2_ndf < 1.8


def test_joint_estimator_agrees_with_the_nominal_one():
    got = fit.joint_asymmetry(_counts(), EDGES, signal="dgauss")
    assert got is not None
    A, sA, _res, joint = got
    assert joint.valid
    assert abs(A - NOMINAL_A) < 0.5 * sA
    assert abs(A - A_INJECTED) < 3 * sA


def test_blinding_offset_is_reproducible_from_an_explicit_passphrase():
    off1 = asy.blind_offset(salt="v3|KK", passphrase="regression", scale=0.25)
    off2 = asy.blind_offset(salt="v3|KK", passphrase="regression", scale=0.25)
    other = asy.blind_offset(salt="v3|PiPi", passphrase="regression",
                             scale=0.25)
    assert off1 == off2
    assert off1 != other
    assert abs(off1) <= 0.25
