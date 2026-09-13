# tests for the joint fit of both flavours

import numpy as np
import pytest

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit

EDGES = np.linspace(140.0, 158.0, 73)
LO, HI = 140.0, 158.0
SHAPE = dict(mu=145.43, sigma=0.55, ratio=2.0, frac=0.75, a=1.2, b=1.5)


def _pred(n_sig, n_bkg, mu=None):
    cdf = (n_sig * fit._sig_cdf_dgauss(EDGES, mu or SHAPE["mu"],
                                       SHAPE["sigma"], SHAPE["ratio"],
                                       SHAPE["frac"], LO, HI)
           + n_bkg * fit._bkg_cdf(EDGES, SHAPE["a"], SHAPE["b"], LO, HI))
    return np.diff(cdf)


def _two_flavour(a_true, n_tot=400_000, n_bkg=120_000, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "D0": rng.poisson(_pred(n_tot * (1 + a_true) / 2, n_bkg, 145.430)),
        "D0bar": rng.poisson(_pred(n_tot * (1 - a_true) / 2, n_bkg, 145.429)),
    }


def test_joint_fit_recovers_a_known_asymmetry():
    a_true = -0.0136
    out = fit.joint_asymmetry(_two_flavour(a_true), EDGES, signal="dgauss")
    assert out is not None
    A, sA, _res, joint = out
    assert joint.valid
    assert abs(A - a_true) < 4 * sA


def test_joint_fit_reports_a_positive_yield_correlation():
    res, joint = fit.fit_categories_joint(_two_flavour(-0.0136), EDGES,
                                          signal="dgauss")
    rho = (joint.cov_nsig("D0", "D0bar")
           / (res["D0"].n_sig_err * res["D0bar"].n_sig_err))
    assert 0.05 < rho < 0.95


def test_dropping_the_covariance_overstates_sigma_A():
    res, joint = fit.fit_categories_joint(_two_flavour(-0.0136), EDGES,
                                          signal="dgauss")
    kw = (res["D0"].n_sig, res["D0bar"].n_sig,
          res["D0"].n_sig_err, res["D0bar"].n_sig_err)
    _a1, s_with = asy.raw_asymmetry(*kw, cov=joint.cov_nsig("D0", "D0bar"))
    _a2, s_without = asy.raw_asymmetry(*kw, cov=0.0)
    assert s_with < s_without


def test_joint_and_two_step_agree_on_the_central_value():
    counts = _two_flavour(-0.0136)
    res2, comb = fit.fit_categories_binned(counts, EDGES, signal="dgauss")
    a2, _s2 = asy.raw_asymmetry(res2["D0"].n_sig, res2["D0bar"].n_sig,
                                res2["D0"].n_sig_err, res2["D0bar"].n_sig_err)
    aj, sj, _r, _j = fit.joint_asymmetry(counts, EDGES, signal="dgauss")
    assert abs(aj - a2) < 0.5 * sj


def test_joint_fit_free_width_adds_two_parameters():
    counts = _two_flavour(-0.0136)
    _r, shared = fit.fit_categories_joint(counts, EDGES, signal="dgauss",
                                          share_width=True)
    _r2, freed = fit.fit_categories_joint(counts, EDGES, signal="dgauss",
                                          share_width=False)
    assert freed.nfree == shared.nfree + 1
    assert "sigma_D0" in freed.params and "sigma" not in freed.params


def test_joint_fit_shared_mean_removes_a_parameter():
    counts = _two_flavour(-0.0136)
    _r, free_mu = fit.fit_categories_joint(counts, EDGES, signal="dgauss",
                                           share_mean=False)
    _r2, one_mu = fit.fit_categories_joint(counts, EDGES, signal="dgauss",
                                           share_mean=True)
    assert one_mu.nfree == free_mu.nfree - 1
    assert "mu" in one_mu.params and "mu_D0" in free_mu.params


def test_joint_fit_goodness_of_fit_is_sane():
    _res, joint = fit.fit_categories_joint(_two_flavour(-0.0136), EDGES,
                                           signal="dgauss")
    assert joint.ndf > 100
    assert 0.5 < joint.chi2_ndf < 1.6
    assert 0.0 <= joint.pvalue <= 1.0


def test_joint_fit_refuses_more_than_two_categories():
    counts = _two_flavour(-0.0136)
    counts["extra"] = counts["D0"]
    with pytest.raises(ValueError):
        fit.fit_categories_joint(counts, EDGES, signal="dgauss")


def test_raw_asymmetry_covariance_default_is_the_old_behaviour():
    a1, s1 = asy.raw_asymmetry(1000.0, 900.0, 32.0, 30.0)
    a2, s2 = asy.raw_asymmetry(1000.0, 900.0, 32.0, 30.0, cov=0.0)
    assert a1 == a2 and s1 == pytest.approx(s2)
