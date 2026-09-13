# tests for chi2, significance and parameters stuck at a limit

import numpy as np
import pytest

from charm_acp import fitting as fit

EDGES = np.linspace(140.0, 158.0, 73)
LO, HI = 140.0, 158.0
TRUE = dict(mu=145.43, sigma=0.55, ratio=2.0, frac=0.75, a=1.2, b=1.5)


def _pred(n_sig, n_bkg, **over):
    p = {**TRUE, **over}
    cdf = (n_sig * fit._sig_cdf_dgauss(EDGES, p["mu"], p["sigma"],
                                       p["ratio"], p["frac"], LO, HI)
           + n_bkg * fit._bkg_cdf(EDGES, p["a"], p["b"], LO, HI))
    return np.diff(cdf)


def _toy(n_sig, n_bkg, seed=0, **over):
    return np.random.default_rng(seed).poisson(_pred(n_sig, n_bkg, **over))


def test_chi2_at_truth_is_unbiased():
    nu = _pred(60_000, 40_000)
    rng = np.random.default_rng(7)
    vals = []
    for _ in range(200):
        c = rng.poisson(nu).astype(float)
        term = nu - c
        nz = c > 0
        term = term.copy()
        term[nz] += c[nz] * np.log(c[nz] / nu[nz])
        vals.append(2.0 * term.sum())
    assert abs(np.mean(vals) / len(nu) - 1.0) < 0.1


def test_chi2_ndf_close_to_one_for_the_right_model():
    rng = np.random.default_rng(11)
    nu = _pred(60_000, 40_000)
    ratios = []
    for _ in range(12):
        r = fit.fit_dm_binned(rng.poisson(nu), EDGES, signal="dgauss")
        if r.valid:
            ratios.append(r.chi2_ndf)
    assert len(ratios) >= 8
    assert 0.75 < float(np.mean(ratios)) < 1.35


def test_chi2_rejects_a_wrong_signal_model():
    c = _toy(60_000, 40_000, seed=3)
    good = fit.fit_dm_binned(c, EDGES, signal="dgauss")
    bad = fit.fit_dm_binned(c, EDGES, signal="gauss")
    assert bad.chi2_ndf > 2.0 * good.chi2_ndf
    assert bad.pvalue < 1e-6
    assert bad.gof()["max_abs_pull"] > good.gof()["max_abs_pull"]


def test_ndf_accounts_for_fixed_parameters():
    c = _toy(60_000, 40_000, seed=5)
    free = fit.fit_dm_binned(c, EDGES, signal="dgauss")
    frozen = fit.fit_dm_binned(c, EDGES, signal="dgauss",
                               fixed_shape={k: TRUE[k] for k in
                                            ("mu", "sigma", "ratio", "frac",
                                             "a", "b")})
    assert free.nfree == 8
    assert frozen.nfree == 2
    assert frozen.ndf == free.ndf + 6


def test_gof_is_none_when_ndf_is_untracked():
    r = fit.fit_dm_binned(_toy(50_000, 30_000, seed=6), EDGES, signal="dgauss")
    r.nfree = -1
    assert r.ndf is None and r.chi2_ndf is None and r.pvalue is None


def test_fval_equals_the_likelihood_ratio_chi2():
    r = fit.fit_dm_binned(_toy(60_000, 40_000, seed=8), EDGES, signal="dgauss")
    assert r.fval == pytest.approx(r.chi2_lr, rel=1e-6)


def test_lr_significance_is_large_for_a_real_peak():
    r = fit.fit_dm_binned(_toy(60_000, 40_000, seed=9), EDGES, signal="dgauss")
    s = fit.lr_significance(r)
    assert s is not None and s > 100
    assert fit.significant(r) is True


def test_lr_significance_rejects_a_pure_background_sample():
    r = fit.fit_dm_binned(_toy(0, 40_000, seed=10), EDGES, signal="dgauss")
    assert fit.significant(r) is False
    s = fit.lr_significance(r)
    assert s is None or s < 3.0


def test_significant_hesse_mode_still_available():
    r = fit.fit_dm_binned(_toy(60_000, 40_000, seed=12), EDGES, signal="dgauss")
    assert fit.significant(r, method="hesse") is True
    with pytest.raises(ValueError):
        fit.significant(r, method="nonsense")


def test_at_limits_is_empty_for_a_healthy_fit():
    r = fit.fit_dm_binned(_toy(60_000, 40_000, seed=21), EDGES,
                          signal="dgauss")
    assert r.valid
    assert list(r.at_limits) == []
    assert r.gof()["at_limits"] == []


def test_at_limits_detects_both_bounds():
    from iminuit import Minuit

    def cost(x, y, z):
        return (x - 5.0) ** 2 + (y - 5.0) ** 2 + (z - 5.0) ** 2

    cost.errordef = Minuit.LEAST_SQUARES
    m = Minuit(cost, x=1.0, y=1.0, z=1.0)
    m.limits["x"] = (0.0, 1.0)
    m.limits["y"] = (9.0, 20.0)
    m.limits["z"] = (0.0, 20.0)
    m.migrad()

    flagged = set(fit._at_limits(m))
    assert "x@hi" in flagged
    assert "y@lo" in flagged
    assert not any(p.startswith("z") for p in flagged)


def test_at_limits_ignores_fixed_parameters():
    c = _toy(60_000, 40_000, seed=22)
    r = fit.fit_dm_binned(c, EDGES, signal="dgauss",
                          fixed_shape={"ratio": 1.05})
    assert not any(p.startswith("ratio") for p in r.at_limits)


def test_joint_cross_check_refuses_a_boundary_solution():
    counts = {"D0": _toy(60_000, 40_000, seed=23),
              "D0bar": _toy(60_000, 40_000, seed=24)}
    res, joint = fit.fit_categories_joint(counts, EDGES, signal="dgauss")
    got = fit.joint_asymmetry(counts, EDGES, signal="dgauss")
    if joint.at_limits:
        assert got is None
    else:
        assert got is not None
