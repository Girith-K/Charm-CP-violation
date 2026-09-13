# tests for the triple Gaussian signal shape

import numpy as np
import pytest

from charm_acp import fitting as fit

EDGES = np.linspace(140.0, 158.0, 73)
LO, HI = 140.0, 158.0


def test_tgauss_is_a_normalised_cdf():
    c = fit._sig_cdf_tgauss(EDGES, 145.43, 0.5, 1.8, 0.6, 2.0, 0.7, LO, HI)
    assert c[0] == pytest.approx(0.0, abs=1e-12)
    assert c[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(c) >= -1e-15)


def test_tgauss_reduces_to_dgauss_when_the_third_width_collapses():
    t = fit._sig_cdf_tgauss(EDGES, 145.43, 0.5, 2.0, 0.8, 3.0, 1.0, LO, HI)
    d = fit._sig_cdf_dgauss(EDGES, 145.43, 0.5, 2.0, 0.8, LO, HI)
    assert np.allclose(t, d, atol=1e-12)


def test_tgauss_has_wider_tails_than_dgauss():
    mu, sg = 145.43, 0.5
    t = fit._sig_cdf_tgauss(EDGES, mu, sg, 1.8, 0.6, 2.5, 0.5, LO, HI)
    d = fit._sig_cdf_dgauss(EDGES, mu, sg, 1.8, 0.6, LO, HI)
    core = (EDGES[:-1] > mu - 1.0) & (EDGES[:-1] < mu + 1.0)
    assert np.diff(t)[core].sum() < np.diff(d)[core].sum()


def test_shape_pars_tracks_the_signal_model():
    assert fit.shape_pars("gauss") == ("mu", "sigma", "a", "b")
    assert fit.shape_pars("dgauss") == fit.DG_SHAPE_PARS
    assert fit.shape_pars("tgauss") == fit.TG_SHAPE_PARS
    with pytest.raises(KeyError):
        fit.shape_pars("johnson")


def test_sig_cdf_dispatches_and_rejects_unknown_models():
    pars = {"ratio": 2.0, "frac": 0.8, "ratio2": 2.0, "frac2": 0.7}
    g = fit.sig_cdf("gauss", EDGES, 145.43, 0.5, pars, LO, HI)
    d = fit.sig_cdf("dgauss", EDGES, 145.43, 0.5, pars, LO, HI)
    t = fit.sig_cdf("tgauss", EDGES, 145.43, 0.5, pars, LO, HI)
    assert not np.allclose(g, d) and not np.allclose(d, t)
    with pytest.raises(ValueError):
        fit.sig_cdf("nonsense", EDGES, 145.43, 0.5, pars, LO, HI)


def _toy_tgauss(seed=4, n_sig=200_000, n_bkg=60_000):
    cdf = (n_sig * fit._sig_cdf_tgauss(EDGES, 145.43, 0.45, 1.7, 0.55, 2.2,
                                       0.6, LO, HI)
           + n_bkg * fit._bkg_cdf(EDGES, 1.2, 1.5, LO, HI))
    return np.random.default_rng(seed).poisson(np.diff(cdf))


def test_tgauss_fit_closes_on_tgauss_data():
    counts = _toy_tgauss()
    r = fit.fit_dm_binned(counts, EDGES, signal="tgauss")
    assert r.valid
    assert r.nfree == 10
    assert r.ndf == 62
    assert 0.5 < r.chi2_ndf < 1.8
    assert r.n_sig == pytest.approx(200_000, rel=0.02)


def test_third_gaussian_is_a_significant_improvement():
    from scipy.stats import chi2 as chi2_dist

    counts = _toy_tgauss()
    good = fit.fit_dm_binned(counts, EDGES, signal="tgauss")
    poor = fit.fit_dm_binned(counts, EDGES, signal="dgauss")
    assert good.valid and poor.valid

    delta = poor.fval - good.fval
    assert good.nfree - poor.nfree == 2
    assert delta > 0
    assert chi2_dist.sf(delta, 2) < 1e-6
    assert poor.chi2_ndf > good.chi2_ndf


def test_shared_shape_freezes_the_extra_tgauss_parameters():
    counts = {"D0": _toy_tgauss(seed=5), "D0bar": _toy_tgauss(seed=6)}
    res, comb = fit.fit_categories_binned(counts, EDGES, signal="tgauss")
    assert comb.valid
    for cat in ("D0", "D0bar"):
        assert res[cat].ratio2 == pytest.approx(comb.ratio2)
        assert res[cat].frac2 == pytest.approx(comb.frac2)
        assert res[cat].nfree == 3
