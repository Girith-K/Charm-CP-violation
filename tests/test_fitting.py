# tests for the dm fits, pulls and blinding

import numpy as np
import pytest

from charm_acp import asymmetry as asy
from charm_acp import config
from charm_acp import fitting as fit
from charm_acp.config import DM_FIT_HI, DM_FIT_LO, SEED


TRUE = dict(n_sig=2000.0, n_bkg=2500.0, mu=145.43, sigma=0.4, a=1.0, b=1.5)


def _make_toy(seed=SEED):
    rng = np.random.default_rng(seed)
    edges = np.linspace(DM_FIT_LO, DM_FIT_HI, 73)
    pred = np.diff(fit.model_cdf(edges, **TRUE))
    counts = rng.poisson(pred)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return np.repeat(centers, counts)


def test_bkg_cdf_normalised():
    edges = np.array([DM_FIT_LO, DM_FIT_HI])
    c = fit._bkg_cdf(edges, a=1.0, b=1.5)
    assert np.isclose(c[0], 0.0) and np.isclose(c[1], 1.0)


def test_threshold_density_vanishes_below_mpi():
    assert fit.threshold_density(139.0, 1.0, 1.0) == 0.0
    assert fit.threshold_density(139.57, 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert fit.threshold_density(145.0, 1.0, 1.0) > 0


def test_fit_closure():
    res = fit.fit_dm(_make_toy())
    assert res.valid
    assert abs(res.n_sig - TRUE["n_sig"]) < 3 * res.n_sig_err
    assert abs(res.mu - TRUE["mu"]) < 0.1
    assert res.n_sig_err > 0


def test_pull_study_small():
    out = fit.toy_pull_study(TRUE, n_toys=30)
    assert out["n_failed"] <= 2
    assert abs(out["mean"]) < 0.6
    assert 0.6 < out["width"] < 1.5


def test_fit_dm_binned_matches_fit_dm():
    values = _make_toy()
    edges = np.linspace(DM_FIT_LO, DM_FIT_HI, 73)
    counts, _ = np.histogram(values, bins=72, range=(DM_FIT_LO, DM_FIT_HI))
    a = fit.fit_dm(values)
    b = fit.fit_dm_binned(counts, edges)
    assert a.n_sig == pytest.approx(b.n_sig, rel=1e-9)
    assert a.n_sig_err == pytest.approx(b.n_sig_err, rel=1e-9)
    assert a.sigma == pytest.approx(b.sigma, rel=1e-9)


def _two_category_toy(seed=SEED + 99, dmu=0.0, edges=None, scale=1.0,
                      sratio=1.0):
    rng = np.random.default_rng(seed)
    edges = np.linspace(DM_FIT_LO, DM_FIT_HI, 73) if edges is None else edges
    centers = 0.5 * (edges[1:] + edges[:-1])
    truth = {"D0": dict(TRUE, n_sig=600.0 * scale, n_bkg=900.0 * scale,
                        mu=TRUE["mu"] + 0.5 * dmu,
                        sigma=TRUE["sigma"] * sratio),
             "D0bar": dict(TRUE, n_sig=450.0 * scale, n_bkg=900.0 * scale,
                           mu=TRUE["mu"] - 0.5 * dmu)}
    counts, by_cat = {}, {}
    for cat, t in truth.items():
        counts[cat] = rng.poisson(np.diff(fit.model_cdf(edges, **t)))
        by_cat[cat] = np.repeat(centers, counts[cat])
    return truth, by_cat, counts, edges


def test_fit_categories_shared_shape_closure():
    truth, by_cat, _, _ = _two_category_toy()

    res, combined = fit.fit_categories(by_cat, share_shape=True)
    assert combined.valid
    for cat in ("D0", "D0bar"):
        r = res[cat]
        assert r.valid
        assert abs(r.n_sig - truth[cat]["n_sig"]) < 3 * r.n_sig_err
        for p in ("sigma", "a", "b"):
            assert getattr(r, p) == pytest.approx(getattr(combined, p))


def test_fit_categories_frees_the_per_flavour_mean():
    _truth, by_cat, _, _ = _two_category_toy(dmu=0.06, scale=25.0)

    free, combined = fit.fit_categories(by_cat, share_shape=True)
    assert free["D0"].mu != free["D0bar"].mu
    assert free["D0"].mu - free["D0bar"].mu == pytest.approx(0.06, abs=0.015)

    frozen, comb2 = fit.fit_categories(by_cat, share_shape=True,
                                       share_mean=True)
    for cat in ("D0", "D0bar"):
        assert frozen[cat].mu == pytest.approx(comb2.mu)


def test_fit_categories_recovers_asymmetry_with_offset_means():
    truth, by_cat, _, _ = _two_category_toy(dmu=0.08, scale=25.0)
    a_true = ((truth["D0"]["n_sig"] - truth["D0bar"]["n_sig"])
              / (truth["D0"]["n_sig"] + truth["D0bar"]["n_sig"]))

    def a_of(**kw):
        res, _ = fit.fit_categories(by_cat, share_shape=True, **kw)
        return asy.raw_asymmetry(res["D0"].n_sig, res["D0bar"].n_sig,
                                 res["D0"].n_sig_err, res["D0bar"].n_sig_err)

    a_free, s_free = a_of()
    a_frozen, _ = a_of(share_mean=True)
    assert abs(a_free - a_true) < 3 * s_free
    assert abs(a_free - a_true) < abs(a_frozen - a_true)


def test_fit_categories_shares_the_width_by_default():
    _truth, by_cat, _, _ = _two_category_toy(sratio=1.05, scale=25.0)

    shared, combined = fit.fit_categories(by_cat, share_shape=True)
    for cat in ("D0", "D0bar"):
        assert shared[cat].sigma == pytest.approx(combined.sigma)

    free, _ = fit.fit_categories(by_cat, share_shape=True, share_width=False)
    assert free["D0"].sigma != free["D0bar"].sigma
    assert free["D0"].sigma > free["D0bar"].sigma
    for p in ("ratio", "frac", "a", "b"):
        for cat in ("D0", "D0bar"):
            assert getattr(free[cat], p) == getattr(shared[cat], p)


def test_asymmetry_depends_on_the_width_sharing_choice():
    def mean_shift(sratio, seeds=(1, 2, 3, 4)):
        out = []
        for k in seeds:
            _t, by_cat, _, _ = _two_category_toy(seed=SEED + 400 + k,
                                                 sratio=sratio, scale=25.0)

            def a_of(**kw):
                res, _ = fit.fit_categories(by_cat, share_shape=True, **kw)
                return asy.raw_asymmetry(res["D0"].n_sig, res["D0bar"].n_sig,
                                         res["D0"].n_sig_err,
                                         res["D0bar"].n_sig_err)[0]
            out.append(abs(a_of(share_width=False) - a_of()))
        return float(np.mean(out))

    assert mean_shift(1.25) > 2 * mean_shift(1.0)


def test_fit_categories_binned_matches_unbinned():
    _truth, by_cat, counts, edges = _two_category_toy(dmu=0.05)

    unb, comb_u = fit.fit_categories(by_cat, share_shape=True)
    bin_, comb_b = fit.fit_categories_binned(counts, edges)
    assert bin_ is not None
    assert comb_b.n_sig == pytest.approx(comb_u.n_sig, rel=1e-6)
    for cat in ("D0", "D0bar"):
        assert bin_[cat].n_sig == pytest.approx(unb[cat].n_sig, rel=1e-6)
        assert bin_[cat].mu == pytest.approx(unb[cat].mu, rel=1e-6)


def test_fit_categories_shape_shift_moves_the_frozen_width():
    _truth, by_cat, _, _ = _two_category_toy()
    base, combined = fit.fit_categories(by_cat, share_shape=True)
    shifted, _ = fit.fit_categories(by_cat, share_shape=True,
                                    shape_shift={"sigma": 0.05})
    assert shifted["D0"].sigma == pytest.approx(combined.sigma + 0.05)
    assert shifted["D0"].n_sig != base["D0"].n_sig


def test_blind_offset_explicit_passphrase():
    kw = dict(passphrase="test-secret", scale=0.25)
    off = asy.blind_offset(**kw)
    assert -0.25 <= off <= 0.25
    assert off == asy.blind_offset(**kw)
    assert off != asy.blind_offset(passphrase="other-secret", scale=0.25)


def test_blind_offset_salts_distinct():
    kw = dict(passphrase="test-secret", scale=0.25)
    offs = {asy.blind_offset(**kw), asy.blind_offset(salt="v2", **kw),
            asy.blind_offset(salt="v3", **kw),
            asy.blind_offset(salt="v3|KK", **kw),
            asy.blind_offset(salt="v3|PiPi", **kw)}
    assert len(offs) == 5


@pytest.mark.skipif(not (config.DATA_DIR / ".blind_passphrase").exists(),
                    reason="analysis passphrase not present on this machine")
def test_blinding_roundtrip():
    d, s = 0.0012, 0.0034
    bd, bs = asy.blind_delta_acp(d, s)
    assert bs == s
    assert bd != d
    assert asy.unblind(bd) == pytest.approx(d, abs=1e-15)
    bk, _ = asy.blind_raw(d, s, "KK")
    bp, _ = asy.blind_raw(d, s, "PiPi")
    assert bk != bp
    assert asy.unblind(bk, mode="KK") == pytest.approx(d, abs=1e-15)


def test_bkg_cdf_windowed_endpoints():
    e = np.linspace(141.0, 155.0, 15)
    c = fit._bkg_cdf(e, a=1.0, b=1.5, lo=141.0, hi=155.0)
    assert np.isclose(c[0], 0.0) and np.isclose(c[-1], 1.0)


def test_fit_window_forwarded():
    values = _make_toy()
    lo, hi = 141.0, 154.0
    res = fit.fit_dm(values, lo=lo, hi=hi, nbins=52)
    inwin = int(((values > lo) & (values < hi)).sum())
    assert res.valid
    assert res.n_sig + res.n_bkg == pytest.approx(inwin, rel=0.02)
    assert res.pred.sum() == pytest.approx(res.n_sig + res.n_bkg, rel=1e-6)


def test_dgauss_defaults_match_gauss():
    edges = np.linspace(DM_FIT_LO, DM_FIT_HI, 73)
    a = fit.model_cdf(edges, **TRUE)
    b = fit.model_cdf_dg(edges, TRUE["n_sig"], TRUE["n_bkg"], TRUE["mu"],
                         TRUE["sigma"], 1.0, 1.0, TRUE["a"], TRUE["b"])
    assert np.allclose(a, b)


def test_fit_dgauss_closure():
    rng = np.random.default_rng(SEED + 7)
    edges = np.linspace(DM_FIT_LO, DM_FIT_HI, 73)
    truth = dict(n_sig=4000.0, n_bkg=3000.0, mu=145.43, sigma=0.35,
                 ratio=2.5, frac=0.75, a=1.0, b=1.5)
    cdf = (truth["n_sig"] * fit._sig_cdf_dgauss(edges, truth["mu"],
                                                truth["sigma"], truth["ratio"],
                                                truth["frac"])
           + truth["n_bkg"] * fit._bkg_cdf(edges, truth["a"], truth["b"]))
    counts = rng.poisson(np.diff(cdf))
    centers = 0.5 * (edges[1:] + edges[:-1])
    res = fit.fit_dm(np.repeat(centers, counts), signal="dgauss")
    assert res.valid
    assert abs(res.n_sig - truth["n_sig"]) < 4 * res.n_sig_err


def test_bkg_cdf_matches_numerical_integral():
    lo, hi, a, b = 140.0, 158.0, 4.0, -1.5
    edges = np.linspace(lo, hi, 25)
    c = fit._bkg_cdf(edges, a=a, b=b, lo=lo, hi=hi)
    fine = np.linspace(lo, hi, 200001)
    dens = fit.threshold_density(fine, a, b)
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (dens[1:] + dens[:-1]) * np.diff(fine))])
    ref = np.interp(edges, fine, cum / cum[-1])
    assert np.max(np.abs(c - ref)) < 1e-5


def test_bkg_cdf_quadrature_and_grid_branches_agree():
    lo, hi, a, b = 140.0, 158.0, 3.0, -2.0
    edges = np.linspace(lo, hi, 19)
    exact = fit._bkg_cdf(edges, a=a, b=b, lo=lo, hi=hi)
    fallback = fit._bkg_cdf(edges, a=a, b=b, lo=lo - 1e-9, hi=hi)
    assert np.max(np.abs(exact - fallback)) < 1e-4


def test_bkg_cdf_is_monotonic_and_shape_sensitive():
    edges = np.linspace(140.0, 158.0, 30)
    c1 = fit._bkg_cdf(edges, a=2.0, b=-1.0)
    c2 = fit._bkg_cdf(edges, a=8.0, b=-3.0)
    assert np.all(np.diff(c1) >= -1e-12)
    assert np.max(np.abs(c1 - c2)) > 1e-3


def test_bkg_cdf_powexp_matches_numerical_integral():
    lo, hi, a, b = 140.0, 158.0, 1.3, -0.4
    edges = np.linspace(lo, hi, 25)
    c = fit._bkg_cdf_powexp(edges, a=a, b=b, lo=lo, hi=hi)
    fine = np.linspace(lo, hi, 200001)
    z = np.clip(fine - fit.M_PI, 0.0, None)
    dens = np.where(z > 0, np.power(z, a) * np.exp(b * z), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (dens[1:] + dens[:-1]) * np.diff(fine))])
    ref = np.interp(edges, fine, cum / cum[-1])
    assert c[0] == 0.0 and c[-1] == 1.0
    assert np.all(np.diff(c) >= -1e-12)
    assert np.max(np.abs(c - ref)) < 1e-5


def test_bkg_powexp_differs_from_threshold():
    edges = np.linspace(140.0, 158.0, 30)
    c1 = fit._bkg_cdf(edges, a=1.0, b=1.0)
    c2 = fit._bkg_cdf_powexp(edges, a=1.0, b=1.0)
    assert np.max(np.abs(c1 - c2)) > 1e-3


def test_fit_dm_powexp_background_recovers_yield():
    rng = np.random.default_rng(7)
    sig = rng.normal(145.43, 0.5, 3000)
    bkg = 140.0 + 18.0 * rng.power(2.0, 9000)
    dm = np.concatenate([sig, bkg])
    dm = dm[(dm > 140.0) & (dm < 158.0)]
    r = fit.fit_dm(dm, bkg="powexp")
    assert r.valid and r.bkg == "powexp"
    assert abs(r.n_sig - 3000) < 5 * r.n_sig_err
    assert np.isclose(r.pred.sum(), r.n_sig + r.n_bkg, rtol=1e-6)


def test_fit_dm_default_bkg_unchanged():
    rng = np.random.default_rng(11)
    dm = np.concatenate([rng.normal(145.43, 0.5, 2000),
                         140.0 + 18.0 * rng.power(2.0, 6000)])
    dm = dm[(dm > 140.0) & (dm < 158.0)]
    r_default = fit.fit_dm(dm)
    r_named = fit.fit_dm(dm, bkg="threshold")
    assert r_default.bkg == "threshold"
    assert r_default.n_sig == r_named.n_sig
    assert r_default.n_sig_err == r_named.n_sig_err


def test_toy_pull_study_dgauss_unbiased():
    true_pars = dict(n_sig=1500.0, n_bkg=1500.0, mu=145.43, sigma=0.35,
                     ratio=2.5, frac=0.8, a=1.0, b=1.0)
    out = fit.toy_pull_study(true_pars, n_toys=40, signal="dgauss")
    assert out["n_failed"] <= 4
    assert abs(out["mean"]) < 4 * out["mean_err"]
    assert 0.6 < out["width"] < 1.4
