# Delta m fit model, fit drivers and toy pull studies

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.special import ndtr

from .config import DM_FIT_HI, DM_FIT_LO, DM_NBINS, SEED
from .kinematics import M_PI

_GRID_N = 2001

_GRID_CACHE = {}

_GL_N = 5
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_N)
_QUAD_CACHE = {}


def _grid(lo, hi):
    g = _GRID_CACHE.get((lo, hi))
    if g is None:
        x = np.linspace(lo, hi, _GRID_N)
        above = x > M_PI
        g = (x, 0.5 * np.diff(x),
             np.maximum(x - M_PI, 0.0),
             np.maximum(x, M_PI) / M_PI,
             above, bool(above.all()))
        _GRID_CACHE[(lo, hi)] = g
    return g


def _quad(e):
    key = (e.size, float(e[0]), float(e[-1]))
    hit = _QUAD_CACHE.get(key)
    if hit is not None and np.array_equal(hit[0], e):
        return hit[1:]
    half = 0.5 * (e[1:] - e[:-1])
    x = 0.5 * (e[1:] + e[:-1])[:, None] + half[:, None] * _GL_X
    above = x > M_PI
    packed = (half, np.maximum(x - M_PI, 0.0), np.maximum(x, M_PI) / M_PI,
              above, bool(above.all()))
    _QUAD_CACHE[key] = (e.copy(),) + packed
    return packed


def threshold_density(dm, a, b, m_pi=M_PI):
    dm = np.asarray(dm, dtype=float)
    x = np.clip(dm - m_pi, 0.0, None)
    out = (1.0 - np.exp(-x / a)) * np.power(np.maximum(dm, m_pi) / m_pi, b)
    return np.where(dm > m_pi, out, 0.0)


def _bkg_cdf(edges, a, b, lo=DM_FIT_LO, hi=DM_FIT_HI):
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or e.size < 2 or e[0] != lo or e[-1] != hi:
        grid, half_dx, shifted, ratio, above, all_above = _grid(lo, hi)
        dens = (1.0 - np.exp(-shifted / a)) * np.power(ratio, b)
        if not all_above:
            dens = np.where(above, dens, 0.0)
        cum = np.empty(_GRID_N)
        cum[0] = 0.0
        np.cumsum((dens[1:] + dens[:-1]) * half_dx, out=cum[1:])
        return np.interp(e, grid, cum / cum[-1])

    half, shifted, ratio, above, all_above = _quad(e)
    dens = (1.0 - np.exp(-shifted / a)) * np.power(ratio, b)
    if not all_above:
        dens = dens * above
    cum = np.empty(e.size)
    cum[0] = 0.0
    np.cumsum(half * (dens @ _GL_W), out=cum[1:])
    return cum / cum[-1]


def _bkg_cdf_powexp(edges, a, b, lo=DM_FIT_LO, hi=DM_FIT_HI):
    # alternative background family x^a exp(b x), x = dm - m_pi, used as a
    # fit-model systematic (cf. the official LHCb open-data omegac example,
    # see LIBRARY_SURVEY.md); same normalisation contract as _bkg_cdf
    e = np.asarray(edges, dtype=float)

    def dens(x):
        z = np.clip(x - M_PI, 0.0, None)
        return np.where(z > 0, np.power(z, a) * np.exp(b * z), 0.0)

    if e.ndim != 1 or e.size < 2 or e[0] != lo or e[-1] != hi:
        grid = np.linspace(lo, hi, _GRID_N)
        d = dens(grid)
        cum = np.empty(_GRID_N)
        cum[0] = 0.0
        np.cumsum((d[1:] + d[:-1]) * 0.5 * np.diff(grid), out=cum[1:])
        return np.interp(e, grid, cum / cum[-1])

    half = 0.5 * (e[1:] - e[:-1])
    x = 0.5 * (e[1:] + e[:-1])[:, None] + half[:, None] * _GL_X
    cum = np.empty(e.size)
    cum[0] = 0.0
    np.cumsum(half * (dens(x) @ _GL_W), out=cum[1:])
    return cum / cum[-1]


_BKG_CDFS = {"threshold": _bkg_cdf, "powexp": _bkg_cdf_powexp}


def _sig_cdf_gauss(edges, mu, sigma, lo=DM_FIT_LO, hi=DM_FIT_HI):
    inv = 1.0 / sigma
    c = ndtr((edges - mu) * inv)
    c_lo = ndtr((lo - mu) * inv)
    c_hi = ndtr((hi - mu) * inv)
    return (c - c_lo) / (c_hi - c_lo)


def _sig_cdf_dgauss(edges, mu, sigma, ratio, frac, lo=DM_FIT_LO, hi=DM_FIT_HI):
    c1 = _sig_cdf_gauss(edges, mu, sigma, lo, hi)
    c2 = _sig_cdf_gauss(edges, mu, sigma * ratio, lo, hi)
    return frac * c1 + (1.0 - frac) * c2


def model_cdf(edges, n_sig, n_bkg, mu, sigma, a, b):
    return n_sig * _sig_cdf_gauss(edges, mu, sigma) + n_bkg * _bkg_cdf(edges, a, b)


def model_cdf_dg(edges, n_sig, n_bkg, mu, sigma, ratio, frac, a, b):
    return (n_sig * _sig_cdf_dgauss(edges, mu, sigma, ratio, frac)
            + n_bkg * _bkg_cdf(edges, a, b))


@dataclass
class DmFitResult:
    n_sig: float
    n_sig_err: float
    n_bkg: float
    n_bkg_err: float
    mu: float
    sigma: float
    a: float
    b: float
    valid: bool
    counts: np.ndarray
    edges: np.ndarray
    ratio: float = 1.0
    frac: float = 1.0
    bkg: str = "threshold"

    @property
    def pred(self):
        lo, hi = float(self.edges[0]), float(self.edges[-1])
        bkg_cdf = _BKG_CDFS[self.bkg]
        cdf = (self.n_sig * _sig_cdf_dgauss(self.edges, self.mu, self.sigma,
                                            self.ratio, self.frac, lo, hi)
               + self.n_bkg * bkg_cdf(self.edges, self.a, self.b, lo, hi))
        return np.diff(cdf)

    @property
    def pulls(self):
        p = self.pred
        return (self.counts - p) / np.sqrt(np.where(p > 0, p, 1.0))


def fit_dm(dm_values, lo=DM_FIT_LO, hi=DM_FIT_HI, nbins=DM_NBINS,
           start=None, fixed_shape=None, signal="gauss", bkg="threshold"):
    dm_values = np.asarray(dm_values, dtype=float)
    counts, edges = np.histogram(dm_values, bins=nbins, range=(lo, hi))
    return fit_dm_binned(counts, edges, start=start, fixed_shape=fixed_shape,
                         signal=signal, bkg=bkg)


def fit_dm_binned(counts, edges, start=None, fixed_shape=None, signal="gauss",
                  bkg="threshold"):
    counts = np.asarray(counts)
    edges = np.asarray(edges, dtype=float)
    n_tot = counts.sum()
    # CDF normalisation follows the requested window, not the default one
    lo, hi = float(edges[0]), float(edges[-1])
    bkg_cdf = _BKG_CDFS[bkg]

    def _model(xe, n_sig, n_bkg, mu, sigma, a, b):
        return (n_sig * _sig_cdf_gauss(xe, mu, sigma, lo, hi)
                + n_bkg * bkg_cdf(xe, a, b, lo, hi))

    def _model_dg(xe, n_sig, n_bkg, mu, sigma, ratio, frac, a, b):
        return (n_sig * _sig_cdf_dgauss(xe, mu, sigma, ratio, frac, lo, hi)
                + n_bkg * bkg_cdf(xe, a, b, lo, hi))

    s = dict(n_sig=0.5 * n_tot, n_bkg=0.5 * n_tot,
             mu=145.43, sigma=0.4, a=1.0, b=1.0)
    if signal == "dgauss":
        s.update(ratio=2.0, frac=0.8)
        s = {k: s[k] for k in ("n_sig", "n_bkg", "mu", "sigma",
                               "ratio", "frac", "a", "b")}
    if start:
        s.update(start)

    cost = ExtendedBinnedNLL(counts, edges,
                             _model_dg if signal == "dgauss" else _model)
    m = Minuit(cost, **s)
    m.limits["n_sig", "n_bkg"] = (0, None)
    m.limits["mu"] = (144.0, 147.0)
    m.limits["sigma"] = (0.1, 3.0)
    m.limits["a"] = (0.05, 50.0)
    m.limits["b"] = (-10.0, 10.0)
    if signal == "dgauss":
        m.limits["ratio"] = (1.05, 10.0)
        m.limits["frac"] = (0.05, 0.95)
    if fixed_shape:
        for k, v in fixed_shape.items():
            m.values[k] = v
            m.fixed[k] = True
    m.migrad()
    m.hesse()

    return DmFitResult(
        n_sig=m.values["n_sig"], n_sig_err=m.errors["n_sig"],
        n_bkg=m.values["n_bkg"], n_bkg_err=m.errors["n_bkg"],
        mu=m.values["mu"], sigma=m.values["sigma"],
        a=m.values["a"], b=m.values["b"],
        valid=bool(m.valid), counts=counts, edges=edges,
        ratio=m.values["ratio"] if signal == "dgauss" else 1.0,
        frac=m.values["frac"] if signal == "dgauss" else 1.0,
        bkg=bkg,
    )


SHAPE_PARS = ("mu", "sigma", "a", "b")
DG_SHAPE_PARS = ("mu", "sigma", "ratio", "frac", "a", "b")


def significant(r, n_sigma=3.0):
    return bool(r.valid and r.n_sig_err > 0 and r.n_sig >= n_sigma * r.n_sig_err)


def fit_categories(dm_by_cat, share_shape=True, lo=DM_FIT_LO, hi=DM_FIT_HI,
                   nbins=DM_NBINS, signal="gauss", bkg="threshold"):
    kw = dict(lo=lo, hi=hi, nbins=nbins, signal=signal, bkg=bkg)
    if not share_shape:
        return {k: fit_dm(v, **kw) for k, v in dm_by_cat.items()}, None

    combined = fit_dm(np.concatenate([np.asarray(v, float)
                                      for v in dm_by_cat.values()]), **kw)
    pars = DG_SHAPE_PARS if signal == "dgauss" else SHAPE_PARS
    shape = {p: getattr(combined, p) for p in pars}
    results = {}
    for name, values in dm_by_cat.items():
        frac = len(values) / max(sum(len(v) for v in dm_by_cat.values()), 1)
        results[name] = fit_dm(values, start={"n_sig": combined.n_sig * frac,
                                              "n_bkg": combined.n_bkg * frac},
                               fixed_shape=shape, **kw)
    return results, combined


def toy_pull_study(true_pars, n_toys=300, lo=DM_FIT_LO, hi=DM_FIT_HI,
                   nbins=DM_NBINS, seed_offset=0, signal="gauss"):
    rng = np.random.default_rng(SEED + seed_offset)
    edges = np.linspace(lo, hi, nbins + 1)

    def _model(xe, n_sig, n_bkg, mu, sigma, a, b):
        return (n_sig * _sig_cdf_gauss(xe, mu, sigma, lo, hi)
                + n_bkg * _bkg_cdf(xe, a, b, lo, hi))

    def _model_dg(xe, n_sig, n_bkg, mu, sigma, ratio, frac, a, b):
        return (n_sig * _sig_cdf_dgauss(xe, mu, sigma, ratio, frac, lo, hi)
                + n_bkg * _bkg_cdf(xe, a, b, lo, hi))

    model = _model_dg if signal == "dgauss" else _model
    pred = np.diff(model(edges, **true_pars))

    pulls, failed = [], 0
    for _ in range(n_toys):
        toy_counts = rng.poisson(pred)
        cost = ExtendedBinnedNLL(toy_counts, edges, model)
        m = Minuit(cost, **true_pars)
        m.limits["n_sig", "n_bkg"] = (0, None)
        m.limits["mu"] = (144.0, 147.0)
        m.limits["sigma"] = (0.1, 3.0)
        m.limits["a"] = (0.05, 50.0)
        m.limits["b"] = (-10.0, 10.0)
        if signal == "dgauss":
            m.limits["ratio"] = (1.05, 10.0)
            m.limits["frac"] = (0.05, 0.95)
        m.migrad()
        m.hesse()
        if not m.valid or m.errors["n_sig"] == 0:
            failed += 1
            continue
        pulls.append((m.values["n_sig"] - true_pars["n_sig"]) / m.errors["n_sig"])

    pulls = np.asarray(pulls)
    return {
        "pulls": pulls,
        "mean": float(np.mean(pulls)),
        "mean_err": float(np.std(pulls) / np.sqrt(len(pulls))),
        "width": float(np.std(pulls)),
        "width_err": float(np.std(pulls) / np.sqrt(2 * len(pulls))),
        "n_failed": failed,
        "n_toys": n_toys,
    }
