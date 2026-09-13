# dm fit model, fit functions, joint fit and toys

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.special import ndtr
from scipy.stats import chi2 as _chi2_dist

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


def _sig_cdf_tgauss(edges, mu, sigma, ratio, frac, ratio2, frac2,
                    lo=DM_FIT_LO, hi=DM_FIT_HI):
    c1 = _sig_cdf_gauss(edges, mu, sigma, lo, hi)
    c2 = _sig_cdf_gauss(edges, mu, sigma * ratio, lo, hi)
    c3 = _sig_cdf_gauss(edges, mu, sigma * ratio * ratio2, lo, hi)
    return frac * c1 + (1.0 - frac) * (frac2 * c2 + (1.0 - frac2) * c3)


def sig_cdf(signal, edges, mu, sigma, pars, lo, hi):
    if signal == "gauss":
        return _sig_cdf_gauss(edges, mu, sigma, lo, hi)
    if signal == "dgauss":
        return _sig_cdf_dgauss(edges, mu, sigma, pars["ratio"], pars["frac"],
                               lo, hi)
    if signal == "tgauss":
        return _sig_cdf_tgauss(edges, mu, sigma, pars["ratio"], pars["frac"],
                               pars["ratio2"], pars["frac2"], lo, hi)
    raise ValueError(f"unknown signal model {signal!r}")


SIGNAL_EXTRA = {"gauss": (), "dgauss": ("ratio", "frac"),
                "tgauss": ("ratio", "frac", "ratio2", "frac2")}


def model_cdf(edges, n_sig, n_bkg, mu, sigma, a, b):
    return n_sig * _sig_cdf_gauss(edges, mu, sigma) + n_bkg * _bkg_cdf(edges, a, b)


def model_cdf_dg(edges, n_sig, n_bkg, mu, sigma, ratio, frac, a, b):
    return (n_sig * _sig_cdf_dgauss(edges, mu, sigma, ratio, frac)
            + n_bkg * _bkg_cdf(edges, a, b))


def _at_limits(m, tol=1e-6):
    out = []
    for name in m.parameters:
        if m.fixed[name]:
            continue
        lo, hi = m.limits[name]
        v = m.values[name]
        scale = max(abs(v), 1.0)
        if lo is not None and abs(v - lo) <= tol * scale:
            out.append(f"{name}@lo")
        elif hi is not None and abs(v - hi) <= tol * scale:
            out.append(f"{name}@hi")
    return out


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
    ratio2: float = 1.0
    frac2: float = 1.0
    bkg: str = "threshold"
    signal: str = "gauss"
    nfree: int = -1
    fval: float = float("nan")
    fixed_shape: dict | None = None
    at_limits: tuple = ()

    @property
    def shape_values(self):
        return {"ratio": self.ratio, "frac": self.frac,
                "ratio2": self.ratio2, "frac2": self.frac2}

    @property
    def pred(self):
        lo, hi = float(self.edges[0]), float(self.edges[-1])
        bkg_cdf = _BKG_CDFS[self.bkg]
        sig = self.signal if self.signal in SIGNAL_EXTRA else "dgauss"
        cdf = (self.n_sig * sig_cdf(sig, self.edges, self.mu, self.sigma,
                                    self.shape_values, lo, hi)
               + self.n_bkg * bkg_cdf(self.edges, self.a, self.b, lo, hi))
        return np.diff(cdf)

    @property
    def pulls(self):
        p = self.pred
        return (self.counts - p) / np.sqrt(np.where(p > 0, p, 1.0))


    @property
    def chi2_pearson(self):
        nu = self.pred
        ok = nu > 0
        return float(np.sum((self.counts[ok] - nu[ok]) ** 2 / nu[ok]))

    @property
    def chi2_lr(self):
        nu = np.asarray(self.pred, dtype=float)
        n = np.asarray(self.counts, dtype=float)
        ok = nu > 0
        nu, n = nu[ok], n[ok]
        term = nu - n
        nz = n > 0
        term[nz] += n[nz] * np.log(n[nz] / nu[nz])
        return float(2.0 * np.sum(term))

    @property
    def chi2(self):
        return self.chi2_lr

    @property
    def nbins_used(self):
        return int(np.sum(np.asarray(self.pred) > 0))

    @property
    def ndf(self):
        if self.nfree is None or self.nfree < 0:
            return None
        return int(self.nbins_used - self.nfree)

    @property
    def chi2_ndf(self):
        ndf = self.ndf
        if not ndf or ndf <= 0:
            return None
        return self.chi2 / ndf

    @property
    def pvalue(self):
        ndf = self.ndf
        if not ndf or ndf <= 0:
            return None
        return float(_chi2_dist.sf(self.chi2, ndf))

    def gof(self):
        return {
            "chi2_lr": self.chi2_lr,
            "chi2_pearson": self.chi2_pearson,
            "ndf": self.ndf,
            "chi2_ndf": self.chi2_ndf,
            "p_value": self.pvalue,
            "nbins": self.nbins_used,
            "n_free": None if self.nfree < 0 else self.nfree,
            "max_abs_pull": float(np.max(np.abs(self.pulls))) if
                            len(self.counts) else None,
            "at_limits": list(self.at_limits),
        }


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
    lo, hi = float(edges[0]), float(edges[-1])
    bkg_cdf = _BKG_CDFS[bkg]

    def _model(xe, n_sig, n_bkg, mu, sigma, a, b):
        return (n_sig * _sig_cdf_gauss(xe, mu, sigma, lo, hi)
                + n_bkg * bkg_cdf(xe, a, b, lo, hi))

    def _model_dg(xe, n_sig, n_bkg, mu, sigma, ratio, frac, a, b):
        return (n_sig * _sig_cdf_dgauss(xe, mu, sigma, ratio, frac, lo, hi)
                + n_bkg * bkg_cdf(xe, a, b, lo, hi))

    def _model_tg(xe, n_sig, n_bkg, mu, sigma, ratio, frac, ratio2, frac2,
                  a, b):
        return (n_sig * _sig_cdf_tgauss(xe, mu, sigma, ratio, frac, ratio2,
                                        frac2, lo, hi)
                + n_bkg * bkg_cdf(xe, a, b, lo, hi))

    s = dict(n_sig=0.5 * n_tot, n_bkg=0.5 * n_tot,
             mu=145.43, sigma=0.4, a=1.0, b=1.0)
    if signal == "dgauss":
        s.update(ratio=2.0, frac=0.8)
        s = {k: s[k] for k in ("n_sig", "n_bkg", "mu", "sigma",
                               "ratio", "frac", "a", "b")}
    elif signal == "tgauss":
        s.update(ratio=1.8, frac=0.6, ratio2=2.0, frac2=0.7)
        s = {k: s[k] for k in ("n_sig", "n_bkg", "mu", "sigma", "ratio",
                               "frac", "ratio2", "frac2", "a", "b")}
    if start:
        s.update(start)

    model = {"gauss": _model, "dgauss": _model_dg,
             "tgauss": _model_tg}[signal]
    cost = ExtendedBinnedNLL(counts, edges, model)
    m = Minuit(cost, **s)
    m.limits["n_sig", "n_bkg"] = (0, None)
    m.limits["mu"] = (144.0, 147.0)
    m.limits["sigma"] = (0.1, 3.0)
    m.limits["a"] = (0.05, 50.0)
    m.limits["b"] = (-10.0, 10.0)
    if signal in ("dgauss", "tgauss"):
        m.limits["ratio"] = (1.05, 10.0)
        m.limits["frac"] = (0.05, 0.95)
    if signal == "tgauss":
        m.limits["ratio2"] = (1.02, 10.0)
        m.limits["frac2"] = (0.02, 0.98)
    if fixed_shape:
        for k, v in fixed_shape.items():
            m.values[k] = v
            m.fixed[k] = True
    m.migrad()
    m.hesse()

    n_free = int(sum(1 for k in m.parameters if not m.fixed[k]))
    return DmFitResult(
        at_limits=_at_limits(m),
        n_sig=m.values["n_sig"], n_sig_err=m.errors["n_sig"],
        n_bkg=m.values["n_bkg"], n_bkg_err=m.errors["n_bkg"],
        mu=m.values["mu"], sigma=m.values["sigma"],
        a=m.values["a"], b=m.values["b"],
        valid=bool(m.valid), counts=counts, edges=edges,
        ratio=m.values["ratio"] if signal in ("dgauss", "tgauss") else 1.0,
        frac=m.values["frac"] if signal in ("dgauss", "tgauss") else 1.0,
        ratio2=m.values["ratio2"] if signal == "tgauss" else 1.0,
        frac2=m.values["frac2"] if signal == "tgauss" else 1.0,
        bkg=bkg, signal=signal, nfree=n_free, fval=float(m.fval),
        fixed_shape=dict(fixed_shape) if fixed_shape else None,
    )


SHAPE_PARS = ("mu", "sigma", "a", "b")
DG_SHAPE_PARS = ("mu", "sigma", "ratio", "frac", "a", "b")
TG_SHAPE_PARS = ("mu", "sigma", "ratio", "frac", "ratio2", "frac2", "a", "b")


def shape_pars(signal):
    return ("mu", "sigma") + SIGNAL_EXTRA[signal] + ("a", "b")


def lr_significance(r):
    if not (r.valid and np.isfinite(r.fval)):
        return None
    fixed = dict(r.fixed_shape or {})
    fixed["n_sig"] = 0.0
    n_tot = float(np.sum(r.counts))
    try:
        b = fit_dm_binned(r.counts, r.edges, signal=r.signal, bkg=r.bkg,
                          fixed_shape=fixed,
                          start={"n_bkg": max(n_tot, 1.0),
                                 "mu": r.mu, "sigma": r.sigma,
                                 "a": r.a, "b": r.b,
                                 **{k: r.shape_values[k]
                                    for k in SIGNAL_EXTRA.get(r.signal, ())}})
    except (RuntimeError, ValueError):
        return None
    if not np.isfinite(b.fval):
        return None
    return float(np.sqrt(max(b.fval - r.fval, 0.0)))


def significant(r, n_sigma=3.0, method="lr"):
    if not (r.valid and r.n_sig_err > 0):
        return False
    if method == "hesse":
        return bool(r.n_sig >= n_sigma * r.n_sig_err)
    if method != "lr":
        raise ValueError(f"unknown significance method {method!r}")
    s = lr_significance(r)
    if s is None:
        return bool(r.n_sig >= n_sigma * r.n_sig_err)
    return bool(s >= n_sigma)


def _shared_shape(combined, signal, share_mean, shape_shift=None,
                  share_width=True):
    pars = shape_pars(signal)
    drop = set()
    if not share_mean:
        drop.add("mu")
    if not share_width:
        drop.add("sigma")
    pars = tuple(p for p in pars if p not in drop)
    shape = {p: getattr(combined, p) for p in pars}
    if shape_shift:
        for k, v in shape_shift.items():
            if k in shape:
                shape[k] += v
    return shape


def _category_start(combined, n_frac, share_mean, share_width=True):
    start = {"n_sig": combined.n_sig * n_frac, "n_bkg": combined.n_bkg * n_frac}
    if not share_mean:
        start["mu"] = combined.mu
    if not share_width:
        start["sigma"] = combined.sigma
    return start


def fit_categories(dm_by_cat, share_shape=True, lo=DM_FIT_LO, hi=DM_FIT_HI,
                   nbins=DM_NBINS, signal="gauss", bkg="threshold",
                   share_mean=False, shape_shift=None, share_width=True):
    kw = dict(lo=lo, hi=hi, nbins=nbins, signal=signal, bkg=bkg)
    if not share_shape:
        return {k: fit_dm(v, **kw) for k, v in dm_by_cat.items()}, None

    combined = fit_dm(np.concatenate([np.asarray(v, float)
                                      for v in dm_by_cat.values()]), **kw)
    shape = _shared_shape(combined, signal, share_mean, shape_shift,
                          share_width)
    n_all = max(sum(len(v) for v in dm_by_cat.values()), 1)
    results = {}
    for name, values in dm_by_cat.items():
        results[name] = fit_dm(
            values, fixed_shape=shape,
            start=_category_start(combined, len(values) / n_all, share_mean,
                                  share_width),
            **kw)
    return results, combined


def fit_categories_binned(counts_by_cat, edges, signal="gauss",
                          bkg="threshold", share_mean=False, shape_shift=None,
                          share_width=True):
    counts_by_cat = {k: np.asarray(v) for k, v in counts_by_cat.items()}
    total = sum(counts_by_cat.values())
    combined = fit_dm_binned(total, edges, signal=signal, bkg=bkg)
    if not combined.valid:
        return None, combined
    shape = _shared_shape(combined, signal, share_mean, shape_shift,
                          share_width)
    n_all = max(int(total.sum()), 1)
    results = {}
    for name, counts in counts_by_cat.items():
        results[name] = fit_dm_binned(
            counts, edges, fixed_shape=shape, signal=signal, bkg=bkg,
            start=_category_start(combined, int(counts.sum()) / n_all,
                                  share_mean, share_width))
    return results, combined


def toy_pull_study(true_pars, n_toys=300, lo=DM_FIT_LO, hi=DM_FIT_HI,
                   nbins=DM_NBINS, seed_offset=0, signal="gauss"):
    rng = np.random.default_rng(SEED + seed_offset)
    edges = np.linspace(lo, hi, nbins + 1)
    keys = ("n_sig", "n_bkg", "mu", "sigma", "a", "b") + SIGNAL_EXTRA[signal]
    start = {k: true_pars[k] for k in keys}
    extra = {k: start[k] for k in SIGNAL_EXTRA[signal]}
    cdf = (start["n_sig"] * sig_cdf(signal, edges, start["mu"], start["sigma"],
                                    extra, lo, hi)
           + start["n_bkg"] * _bkg_cdf(edges, start["a"], start["b"], lo, hi))
    pred = np.diff(cdf)

    pulls, failed = [], 0
    for _ in range(n_toys):
        r = fit_dm_binned(rng.poisson(pred), edges, start=start, signal=signal)
        if not r.valid or r.n_sig_err == 0:
            failed += 1
            continue
        pulls.append((r.n_sig - start["n_sig"]) / r.n_sig_err)

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


@dataclass
class JointFitResult:
    valid: bool
    at_limits: tuple
    fval: float
    nfree: int
    nbins: int
    cats: tuple
    params: dict
    errors: dict
    covariance: dict
    edges: np.ndarray
    signal: str = "dgauss"
    bkg: str = "threshold"

    @property
    def ndf(self):
        return int(self.nbins - self.nfree)

    @property
    def chi2_ndf(self):
        return self.fval / self.ndf if self.ndf > 0 else None

    @property
    def pvalue(self):
        return (float(_chi2_dist.sf(self.fval, self.ndf))
                if self.ndf > 0 else None)

    def cov_nsig(self, c1, c2):
        return float(self.covariance.get((f"n_sig_{c1}", f"n_sig_{c2}"), 0.0))

    def gof(self):
        return {"chi2_lr": self.fval, "ndf": self.ndf,
                "chi2_ndf": self.chi2_ndf, "p_value": self.pvalue,
                "n_free": self.nfree, "nbins": self.nbins,
                "at_limits": list(self.at_limits)}


def _bc_cost(counts, nu):
    nu = np.clip(nu, 1e-12, None)
    term = nu - counts
    nz = counts > 0
    term[nz] += counts[nz] * np.log(counts[nz] / nu[nz])
    return 2.0 * float(np.sum(term))


def fit_categories_joint(counts_by_cat, edges, signal="dgauss",
                         bkg="threshold", share_mean=False, share_width=True,
                         seed_from=None):
    cats = tuple(counts_by_cat)
    if len(cats) != 2:
        raise ValueError("the joint fit is defined for exactly two flavour "
                         f"categories, got {cats}")
    counts = {c: np.asarray(counts_by_cat[c], dtype=float) for c in cats}
    edges = np.asarray(edges, dtype=float)
    lo, hi = float(edges[0]), float(edges[-1])
    bkg_cdf = _BKG_CDFS[bkg]
    if signal not in SIGNAL_EXTRA:
        raise ValueError(f"unknown signal model {signal!r}")
    extra = SIGNAL_EXTRA[signal]

    names, x0, limits = [], [], []

    def add(name, value, limit):
        names.append(name)
        x0.append(value)
        limits.append(limit)

    seed = seed_from
    if seed is None:
        try:
            seed = fit_dm_binned(sum(counts.values()), edges, signal=signal,
                                 bkg=bkg)
        except (RuntimeError, ValueError):
            seed = None
        if seed is not None and not seed.valid:
            seed = None
    for c in cats:
        n_tot = max(float(counts[c].sum()), 1.0)
        add(f"n_sig_{c}", 0.5 * n_tot, (0.0, None))
        add(f"n_bkg_{c}", 0.5 * n_tot, (0.0, None))
    mu0 = float(getattr(seed, "mu", 145.43))
    if share_mean:
        add("mu", mu0, (144.0, 147.0))
    else:
        for c in cats:
            add(f"mu_{c}", mu0, (144.0, 147.0))
    sg0 = float(getattr(seed, "sigma", 0.4))
    if share_width:
        add("sigma", sg0, (0.1, 3.0))
    else:
        for c in cats:
            add(f"sigma_{c}", sg0, (0.1, 3.0))
    _EXTRA_START = {"ratio": 2.0, "frac": 0.8, "ratio2": 2.0, "frac2": 0.7}
    _EXTRA_LIMITS = {"ratio": (1.05, 10.0), "frac": (0.05, 0.95),
                     "ratio2": (1.02, 10.0), "frac2": (0.02, 0.98)}
    for name in extra:
        add(name, float(getattr(seed, name, _EXTRA_START[name])),
            _EXTRA_LIMITS[name])
    add("a", float(getattr(seed, "a", 1.0)), (0.05, 50.0))
    add("b", float(getattr(seed, "b", 1.0)), (-10.0, 10.0))

    index = {n: i for i, n in enumerate(names)}

    def per_cat(pars, c):
        mu = pars[index["mu" if share_mean else f"mu_{c}"]]
        sigma = pars[index["sigma" if share_width else f"sigma_{c}"]]
        shape = {k: pars[index[k]] for k in extra}
        sig = sig_cdf(signal, edges, mu, sigma, shape, lo, hi)
        cdf = (pars[index[f"n_sig_{c}"]] * sig
               + pars[index[f"n_bkg_{c}"]]
               * bkg_cdf(edges, pars[index["a"]], pars[index["b"]], lo, hi))
        return np.diff(cdf)

    def cost(pars):
        return sum(_bc_cost(counts[c], per_cat(pars, c)) for c in cats)

    cost.errordef = Minuit.LEAST_SQUARES

    m = Minuit(cost, np.asarray(x0, dtype=float), name=names)
    for n, lim in zip(names, limits):
        m.limits[n] = lim
    m.migrad()
    m.hesse()

    cov = {}
    if m.covariance is not None:
        for i in names:
            for j in names:
                cov[(i, j)] = float(m.covariance[i, j])

    pars = np.asarray([m.values[n] for n in names], dtype=float)
    nbins_tot = int(sum(int(np.sum(per_cat(pars, c) > 0)) for c in cats))
    joint = JointFitResult(
        valid=bool(m.valid), at_limits=tuple(_at_limits(m)),
        fval=float(m.fval), nfree=len(names),
        nbins=nbins_tot, cats=cats,
        params={n: float(m.values[n]) for n in names},
        errors={n: float(m.errors[n]) for n in names},
        covariance=cov, edges=edges, signal=signal, bkg=bkg)

    results = {}
    for c in cats:
        results[c] = DmFitResult(
            n_sig=float(m.values[f"n_sig_{c}"]),
            n_sig_err=float(m.errors[f"n_sig_{c}"]),
            n_bkg=float(m.values[f"n_bkg_{c}"]),
            n_bkg_err=float(m.errors[f"n_bkg_{c}"]),
            mu=float(m.values["mu" if share_mean else f"mu_{c}"]),
            sigma=float(m.values["sigma" if share_width else f"sigma_{c}"]),
            a=float(m.values["a"]), b=float(m.values["b"]),
            valid=bool(m.valid), counts=counts[c].astype(np.int64),
            edges=edges,
            **{k: float(m.values[k]) for k in extra},
            bkg=bkg, signal=signal, nfree=-1, fval=float(m.fval))
    return results, joint


def joint_asymmetry(counts_by_cat, edges, cats=("D0", "D0bar"), **kwargs):
    from .asymmetry import raw_asymmetry

    results, joint = fit_categories_joint(counts_by_cat, edges, **kwargs)
    if not joint.valid or joint.at_limits:
        return None
    c1, c2 = cats
    A, sA = raw_asymmetry(results[c1].n_sig, results[c2].n_sig,
                          results[c1].n_sig_err, results[c2].n_sig_err,
                          cov=joint.cov_nsig(c1, c2))
    return float(A), float(sA), results, joint
