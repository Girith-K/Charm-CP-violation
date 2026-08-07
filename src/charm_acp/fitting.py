"""Δm fit model (Gaussian + threshold background), fit driver, toy pulls

Models are CDFs, not PDFs, because ExtendedBinnedNLL wants cumulative expected
counts at the bin edges
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import norm

from .config import DM_FIT_HI, DM_FIT_LO, DM_NBINS, SEED
from .kinematics import M_PI

# no closed form CDF for the threshold shape, so integrate it on a fine grid
_GRID_N = 2001


def threshold_density(dm, a, b, m_pi=M_PI):
    """Unnormalised threshold background density, 0 below m_pi"""
    dm = np.asarray(dm, dtype=float)
    x = np.clip(dm - m_pi, 0.0, None)
    out = (1.0 - np.exp(-x / a)) * np.power(np.maximum(dm, m_pi) / m_pi, b)
    return np.where(dm > m_pi, out, 0.0)


def _bkg_cdf(edges, a, b, lo=DM_FIT_LO, hi=DM_FIT_HI):
    """CDF of the normalised threshold density on [lo, hi]"""
    grid = np.linspace(lo, hi, _GRID_N)
    dens = threshold_density(grid, a, b)
    cum = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) * 0.5 * np.diff(grid))])
    total = cum[-1]
    return np.interp(edges, grid, cum / total)


def _sig_cdf_gauss(edges, mu, sigma, lo=DM_FIT_LO, hi=DM_FIT_HI):
    """Gaussian CDF normalised to the fit window"""
    c = norm.cdf(edges, mu, sigma)
    c_lo, c_hi = norm.cdf(lo, mu, sigma), norm.cdf(hi, mu, sigma)
    return (c - c_lo) / (c_hi - c_lo)


def model_cdf(edges, n_sig, n_bkg, mu, sigma, a, b):
    """Cumulative expected counts at `edges` for signal + background"""
    return n_sig * _sig_cdf_gauss(edges, mu, sigma) + n_bkg * _bkg_cdf(edges, a, b)


@dataclass
class DmFitResult:
    """Result of one Δm fit"""

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

    @property
    def pred(self):
        return np.diff(model_cdf(self.edges, self.n_sig, self.n_bkg,
                                 self.mu, self.sigma, self.a, self.b))

    @property
    def pulls(self):
        p = self.pred
        return (self.counts - p) / np.sqrt(np.where(p > 0, p, 1.0))


def fit_dm(dm_values, lo=DM_FIT_LO, hi=DM_FIT_HI, nbins=DM_NBINS,
           start=None, fixed_shape=None):
    """Extended binned ML fit of one Δm distribution"""
    dm_values = np.asarray(dm_values, dtype=float)
    counts, edges = np.histogram(dm_values, bins=nbins, range=(lo, hi))
    return fit_dm_binned(counts, edges, start=start, fixed_shape=fixed_shape)


def fit_dm_binned(counts, edges, start=None, fixed_shape=None):
    counts = np.asarray(counts)
    n_tot = counts.sum()

    s = dict(n_sig=0.5 * n_tot, n_bkg=0.5 * n_tot,
             mu=145.43, sigma=0.4, a=1.0, b=1.0)
    if start:
        s.update(start)

    cost = ExtendedBinnedNLL(counts, edges, model_cdf)
    m = Minuit(cost, **s)
    m.limits["n_sig", "n_bkg"] = (0, None)
    m.limits["mu"] = (144.0, 147.0)      # peak can't wander off
    m.limits["sigma"] = (0.1, 3.0)       # Δm resolution is under 1 MeV at LHCb
    m.limits["a"] = (0.05, 50.0)
    m.limits["b"] = (-10.0, 10.0)
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
    )


SHAPE_PARS = ("mu", "sigma", "a", "b")


def fit_categories(dm_by_cat, share_shape=True, lo=DM_FIT_LO, hi=DM_FIT_HI,
                   nbins=DM_NBINS):
    kw = dict(lo=lo, hi=hi, nbins=nbins)
    if not share_shape:
        return {k: fit_dm(v, **kw) for k, v in dm_by_cat.items()}, None

    combined = fit_dm(np.concatenate([np.asarray(v, float)
                                      for v in dm_by_cat.values()]), **kw)
    shape = {p: getattr(combined, p) for p in SHAPE_PARS}
    results = {}
    for name, values in dm_by_cat.items():
        # start at each category's share of the combined yields
        frac = len(values) / max(sum(len(v) for v in dm_by_cat.values()), 1)
        results[name] = fit_dm(values, start={"n_sig": combined.n_sig * frac,
                                              "n_bkg": combined.n_bkg * frac},
                               fixed_shape=shape, **kw)
    return results, combined


def toy_pull_study(true_pars, n_toys=300, lo=DM_FIT_LO, hi=DM_FIT_HI,
                   nbins=DM_NBINS, seed_offset=0):
    rng = np.random.default_rng(SEED + seed_offset)
    edges = np.linspace(lo, hi, nbins + 1)
    pred = np.diff(model_cdf(edges, **true_pars))

    pulls, failed = [], 0
    for _ in range(n_toys):
        toy_counts = rng.poisson(pred)
        cost = ExtendedBinnedNLL(toy_counts, edges, model_cdf)
        m = Minuit(cost, **true_pars)
        m.limits["n_sig", "n_bkg"] = (0, None)
        m.limits["mu"] = (144.0, 147.0)
        m.limits["sigma"] = (0.1, 3.0)
        m.limits["a"] = (0.05, 50.0)
        m.limits["b"] = (-10.0, 10.0)
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
