"""Raw asymmetries, ΔA_CP, blinding, and (pT, η) binning.

    A_raw(f) = [N(D0→f) - N(D̄0→f)] / [N(D0→f) + N(D̄0→f)]
    ΔA_CP    = A_raw(KK) - A_raw(ππ) ≈ A_CP(KK) - A_CP(ππ)
Errors propagate from the fit yields, not raw √N.
"""

from __future__ import annotations

import numpy as np


def raw_asymmetry(n_d0, n_d0bar, s_d0, s_d0bar):
    """Raw asymmetry and its error, propagated from the fit yield errors."""
    N = n_d0 + n_d0bar
    A = (n_d0 - n_d0bar) / N
    dA_dn = 2 * n_d0bar / N**2
    dA_dnb = -2 * n_d0 / N**2
    sigma_A = np.hypot(dA_dn * s_d0, dA_dnb * s_d0bar)
    return A, sigma_A


def raw_asymmetry_counting(n_d0, n_d0bar):
    N = n_d0 + n_d0bar
    A = (n_d0 - n_d0bar) / N
    sigma_A = np.sqrt((1 - A * A) / N)
    return A, sigma_A


def delta_acp(A_kk, s_kk, A_pp, s_pp):
    return A_kk - A_pp, np.hypot(s_kk, s_pp)


def blind_offset(passphrase=None, scale=None):
    """Deterministic hidden offset for blinding.
    Reproducible across runs, and nobody computes it by hand before the end.
    """
    import hashlib

    from . import config

    passphrase = passphrase or config.BLIND_PASSPHRASE
    scale = scale if scale is not None else config.BLIND_SCALE
    h = hashlib.sha256(passphrase.encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64  # uniform in [0, 1)
    return (2.0 * u - 1.0) * scale


def blind_delta_acp(delta, sigma):
    """Blinded ΔA_CP. The error is untouched so it can be worked on blind."""
    return delta + blind_offset(), sigma


def unblind(blinded_value):
    return blinded_value - blind_offset()


def weighted_average(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    w = 1.0 / errors**2
    avg = np.sum(w * values) / np.sum(w)
    err = np.sqrt(1.0 / np.sum(w))
    return avg, err


def bin_index_2d(x, y, x_edges, y_edges):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    ix = np.digitize(x, x_edges) - 1
    iy = np.digitize(y, y_edges) - 1
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.where(valid, ix * ny + iy, -1)


def binned_delta_acp(n_d0_kk, n_d0bar_kk, s_d0_kk, s_d0bar_kk,
                     n_d0_pp, n_d0bar_pp, s_d0_pp, s_d0bar_pp):
    A_kk, sA_kk = raw_asymmetry(np.asarray(n_d0_kk, float), np.asarray(n_d0bar_kk, float),
                                np.asarray(s_d0_kk, float), np.asarray(s_d0bar_kk, float))
    A_pp, sA_pp = raw_asymmetry(np.asarray(n_d0_pp, float), np.asarray(n_d0bar_pp, float),
                                np.asarray(s_d0_pp, float), np.asarray(s_d0bar_pp, float))
    d_bins = A_kk - A_pp
    s_bins = np.hypot(sA_kk, sA_pp)
    good = np.isfinite(d_bins) & np.isfinite(s_bins) & (s_bins > 0)
    if not np.any(good):
        return np.nan, np.nan
    return weighted_average(d_bins[good], s_bins[good])
