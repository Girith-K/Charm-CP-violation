# Raw asymmetries, Delta A_CP, blinding and (pT, eta) binning

from __future__ import annotations

import numpy as np


def raw_asymmetry(n_d0, n_d0bar, s_d0, s_d0bar):
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


def polarity_average(a1, s1, a2, s2):
    # equal weights by construction, cancels polarity-odd detection asymmetry
    return 0.5 * (a1 + a2), 0.5 * float(np.hypot(s1, s2))


def blind_offset(salt="", passphrase=None, scale=None):
    import hashlib

    from . import config

    passphrase = passphrase or config.blind_passphrase()
    scale = scale if scale is not None else config.blind_scale()
    msg = passphrase if not salt else f"{passphrase}|{salt}"
    h = hashlib.sha256(msg.encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    return (2.0 * u - 1.0) * scale


def _production_salt(mode=None):
    from . import config

    return config.PRODUCTION if mode is None else f"{config.PRODUCTION}|{mode}"


def blind_raw(value, sigma, mode):
    return value + blind_offset(salt=_production_salt(mode)), sigma


def blind_delta_acp(delta, sigma):
    return delta + blind_offset(salt=_production_salt()), sigma


def unblind(blinded_value, mode=None):
    return blinded_value - blind_offset(salt=_production_salt(mode))


def weighted_average(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(errors))
            and np.all(errors > 0)):
        raise ValueError("weighted_average needs finite values and errors > 0")
    w = 1.0 / errors**2
    avg = np.sum(w * values) / np.sum(w)
    err = np.sqrt(1.0 / np.sum(w))
    return avg, err


def bin_index_2d(x, y, x_edges, y_edges):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    ix = np.digitize(x, x_edges) - 1
    iy = np.digitize(y, y_edges) - 1
    # keep entries sitting exactly on the top edge, matching np.histogram
    ix = np.where(x == x_edges[-1], nx - 1, ix)
    iy = np.where(y == y_edges[-1], ny - 1, iy)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.where(valid, ix * ny + iy, -1)


def binned_delta_acp(n_d0_kk, n_d0bar_kk, s_d0_kk, s_d0bar_kk,
                     n_d0_pp, n_d0bar_pp, s_d0_pp, s_d0bar_pp):
    with np.errstate(invalid="ignore", divide="ignore"):
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
