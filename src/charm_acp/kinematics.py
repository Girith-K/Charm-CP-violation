# masses, dm, pT and eta

from __future__ import annotations

import numpy as np

M_PI = 139.570
M_K = 493.677
M_D0 = 1864.84
DM_DSTAR = 145.4258


def energy(px, py, pz, mass):
    return np.sqrt(px * px + py * py + pz * pz + mass * mass)

def invariant_mass(px1, py1, pz1, m1, px2, py2, pz2, m2):
    e1 = energy(px1, py1, pz1, m1)
    e2 = energy(px2, py2, pz2, m2)
    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    m2_ = e * e - (px * px + py * py + pz * pz)
    return np.sqrt(np.clip(m2_, 0.0, None))


def sum_four_momentum(parts):
    px = sum(p[0] for p in parts)
    py = sum(p[1] for p in parts)
    pz = sum(p[2] for p in parts)
    e = sum(p[3] for p in parts)
    return px, py, pz, e


def mass_from_four_momentum(px, py, pz, e):
    m2 = e * e - (px * px + py * py + pz * pz)
    return np.sqrt(np.clip(m2, 0.0, None))


def delta_m(d0_p4, soft_pi_p4):
    m_d0 = mass_from_four_momentum(*d0_p4)
    dstar = sum_four_momentum([d0_p4, soft_pi_p4])
    m_dstar = mass_from_four_momentum(*dstar)
    return m_dstar - m_d0


def pt(px, py):
    return np.sqrt(px * px + py * py)


def eta(px, py, pz):
    p = np.sqrt(px * px + py * py + pz * pz)
    ratio = np.clip(pz / np.where(p == 0.0, np.nan, p), -1 + 1e-12, 1 - 1e-12)
    return np.arctanh(ratio)
