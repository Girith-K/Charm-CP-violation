"""Invariant mass, Δm, and four vector helpers

Vectorised over candidates, no python loops. Everything in MeV
"""

from __future__ import annotations

import numpy as np

# PDG values (MeV). Use `particle` for real code, these are just the default
M_PI = 139.570  # charged pion
M_K = 493.677  # charged kaon
M_D0 = 1864.84  # D0
DM_DSTAR = 145.43  # m(D*+) - m(D0), the Δm signal peak


def energy(px, py, pz, mass):
    """E = sqrt(|p|^2 + m^2), elementwise. Works for numpy or awkward arrays"""
    return np.sqrt(px * px + py * py + pz * pz + mass * mass)


def invariant_mass(px1, py1, pz1, m1, px2, py2, pz2, m2):
    """Invariant mass of a two body system under assigned daughter masses

    `m1` and `m2` are the hypothesis masses. A wrong hypothesis shifts and
    smears the peak, which is why PID matters
    """
    e1 = energy(px1, py1, pz1, m1)
    e2 = energy(px2, py2, pz2, m2)
    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    m2_ = e * e - (px * px + py * py + pz * pz)
    # careful with tiny negative values from float error before sqrt
    return np.sqrt(np.clip(m2_, 0.0, None))


def sum_four_momentum(parts):
    """Sum (px, py, pz, E) tuples into one four momentum"""
    px = sum(p[0] for p in parts)
    py = sum(p[1] for p in parts)
    pz = sum(p[2] for p in parts)
    e = sum(p[3] for p in parts)
    return px, py, pz, e


def mass_from_four_momentum(px, py, pz, e):
    """Invariant mass from a summed four momentum"""
    m2 = e * e - (px * px + py * py + pz * pz)
    return np.sqrt(np.clip(m2, 0.0, None))


def delta_m(d0_p4, soft_pi_p4):
    """Δm = m(D0 πs) - m(D0)

    Same D0 four momentum in both terms, so the D0 mass resolution mostly
    cancels and Δm peaks sharply at 145.43 MeV
    """
    m_d0 = mass_from_four_momentum(*d0_p4)
    dstar = sum_four_momentum([d0_p4, soft_pi_p4])
    m_dstar = mass_from_four_momentum(*dstar)
    return m_dstar - m_d0


def pt(px, py):
    return np.sqrt(px * px + py * py)
def eta(px, py, pz):
    p = np.sqrt(px * px + py * py + pz * pz)
    # eta = atanh(pz/p), clipped to guard the |pz/p| -> 1 limit
    ratio = np.clip(pz / np.where(p == 0.0, np.nan, p), -1 + 1e-12, 1 - 1e-12)
    return np.arctanh(ratio)
