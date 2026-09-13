# tests for the kinematics functions

import numpy as np

from charm_acp import kinematics as kin


def test_invariant_mass_at_rest():
    m = kin.invariant_mass(0, 0, 0, kin.M_K, 0, 0, 0, kin.M_K)
    assert np.isclose(m, 2 * kin.M_K, atol=1e-6)


def test_invariant_mass_back_to_back():
    p, m = 1000.0, kin.M_K
    e = np.sqrt(p * p + m * m)
    got = kin.invariant_mass(0, 0, p, m, 0, 0, -p, m)
    assert np.isclose(got, 2 * e, atol=1e-6)


def test_invariant_mass_is_vectorised():
    px1 = np.array([0.0, 0.0]); pz1 = np.array([1000.0, 500.0])
    px2 = np.array([0.0, 0.0]); pz2 = np.array([-1000.0, -500.0])
    z = np.zeros(2)
    out = kin.invariant_mass(px1, z, pz1, kin.M_PI, px2, z, pz2, kin.M_PI)
    assert out.shape == (2,)
    assert np.all(out > 2 * kin.M_PI)


def test_delta_m_peaks_near_physical_value():
    d0 = (0.0, 0.0, 0.0, kin.M_D0)
    ppi = 40.0
    epi = np.sqrt(ppi * ppi + kin.M_PI**2)
    soft = (0.0, 0.0, ppi, epi)
    dm = kin.delta_m(d0, soft)
    assert dm > 0
    assert 100.0 < dm < 200.0


def test_eta_zero_in_transverse_plane():
    assert np.isclose(kin.eta(1000.0, 0.0, 0.0), 0.0, atol=1e-9)
