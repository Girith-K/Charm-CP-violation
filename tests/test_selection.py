# tests for one candidate per event and the tree layouts

import numpy as np

from charm_acp import selection as sel


def test_single_candidate_pick_unique_per_event():
    run = np.array([1, 1, 1, 2, 2, 3])
    evt = np.array([10, 10, 11, 5, 5, 7])
    vchi2 = np.array([2.0, 1.0, 3.0, 4.0, 4.0, 1.0])
    pick = sel.single_candidate_pick(run, evt, vchi2)
    assert int(pick.sum()) == 4
    assert pick[1] and not pick[0]


def test_single_candidate_pick_charge_symmetric_on_ties():
    n = 20000
    run = np.repeat(np.arange(n) // 100, 2)
    evt = np.repeat(np.arange(n), 2)
    vchi2 = np.ones(2 * n)
    tag = np.tile(np.array([1, -1], dtype=np.int8), n)
    pick = sel.single_candidate_pick(run, evt, vchi2)
    assert int(pick.sum()) == n
    asym = float(tag[pick].mean())
    assert abs(asym) < 4.0 / np.sqrt(n)


def test_single_candidate_pick_deterministic():
    run = np.repeat(np.arange(50), 2)
    evt = np.repeat(np.arange(50), 2)
    vchi2 = np.ones(100)
    p1 = sel.single_candidate_pick(run, evt, vchi2)
    p2 = sel.single_candidate_pick(run, evt, vchi2)
    assert np.array_equal(p1, p2)


def test_normalize_d0_prefix():
    arr = {"D_0_MM": np.array([1.0]), "Dst_2010_plus_MM": np.array([2.0])}
    out = sel._normalize(arr, "D_0")
    assert "D0_MM" in out and "Dst_2010_plus_MM" in out
    assert sel._normalize(arr, "D0") is arr


def test_layouts_cover_modes():
    for layout in ("v2", "v3"):
        for mode in ("KK", "PiPi", "KPi"):
            entries = sel.trees(mode, layout)
            assert entries
            for tname, d0 in entries:
                assert tname.endswith("/DecayTree")
                assert f"{d0}_MM" in sel.branches(mode, d0)


def test_flavour_tag_sign_convention():
    arr = {"Dst_2010_plus_ID": np.array([413, -413, 413, -413])}
    tag = sel.flavour_tag(arr)
    assert list(tag) == [1, -1, 1, -1]
    mask = np.ones(4, dtype=bool)
    cats = sel.split_categories(arr, mask, "KK")
    assert list(cats["D0"]) == [True, False, True, False]
    assert list(cats["D0bar"]) == [False, True, False, True]


def test_soft_fiducial_mask_geometry():
    arr = {
        "D0_MM": np.zeros(5),
        "piplus_PX": np.array([0.0, 0.0, 1500.0, 500.0, 0.0]) ,
        "piplus_PZ": np.array([3000.0, 5000.0, 5000.0, 5000.0, 9000.0]),
    }
    keep = sel.soft_fiducial_mask(arr, "KK", sel.NOMINAL_CUTS)
    assert list(keep) == [False, True, False, True, True]


def test_soft_fiducial_mask_off_keeps_everything():
    import dataclasses
    arr = {"D0_MM": np.zeros(3),
           "piplus_PX": np.zeros(3), "piplus_PZ": np.zeros(3)}
    cuts = dataclasses.replace(sel.NOMINAL_CUTS, soft_fiducial=False)
    assert sel.soft_fiducial_mask(arr, "KK", cuts).all()


def _synthetic(mode, n):
    p = sel.PARTS[mode]
    arr = {
        "D0_MM": np.full(n, 1864.84),
        "D0_IPCHI2_OWNPV": np.zeros(n),
        "D0_TAU": np.full(n, 1.0),
    }
    for part in (p["h1"], p["h2"], p["soft"]):
        arr[f"{part}_TRACK_GhostProb"] = np.zeros(n)
        arr[f"{part}_PT"] = np.full(n, 1000.0)
        arr[f"{part}_PIDK"] = np.full(n, 10.0 if part.startswith("K") else -10.0)
    arr[f"{p['soft']}_PX"] = np.zeros(n)
    arr[f"{p['soft']}_PZ"] = np.full(n, 9000.0)
    return arr


def test_cutflow_ratios_are_per_stage():
    n = 10
    arr = _synthetic("KK", n)
    mask, table = sel.cutflow(arr, "KK", sel.NOMINAL_CUTS)
    assert table[0] == ("all candidates", n, 1.0)
    for name, count, ratio in table[1:]:
        assert 0.0 <= ratio <= 1.0
    assert int(mask.sum()) == table[-1][1]
