# Offline selection cuts, cutflow, flavour tagging, tree layouts and dedup

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import uproot

from .config import SEED

PARTS = {
    "KK": {"h1": "Kplus", "h2": "Kminus", "soft": "piplus"},
    "PiPi": {"h1": "piplus", "h2": "piminus", "soft": "piplus_0"},
    "KPi": {"h1": "Kminus", "h2": "piplus", "soft": "piplus_0"},
}

LAYOUTS = {
    "v2": {
        "KK": [("DstD02KK/DecayTree", "D0")],
        "PiPi": [("DstD02PiPi/DecayTree", "D0")],
        "KPi": [("DstD02KPi/DecayTree", "D0")],
    },
    "v3": {
        "KK": [("DstpD02KK/DecayTree", "D0"), ("DstmD02KK/DecayTree", "D_0")],
        "PiPi": [("DstpD02PiPi/DecayTree", "D0"),
                 ("DstmD02PiPi/DecayTree", "D_0")],
        "KPi": [("DstD02KPi/DecayTree", "D0")],
    },
}


def trees(mode, layout):
    return LAYOUTS[layout][mode]


TUNE = "MC15TuneV1"


def branches(mode, d0="D0", tune=None):
    p = PARTS[mode]
    out = ["Dst_2010_plus_MM", "Dst_2010_plus_ID",
           "Dst_2010_plus_PX", "Dst_2010_plus_PY", "Dst_2010_plus_PZ",
           f"{d0}_MM", f"{d0}_IPCHI2_OWNPV", f"{d0}_ENDVERTEX_CHI2",
           f"{d0}_ENDVERTEX_NDOF", f"{d0}_TAU",
           "runNumber", "eventNumber"]
    for part in (p["h1"], p["h2"], p["soft"]):
        out += [f"{part}_PIDK", f"{part}_TRACK_GhostProb"]
        if tune:
            out += [f"{part}_{tune}_ProbNNk", f"{part}_{tune}_ProbNNpi"]
    soft = p["soft"]
    out += [f"{soft}_ID", f"{soft}_PT", f"{soft}_PX", f"{soft}_PZ"]
    return out


def _normalize(arr, d0):
    if d0 == "D0":
        return arr
    pre = f"{d0}_"
    return {("D0_" + k[len(pre):] if k.startswith(pre) else k): v
            for k, v in arr.items()}


def iter_mode(path, mode, layout, step_size="512 MB"):
    tune = TUNE if layout == "v3" else None
    for tname, d0 in LAYOUTS[layout][mode]:
        for arr in uproot.iterate({str(path): tname}, branches(mode, d0, tune),
                                  library="np", step_size=step_size):
            yield _normalize(arr, d0)


def single_candidate_pick(run, evt, vchi2, seed=SEED):
    rng = np.random.default_rng(seed)
    evt = np.asarray(evt).astype(np.int64)
    if evt.size and int(evt.max()) >= 10**10:
        raise ValueError("eventNumber >= 1e10 would collide the dedup key, "
                         "switch to a (run, evt) pair key")
    key = (np.asarray(run).astype(np.int64) * np.int64(10**10) + evt)
    order = np.lexsort((rng.random(len(key)), np.asarray(vchi2), key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    pick = np.zeros(len(order), dtype=bool)
    pick[order[first]] = True
    return pick


@dataclass(frozen=True)
class Cuts:
    pid_k_min: float = 5.0
    pid_pi_max: float = 0.0
    soft_pid: bool = True
    ghost_max: float = 0.3
    d0_mass_win: float = 25.0
    d0_ipchi2_max: float = 9.0
    d0_tau_min: float = 0.0
    soft_pt_min: float = 200.0
    soft_fiducial: bool = False
    probnn_k_min: float | None = None
    probnn_pi_min: float | None = None


DEFAULT_CUTS = Cuts()
NOMINAL_CUTS = Cuts(pid_k_min=0.0, soft_pid=False, d0_ipchi2_max=9.0,
                    soft_fiducial=True)

M_D0_PDG = 1864.84


def pid_mask(arr, mode, cuts=DEFAULT_CUTS):
    p = PARTS[mode]
    masks = []
    for part in (p["h1"], p["h2"]):
        if part.startswith("K"):
            masks.append(arr[f"{part}_PIDK"] > cuts.pid_k_min)
            if cuts.probnn_k_min is not None:
                masks.append(arr[f"{part}_{TUNE}_ProbNNk"] > cuts.probnn_k_min)
        else:
            masks.append(arr[f"{part}_PIDK"] < cuts.pid_pi_max)
            if cuts.probnn_pi_min is not None:
                masks.append(arr[f"{part}_{TUNE}_ProbNNpi"] > cuts.probnn_pi_min)
    if cuts.soft_pid:
        masks.append(arr[f"{p['soft']}_PIDK"] < cuts.pid_pi_max)
    return np.logical_and.reduce(masks)


def quality_mask(arr, mode, cuts=DEFAULT_CUTS):
    p = PARTS[mode]
    masks = [arr[f"{part}_TRACK_GhostProb"] < cuts.ghost_max
             for part in (p["h1"], p["h2"], p["soft"])]
    return np.logical_and.reduce(masks)


def d0_mass_mask(arr, mode, cuts=DEFAULT_CUTS):
    return np.abs(arr["D0_MM"] - M_D0_PDG) < cuts.d0_mass_win


def prompt_mask(arr, mode, cuts=DEFAULT_CUTS):
    return (arr["D0_IPCHI2_OWNPV"] < cuts.d0_ipchi2_max) & \
           (arr["D0_TAU"] > cuts.d0_tau_min)


def soft_pion_mask(arr, mode, cuts=DEFAULT_CUTS):
    return arr[f"{PARTS[mode]['soft']}_PT"] > cuts.soft_pt_min


def soft_fiducial_mask(arr, mode, cuts=DEFAULT_CUTS):
    if not cuts.soft_fiducial:
        return np.ones(len(arr["D0_MM"]), dtype=bool)
    px = arr[f"{PARTS[mode]['soft']}_PX"] / 1000.0
    pz = arr[f"{PARTS[mode]['soft']}_PZ"] / 1000.0
    edge = (pz < 4.0) | ((pz < 6.0) & (np.abs(px) > 1.0))
    return ~edge


CUT_SEQUENCE = [
    ("PID", pid_mask),
    ("track quality", quality_mask),
    ("D0 mass window", d0_mass_mask),
    ("prompt D0", prompt_mask),
    ("soft pion pT", soft_pion_mask),
    ("soft pion fiducial", soft_fiducial_mask),
]


def cutflow(arr, mode, cuts=DEFAULT_CUTS):
    n0 = len(arr["D0_MM"])
    mask = np.ones(n0, dtype=bool)
    table = [("all candidates", n0, 1.0)]
    for name, fn in CUT_SEQUENCE:
        prev = int(mask.sum())
        mask &= fn(arr, mode, cuts)
        now = int(mask.sum())
        table.append((name, now, now / prev if prev else 0.0))
    return mask, table


def flavour_tag(arr, mode=None):
    return np.where(arr["Dst_2010_plus_ID"] > 0, 1, -1).astype(np.int8)


def split_categories(arr, mask, mode):
    tag = flavour_tag(arr, mode)
    return {"D0": mask & (tag == 1), "D0bar": mask & (tag == -1)}
