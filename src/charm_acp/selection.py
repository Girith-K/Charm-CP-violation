
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
PARTS = {
    "KK": {"h1": "Kplus", "h2": "Kminus", "soft": "piplus"},
    "PiPi": {"h1": "piplus", "h2": "piminus", "soft": "piplus_0"},
    "KPi": {"h1": "Kminus", "h2": "piplus", "soft": "piplus_0"},
}

TREE = {"KK": "DstD02KK/DecayTree", "PiPi": "DstD02PiPi/DecayTree",
        "KPi": "DstD02KPi/DecayTree"}


def branches(mode):
    """All branches the selection, Δm and tagging need"""
    p = PARTS[mode]
    out = ["Dst_2010_plus_MM", "Dst_2010_plus_ID", "Dst_2010_plus_PT",
           "Dst_2010_plus_PX", "Dst_2010_plus_PY", "Dst_2010_plus_PZ",
           "D0_MM", "D0_ID", "D0_IPCHI2_OWNPV", "D0_ENDVERTEX_CHI2",
           "D0_ENDVERTEX_NDOF", "D0_TAU", "D0_PT",
           "runNumber", "eventNumber", "totCandidates", "nCandidate"]
    for part in (p["h1"], p["h2"], p["soft"]):
        out += [f"{part}_PIDK", f"{part}_ID", f"{part}_PT",
                f"{part}_TRACK_GhostProb", f"{part}_TRACK_CHI2NDOF"]
    out += [f"{p['soft']}_PX", f"{p['soft']}_PZ"]
    return out


@dataclass(frozen=True)
class Cuts:
    """Offline thresholds. Defaults = analysis baseline, Phase 9 varies them"""

    pid_k_min: float = 5.0
    pid_pi_max: float = 0.0
    soft_pid: bool = True
    ghost_max: float = 0.3
    d0_mass_win: float = 25.0    # |m(D0) - PDG| [MeV]
    d0_ipchi2_max: float = 9.0   # prompt cut
    d0_tau_min: float = 0.0      # drops the -100 ns failed fit sentinels
    soft_pt_min: float = 200.0   # MeV
    soft_fiducial: bool = False


DEFAULT_CUTS = Cuts()

M_D0_PDG = 1864.84  # MeV


def pid_mask(arr, mode, cuts=DEFAULT_CUTS):
    p = PARTS[mode]
    masks = []
    for part in (p["h1"], p["h2"]):
        if part.startswith("K"):
            masks.append(arr[f"{part}_PIDK"] > cuts.pid_k_min)
        else:
            masks.append(arr[f"{part}_PIDK"] < cuts.pid_pi_max)
    if cuts.soft_pid:
        masks.append(arr[f"{p['soft']}_PIDK"] < cuts.pid_pi_max)
    return np.logical_and.reduce(masks)


def quality_mask(arr, mode, cuts=DEFAULT_CUTS):
    """Reject ghost tracks"""
    p = PARTS[mode]
    masks = [arr[f"{part}_TRACK_GhostProb"] < cuts.ghost_max
             for part in (p["h1"], p["h2"], p["soft"])]
    return np.logical_and.reduce(masks)


def d0_mass_mask(arr, mode, cuts=DEFAULT_CUTS):
    """Keep candidates near the D0 mass, Δm fit does the rest"""
    return np.abs(arr["D0_MM"] - M_D0_PDG) < cuts.d0_mass_win


def prompt_mask(arr, mode, cuts=DEFAULT_CUTS):
    """Prompt D0, small IPχ² means from the PV not a b hadron decay

    Secondary charm has a different production asymmetry, which breaks the
    ΔA_CP cancellation
    """
    return (arr["D0_IPCHI2_OWNPV"] < cuts.d0_ipchi2_max) & \
           (arr["D0_TAU"] > cuts.d0_tau_min)


def soft_pion_mask(arr, mode, cuts=DEFAULT_CUTS):
    return arr[f"{PARTS[mode]['soft']}_PT"] > cuts.soft_pt_min


def soft_fiducial_mask(arr, mode, cuts=DEFAULT_CUTS):
    if not cuts.soft_fiducial:
        return np.ones(len(arr["D0_MM"]), dtype=bool)
    px = arr[f"{PARTS[mode]['soft']}_PX"] / 1000.0  # GeV
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
    """Apply the cut sequence, return (mask, table of name/survivors/eff)"""
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
    """+1 (D0) / -1 (D0bar) from the D* charge

    Identical to the soft pion charge but independent of the per tree naming
    """
    return np.where(arr["Dst_2010_plus_ID"] > 0, 1, -1).astype(np.int8)


def split_categories(arr, mask, mode):
    """Split selected candidates into the two flavour tagged categories"""
    tag = flavour_tag(arr, mode)
    return {"D0": mask & (tag == 1), "D0bar": mask & (tag == -1)}
