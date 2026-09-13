# reflection check on the v3 KK and pipi candidates

from __future__ import annotations

import numpy as np
import uproot

from charm_acp import kinematics as kin
from charm_acp.config import BR, DATA_DIR
from charm_acp.kinematics import DM_DSTAR, M_D0, M_K, M_PI

FILES = [DATA_DIR / "archive" / "v3test_job0_magdown.root",
         DATA_DIR / "archive" / "v3test_job1_magup.root"]

PAIRS = {
    "KK": [("DstpD02KK", "D0"), ("DstmD02KK", "D_0")],
    "PiPi": [("DstpD02PiPi", "D0"), ("DstmD02PiPi", "D_0")],
    "KPi": [("DstD02KPi", "D0")],
}
PARTS = {"KK": ("Kplus", "Kminus"), "PiPi": ("piplus", "piminus"),
         "KPi": ("Kminus", "piplus")}
TUNE = "MC15TuneV1"


def load(mode):
    h1, h2 = PARTS[mode]
    cols = {}
    for path in FILES:
        f = uproot.open(path)
        for tname, d0 in PAIRS[mode]:
            t = f[f"{tname}/DecayTree"]
            want = [f"{d0}_MM", "Dst_2010_plus_MM"]
            for q in (h1, h2):
                want += [f"{q}_P{c}" for c in "XYZ"]
                want += [f"{q}_{TUNE}_ProbNNk", f"{q}_{TUNE}_ProbNNpi"]
            a = t.arrays(want, library="np")
            a["dm"] = a["Dst_2010_plus_MM"] - a[f"{d0}_MM"]
            a["m"] = a[f"{d0}_MM"]
            for k, v in a.items():
                cols.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items()}


print("both jobs combined (0.149 /pb)\n")

for mode in ("KK", "PiPi"):
    h1, h2 = PARTS[mode]
    a = load(mode)
    dm_tag = np.abs(a["dm"] - DM_DSTAR) < 0.6
    m = a["m"]
    n_tag = int(dm_tag.sum())

    in_win = dm_tag & (np.abs(m - M_D0) < 20)
    lo_sb = dm_tag & (m < M_D0 - 35)
    hi_sb = dm_tag & (m > M_D0 + 35)
    mid = n_tag - int(in_win.sum()) - int(lo_sb.sum()) - int(hi_sb.sum())
    print(f"===== {mode}: {len(m)} candidates, {n_tag} dm-tagged =====")
    print(f"  dm-tagged mass location: D0 window {int(in_win.sum())},  "
          f"below {int(lo_sb.sum())},  above {int(hi_sb.sum())},  "
          f"between {mid}")

    p1 = [a[f"{h1}_P{c}"] for c in "XYZ"]
    p2 = [a[f"{h2}_P{c}"] for c in "XYZ"]
    alt1 = kin.invariant_mass(*p1, M_K, *p2, M_PI)
    alt2 = kin.invariant_mass(*p1, M_PI, *p2, M_K)
    sb = lo_sb | hi_sb
    refl = (np.abs(alt1 - M_D0) < 25) | (np.abs(alt2 - M_D0) < 25)
    n_sb = int(sb.sum())
    n_refl = int((sb & refl).sum())
    print(f"  reflection test: {n_refl}/{n_sb} dm-tagged sideband candidates "
          f"land on the D0 mass under a K<->pi swap")

    print(f"  {'PID cut':34s} {'n(tag)':>7s} {'window':>7s} {'sb':>5s} "
          f"{'sig-0.5*sb':>11s}")
    def count(cut, label):
        c = dm_tag & cut
        w = int((c & (np.abs(m - M_D0) < 20)).sum())
        s = int((c & (np.abs(m - M_D0) > 35) & (np.abs(m - M_D0) < 75)).sum())
        print(f"  {label:34s} {int(c.sum()):>7d} {w:>7d} {s:>5d} "
              f"{w - 0.5 * s:>+11.1f}")

    count(np.ones(len(m), bool), "none (stripping only)")
    if mode == "KK":
        for thr in (0.3, 0.5, 0.7):
            cut = (a[f"{h1}_{TUNE}_ProbNNk"] > thr) & (a[f"{h2}_{TUNE}_ProbNNk"] > thr)
            count(cut, f"both kaons ProbNNk > {thr}")
        cut = ((a[f"{h1}_{TUNE}_ProbNNk"] > 0.5) & (a[f"{h2}_{TUNE}_ProbNNk"] > 0.5)
               & (a[f"{h1}_{TUNE}_ProbNNpi"] < 0.3) & (a[f"{h2}_{TUNE}_ProbNNpi"] < 0.3))
        count(cut, "ProbNNk>0.5 & ProbNNpi<0.3 (both)")
    else:
        for thr in (0.3, 0.5, 0.7):
            cut = (a[f"{h1}_{TUNE}_ProbNNpi"] > thr) & (a[f"{h2}_{TUNE}_ProbNNpi"] > thr)
            count(cut, f"both pions ProbNNpi > {thr}")
        cut = ((a[f"{h1}_{TUNE}_ProbNNpi"] > 0.5) & (a[f"{h2}_{TUNE}_ProbNNpi"] > 0.5)
               & (a[f"{h1}_{TUNE}_ProbNNk"] < 0.3) & (a[f"{h2}_{TUNE}_ProbNNk"] < 0.3))
        count(cut, "ProbNNpi>0.5 & ProbNNk<0.3 (both)")
    print()

a = load("KPi")
dm_tag = np.abs(a["dm"] - DM_DSTAR) < 0.6
m = a["m"]
w = int((dm_tag & (np.abs(m - M_D0) < 20)).sum())
s = int((dm_tag & (np.abs(m - M_D0) > 35) & (np.abs(m - M_D0) < 75)).sum())
print(f"KPi control, same counting: window {w}, sb {s}, sig = {w - 0.5 * s:+.1f}")
print(f"BR-scaled true-signal expectations in this slice: "
      f"KK ~{(w - 0.5 * s) * BR['KK'] / BR['KPi']:.0f}, "
      f"PiPi ~{(w - 0.5 * s) * BR['PiPi'] / BR['KPi']:.0f} "
      f"(x line-efficiency ratio ~0.2-0.6 from v1)")
