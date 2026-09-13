# works out which stripping line made each tree from where the cuts sit

import numpy as np
import uproot

from charm_acp import selection as sel
from charm_acp.config import production_files
from charm_acp.kinematics import DM_DSTAR

PATH = production_files("v2")[0][0]
print(f"fingerprinting: {PATH.name}\n")
f = uproot.open(PATH)

kk_keys = set(f[sel.trees("KK", "v2")[0][0]].keys())
d0_keys = sorted(k for k in kk_keys if k.startswith("D0_"))
print("D0-level branches available:")
print("  " + ", ".join(d0_keys) + "\n")


def q(x, lo=0.5, hi=99.5):
    return np.min(x), np.percentile(x, lo), np.percentile(x, hi), np.max(x)


for mode in ["KK", "PiPi", "KPi"]:
    t = f[sel.trees(mode, "v2")[0][0]]
    p = sel.PARTS[mode]
    want = ["D0_ID", "Dst_2010_plus_ID", "D0_MM", "Dst_2010_plus_MM",
            "D0_IPCHI2_OWNPV", "D0_PT",
            f"{p['h1']}_PT", f"{p['h2']}_PT", f"{p['soft']}_PT",
            f"{p['h1']}_PX", f"{p['h1']}_PY", f"{p['h1']}_PZ",
            f"{p['h2']}_PX", f"{p['h2']}_PY", f"{p['h2']}_PZ"]
    for extra in ("D0_FDCHI2_OWNPV", "D0_DIRA_OWNPV", "D0_ENDVERTEX_CHI2",
                  f"{p['h1']}_IPCHI2_OWNPV", f"{p['h2']}_IPCHI2_OWNPV",
                  f"{p['soft']}_IPCHI2_OWNPV"):
        if extra in kk_keys or extra in set(t.keys()):
            want.append(extra)
    a = t.arrays(want, library="np")

    print(f"===== {mode}  ({t.num_entries:,} candidates) =====")

    ids, counts = np.unique(a["D0_ID"], return_counts=True)
    comp = ", ".join(f"{int(i):+d}: {c:,}" for i, c in zip(ids, counts))
    print(f"  D0_ID composition      : {comp}")
    dst = a["Dst_2010_plus_ID"]
    if len(ids) > 1:
        agree = float(np.mean(np.sign(a["D0_ID"]) == np.sign(dst)))
        print(f"  sign(D0_ID)==sign(D*)  : {agree:.4%}")

    for part in (p["h1"], p["h2"]):
        pt = a[f"{part}_PT"]
        mom = np.sqrt(a[f"{part}_PX"]**2 + a[f"{part}_PY"]**2 + a[f"{part}_PZ"]**2)
        print(f"  {part:8s} pT  min={pt.min():8.1f}  p min={mom.min()/1000:6.2f} GeV")
    soft_pt = a[f"{p['soft']}_PT"]
    print(f"  {p['soft']:8s} pT  min={soft_pt.min():8.1f}")

    for part in (p["h1"], p["h2"]):
        k = f"{part}_IPCHI2_OWNPV"
        if k in a:
            print(f"  {part:8s} IPchi2 min = {a[k].min():.2f}")

    if "D0_FDCHI2_OWNPV" in a:
        print(f"  D0 FDchi2   min = {a['D0_FDCHI2_OWNPV'].min():.2f}   (D2hh: BPVVDCHI2>40)")
    if "D0_DIRA_OWNPV" in a:
        print(f"  D0 DIRA     min = {a['D0_DIRA_OWNPV'].min():.6f}   (D2hh: >0.9999)")
    print(f"  D0 IPchi2   median = {np.median(a['D0_IPCHI2_OWNPV']):8.2f}   "
          f"frac<9 = {np.mean(a['D0_IPCHI2_OWNPV'] < 9):.3%}")
    print(f"  D0 pT       min = {a['D0_PT'].min():.1f}  median = {np.median(a['D0_PT']):.1f}")

    mlo, m1, m2, mhi = q(a["D0_MM"])
    print(f"  m(D0) range  [{mlo:.1f}, {mhi:.1f}] MeV   (width {mhi-mlo:.1f})")
    dm = a["Dst_2010_plus_MM"] - a["D0_MM"]
    print(f"  dm    range  [{dm.min():.2f}, {dm.max():.2f}] MeV   (D2hh cut: dm<160)")
    print(f"  dm in +-1 MeV of {DM_DSTAR:.2f}: {np.mean(np.abs(dm - DM_DSTAR) < 1.0):.2%}")
    print()
