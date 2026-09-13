# double checks the v1 vs v2 comparison, cut edges, rates and sidebands

import numpy as np
import uproot

from charm_acp import selection as sel
from charm_acp.config import DATA_DIR, production_files
from charm_acp.kinematics import M_D0

TREES = {"KK": "DstD02KK/DecayTree", "PiPi": "DstD02PiPi/DecayTree",
         "KPi": "DstD02KPi/DecayTree"}
V1 = DATA_DIR / "archive" / "testprod_job0.root"
V2 = DATA_DIR / "archive" / "testprod_v2_job0.root"

print("=" * 72)
print("CLAIM A: do the v1 signal trees carry the D2hh cut edges?")
print("=" * 72)
print("D2hh signature: daughter pT>800, p>5000, IPchi2>9, FDchi2>40, "
      "DIRA>0.9999, D0 pT>2000\n")

for label, path in (("v1", V1), ("v2", V2)):
    f = uproot.open(path)
    print(f"--- {label}: {path.name} ---")
    for mode in ("KK", "PiPi", "KPi"):
        t = f[TREES[mode]]
        p = sel.PARTS[mode]
        want = [f"{p['h1']}_PT", f"{p['h1']}_PX", f"{p['h1']}_PY", f"{p['h1']}_PZ",
                "D0_PT", "D0_FDCHI2_OWNPV", "D0_DIRA_OWNPV", "D0_MM"]
        keys = set(t.keys())
        ipk = f"{p['h1']}_IPCHI2_OWNPV"
        if ipk in keys:
            want.append(ipk)
        a = t.arrays([w for w in want if w in keys], library="np")
        h1 = p["h1"]
        mom = np.sqrt(a[h1 + "_PX"]**2 + a[h1 + "_PY"]**2 + a[h1 + "_PZ"]**2)
        dau_pt_min = a[h1 + "_PT"].min()
        line = (f"  {mode:5s} n={t.num_entries:>6,}  "
                f"daughter pT min={dau_pt_min:7.1f}  "
                f"p min={mom.min()/1000:5.2f} GeV  "
                f"D0 pT min={a['D0_PT'].min():7.1f}")
        if ipk in a:
            line += f"  dau IPchi2 min={a[ipk].min():6.2f}"
        if "D0_FDCHI2_OWNPV" in a:
            line += f"  FDchi2 min={a['D0_FDCHI2_OWNPV'].min():6.2f}"
        if "D0_DIRA_OWNPV" in a:
            line += f"  DIRA min={a['D0_DIRA_OWNPV'].min():.6f}"
        print(line)
        print(f"        m(D0) range [{a['D0_MM'].min():.1f}, {a['D0_MM'].max():.1f}]")
    print()

print("=" * 72)
print("CLAIM B: is the v1/v2 rate difference statistically real?")
print("=" * 72)


def raw_counts(path):
    f = uproot.open(path)
    out = {}
    for mode in TREES:
        t = f[TREES[mode]]
        dst = t["Dst_2010_plus_ID"].array(library="np")
        out[mode] = {"n": len(dst), "plus": int(np.sum(dst == 413)),
                     "minus": int(np.sum(dst == -413))}
    return out


c1, c2 = raw_counts(V1), raw_counts(V2)
print(f"v1: KK={c1['KK']['n']} (D*+={c1['KK']['plus']}, D*-={c1['KK']['minus']})  "
      f"PiPi={c1['PiPi']['n']} (D*+={c1['PiPi']['plus']}, D*-={c1['PiPi']['minus']})  "
      f"KPi={c1['KPi']['n']}")
print(f"v2: KK={c2['KK']['n']} (D*+={c2['KK']['plus']}, D*-={c2['KK']['minus']})  "
      f"PiPi={c2['PiPi']['n']} (D*+={c2['PiPi']['plus']}, D*-={c2['PiPi']['minus']})  "
      f"KPi={c2['KPi']['n']}\n")


def ratio_with_error(n1, d1, n2, d2, cc_factor):
    r1, r2 = cc_factor * n1 / d1, n2 / d2
    rel = np.sqrt(1 / n1 + 1 / d1 + 1 / n2 + 1 / d2)
    r = r1 / r2
    return r, r * rel


for cc_label, cc in (("with CC x2 correction", 2.0),
                     ("no CC correction", 1.0)):
    print(f"--- {cc_label} ---")
    for mode in ("KK", "PiPi"):
        r, e = ratio_with_error(c1[mode]["n"], c1["KPi"]["n"],
                                c2[mode]["n"], c2["KPi"]["n"], cc)
        print(f"  {mode:5s}: v1/v2 = {r:5.2f} +- {e:4.2f}   "
              f"({abs(r - 1) / e:.1f} sigma from 1.0)")
    n1 = c1["KK"]["n"] + c1["PiPi"]["n"]
    n2 = c2["KK"]["n"] + c2["PiPi"]["n"]
    r, e = ratio_with_error(n1, c1["KPi"]["n"], n2, c2["KPi"]["n"], cc)
    print(f"  KK+pipi COMBINED: v1/v2 = {r:5.2f} +- {e:4.2f}   "
          f"({abs(r - 1) / e:.1f} sigma from 1.0)\n")

print("=" * 72)
print("SIDEBAND CHECK: are the m(D0) sidebands populated in production data?")
print("=" * 72)
SIG, SB = 20.0, (35.0, 75.0)
for mode in ("KK", "PiPi"):
    m = np.concatenate([uproot.open(f)[TREES[mode]]["D0_MM"].array(library="np")
                        for f, _pol in production_files("v2")])
    lo = int(np.sum((m > M_D0 - SB[1]) & (m < M_D0 - SB[0])))
    hi = int(np.sum((m > M_D0 + SB[0]) & (m < M_D0 + SB[1])))
    print(f"  {mode:5s}: lower sideband [{M_D0-SB[1]:.0f},{M_D0-SB[0]:.0f}] = {lo:,}   "
          f"upper [{M_D0+SB[0]:.0f},{M_D0+SB[1]:.0f}] = {hi:,}   "
          f"m(D0) min={m.min():.1f}")
    if min(lo, hi) == 0:
        print("        !! empty sideband")
