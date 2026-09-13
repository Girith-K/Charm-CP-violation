# compares the v1 and v2 line yields, scaled to Kpi

import numpy as np
import uproot

from charm_acp.config import BR, DATA_DIR
from charm_acp.kinematics import DM_DSTAR

SOFT = {("v1", "KK"): "piplus", ("v1", "PiPi"): "piplus_0", ("v1", "KPi"): "piplus_0",
        ("v2", "KK"): "piplus", ("v2", "PiPi"): "piplus_0", ("v2", "KPi"): "piplus_0"}

FILES = {"v1": DATA_DIR / "archive" / "testprod_job0.root",
         "v2": DATA_DIR / "archive" / "testprod_v2_job0.root"}
TREES = {"KK": "DstD02KK/DecayTree", "PiPi": "DstD02PiPi/DecayTree",
         "KPi": "DstD02KPi/DecayTree"}


def measure(path, version, mode):
    t = uproot.open(path)[TREES[mode]]
    a = t.arrays(["Dst_2010_plus_ID", "Dst_2010_plus_MM", "D0_MM"], library="np")
    dst = a["Dst_2010_plus_ID"]
    dm = a["Dst_2010_plus_MM"] - a["D0_MM"]
    n_p, n_m = int(np.sum(dst == 413)), int(np.sum(dst == -413))
    in_peak = int(np.sum(np.abs(dm - DM_DSTAR) < 1.0))
    return {"n": len(dm), "dstar_plus": n_p, "dstar_minus": n_m, "peak": in_peak}


out = {}
for version, path in FILES.items():
    if not path.exists():
        print(f"missing {path.name} - skipping {version}")
        continue
    out[version] = {m: measure(path, version, m) for m in TREES}
    print(f"=== {version}: {path.name} ===")
    for mode, r in out[version].items():
        print(f"  {mode:5s} {r['n']:>8,} candidates   "
              f"D*+={r['dstar_plus']:>7,}  D*-={r['dstar_minus']:>7,}   "
              f"dm-peak={r['peak']:>7,}")
    print()

print("=== yield relative to the Kpi control (same line in both versions) ===")
print(f"{'':6s} {'mode':6s} {'ratio to Kpi':>14s} {'CC-corrected':>14s} "
      f"{'BR ratio':>10s} {'obs/exp':>9s}")
summary = {}
for version, r in out.items():
    n_kpi = r["KPi"]["n"]
    for mode in ("KK", "PiPi"):
        raw = r[mode]["n"] / n_kpi
        only_one_charge = r[mode]["dstar_minus"] == 0 or r[mode]["dstar_plus"] == 0
        corrected = raw * (2.0 if only_one_charge else 1.0)
        exp = BR[mode] / BR["KPi"]
        summary[(version, mode)] = corrected / exp
        print(f"{version:6s} {mode:6s} {raw:>14.5f} {corrected:>14.5f} "
              f"{exp:>10.5f} {corrected / exp:>9.4f}")

print("\n=== v1/v2 gain per unit Kpi ===")
for mode in ("KK", "PiPi"):
    if ("v1", mode) in summary and ("v2", mode) in summary:
        gain = summary[("v1", mode)] / summary[("v2", mode)]
        print(f"  {mode:5s}: DstarCPForPromptCharm (v1) yields {gain:5.1f}x "
              f"the D2hh line (v2), per unit Kpi")
        print(f"         -> sigma(A_raw) would improve by sqrt({gain:.1f}) = "
              f"{np.sqrt(gain):.2f}x if we went back to it")
