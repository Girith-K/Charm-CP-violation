# checks the stored D0 mass matches the one from the daughter momenta, v2 files

import numpy as np
import uproot

from charm_acp import kinematics as kin
from charm_acp.config import production_files

for path, _pol in production_files("v2"):
    t = uproot.open(path)["DstD02KK/DecayTree"]
    a = t.arrays(["Kplus_PX", "Kplus_PY", "Kplus_PZ",
                  "Kminus_PX", "Kminus_PY", "Kminus_PZ", "D0_MM",
                  "runNumber", "eventNumber"], library="np")
    m = kin.invariant_mass(a["Kplus_PX"], a["Kplus_PY"], a["Kplus_PZ"], kin.M_K,
                           a["Kminus_PX"], a["Kminus_PY"], a["Kminus_PZ"], kin.M_K)
    d = np.abs(m - a["D0_MM"])
    n = len(d)
    print(f"{path.name}:  n={n:,}")
    print(f"  median |diff| = {np.median(d)*1000:.1f} keV   "
          f"99.9%: {np.percentile(d, 99.9):.4f} MeV   max: {d.max():.3f} MeV")
    for thr in (0.1, 1.0, 5.0):
        print(f"  candidates with |diff| > {thr:4.1f} MeV : {int(np.sum(d > thr))}")
    worst = np.argsort(d)[-3:][::-1]
    for i in worst:
        print(f"    worst: run {a['runNumber'][i]} evt {a['eventNumber'][i]}  "
              f"D0_MM={a['D0_MM'][i]:.2f}  ours={m[i]:.2f}  diff={d[i]:.3f}")
    print()
