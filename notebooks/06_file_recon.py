# quick look at the data files, luminosity, run numbers and candidate counts

import numpy as np
import uproot

from charm_acp import selection as sel
from charm_acp.config import BR, production_files, production_spec

PROD_NAME, SPEC = production_spec()
LAYOUT = SPEC["layout"]
files = production_files()
print(f"{len(files)} production file(s) in manifest '{PROD_NAME}'\n")

tot_lumi = 0.0
tot_entries = {m: 0 for m in sel.LAYOUTS[LAYOUT]}

for path, pol in files:
    f = uproot.open(path)
    print(f"--- {path.name}  ({pol}, {path.stat().st_size / 1e9:.2f} GB) ---")

    lumi_arr = f["GetIntegratedLuminosity/LumiTuple"]["IntegratedLuminosity"].array(library="np")
    lumi = float(np.sum(lumi_arr))
    tot_lumi += lumi
    print(f"  integrated luminosity : {lumi:9.3f} /pb   ({len(lumi_arr)} lumi rows)")

    for mode in sel.LAYOUTS[LAYOUT]:
        n_mode = 0
        run_lo, run_hi, run_set = None, None, set()
        for tpath, d0 in sel.trees(mode, LAYOUT):
            t = f[tpath]
            n_mode += t.num_entries
            runs = t["runNumber"].array(library="np")
            if len(runs):
                run_lo = min(run_lo, runs.min()) if run_lo is not None else runs.min()
                run_hi = max(run_hi, runs.max()) if run_hi is not None else runs.max()
                run_set.update(np.unique(runs).tolist())
        tot_entries[mode] += n_mode
        print(f"  {mode:5s} entries {n_mode:>12,}   runs {run_lo}-{run_hi}  "
              f"({len(run_set)} unique)")
    print()

print("=== totals ===")
print(f"  luminosity : {tot_lumi:.3f} /pb  =  {tot_lumi/1000:.4f} /fb")
for mode, n in tot_entries.items():
    print(f"  {mode:5s} : {n:>12,} candidates")

print("\n=== raw-candidate ratios vs branching fractions ===")
for mode in ("KK", "PiPi"):
    obs = tot_entries[mode] / tot_entries["KPi"]
    exp = BR[mode] / BR["KPi"]
    print(f"  {mode:5s}: observed/KPi = {obs:.5f}   BR ratio = {exp:.5f}   "
          f"obs/exp = {obs/exp:.4f}")
