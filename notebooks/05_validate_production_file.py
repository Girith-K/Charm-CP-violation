# checks one ntuple file is ok, exits 1 if anything fails

import sys
from pathlib import Path

import numpy as np
import uproot

from charm_acp import kinematics as kin
from charm_acp import selection as sel
from charm_acp.config import BR, PRODUCTIONS, production_files
from charm_acp.kinematics import DM_DSTAR, M_D0, M_K

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    path = production_files()[0][0]

layout = polarity = None
for pname, spec in PRODUCTIONS.items():
    if path.name in spec["files"]:
        layout = spec["layout"]
        polarity = spec["files"][path.name]
if layout is None:
    raise SystemExit(f"{path.name} is not in config.PRODUCTIONS, add it to the "
                     "manifest before validating, nothing runs on unlisted files")

print(f"validating: {path.name}  ({path.stat().st_size / 1e9:.2f} GB, "
      f"layout {layout}, {polarity})\n")

f = uproot.open(path)
ok = True


def check(label, cond, detail=""):
    global ok
    if not cond:
        ok = False
    print(f"[{'PASS' if cond else 'FAIL'}] {label}  {detail}")


for mode in ("KK", "PiPi", "KPi"):
    hide = layout == "v3" and mode in ("KK", "PiPi")
    tot, all_live = 0, True
    for tpath, d0 in sel.trees(mode, layout):
        t = f[tpath]
        live = sum(1 for _, b in t.items() if getattr(b, "num_baskets", 0) > 0)
        tree_ok = live == len(t.keys()) and t.num_entries > 0
        tot += t.num_entries
        if hide:
            all_live = all_live and tree_ok
        else:
            check(f"{tpath.split('/')[0]}: tree present, all branches live",
                  tree_ok, f"{t.num_entries:,} entries, "
                           f"{live}/{len(t.keys())} live")
    if hide:
        check(f"{mode}: trees present, all branches live", all_live,
              f"{tot:,} entries combined (per-tree split not printed, blinding)")

lumi = f["GetIntegratedLuminosity/LumiTuple"]["IntegratedLuminosity"].array(
    library="np")
print(f"[INFO] integrated luminosity in this file: {np.sum(lumi):.3f} /pb\n")


def load(mode):
    parts = []
    for tpath, d0 in sel.trees(mode, layout):
        t = f[tpath]
        if t.num_entries == 0:
            continue
        parts.append(sel._normalize(
            t.arrays(sel.branches(mode, d0), library="np"), d0))
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


summary = {}
for mode in ["KK", "PiPi", "KPi"]:
    arr = load(mode)
    if arr is None:
        check(f"{mode}: has candidates", False, "all trees empty")
        continue
    p = sel.PARTS[mode]

    dst_id = arr["Dst_2010_plus_ID"]
    n_p, n_m = int(np.sum(dst_id == 413)), int(np.sum(dst_id == -413))
    if mode == "KPi":
        detail = f"D*+={n_p:,}  D*-={n_m:,}  (ratio {n_p/max(n_m,1):.3f})"
    else:
        detail = f"total {n_p + n_m:,} (charge split not printed, blinding)"
    check(f"{mode}: both D* charges", n_p > 0 and n_m > 0, detail)
    if layout == "v3" and mode in ("KK", "PiPi"):
        pure = all(
            len(np.unique(f[tpath]["Dst_2010_plus_ID"].array(library="np")))
            <= 1 for tpath, _ in sel.trees(mode, layout))
        check(f"{mode}: per-charge trees are charge pure", pure)

    soft_id = arr[f"{p['soft']}_ID"]
    check(f"{mode}: soft-pion charge == D* charge",
          bool(np.all(np.sign(soft_id) == np.sign(dst_id))))

    if mode == "KPi":
        frac = float(np.mean(arr["Kminus_PIDK"] > 5))
        check("KPi: kaons pass line cut PIDK>5", frac > 0.99, f"{frac:.2%}")

    dm = arr["Dst_2010_plus_MM"] - arr["D0_MM"]
    in_peak = float(np.mean(np.abs(dm - DM_DSTAR) < 1.0))
    check(f"{mode}: Delta-m peaks at {DM_DSTAR:.2f}", in_peak > 0.10,
          f"{in_peak:.1%} within +-1.0 MeV")

    mask, table = sel.cutflow(arr, mode)
    cats = sel.split_categories(arr, mask, mode)
    summary[mode] = {
        "n_raw": len(dm), "table": table,
        "n_d0": int(cats["D0"].sum()), "n_d0bar": int(cats["D0bar"].sum()),
        "dm": dm, "arr": arr,
    }

kk = summary.get("KK")
if kk and kk["n_raw"] > 0:
    mom_branches = [f"{h}_P{ax}" for h in ("Kplus", "Kminus") for ax in "XYZ"]
    parts = []
    for tpath, d0 in sel.trees("KK", layout):
        t = f[tpath]
        if t.num_entries == 0:
            continue
        parts.append(t.arrays(mom_branches, library="np"))
    a = ({k: np.concatenate([p[k] for p in parts]) for k in mom_branches}
         if parts else None)
    m_kk = kin.invariant_mass(a["Kplus_PX"], a["Kplus_PY"], a["Kplus_PZ"], M_K,
                              a["Kminus_PX"], a["Kminus_PY"], a["Kminus_PZ"],
                              M_K) if a is not None else None
    if m_kk is not None:
        dev = np.abs(m_kk - kk["arr"]["D0_MM"])
        med = float(np.median(dev))
        frac_off = float(np.mean(dev > 1.0))
        check("KK: our m(K+K-) reproduces stored D0_MM",
              med < 0.5 and frac_off < 0.01,
              f"median = {med * 1000:.0f} keV, frac>1 MeV = {frac_off:.2%} "
              f"(max {dev.max():.1f} MeV, refit tail)")

print("\n=== selection cutflow (this file, validation DEFAULT_CUTS: PIDK>5, "
      "soft PID on, fiducial off, not the analysis nominal) ===")
for mode, s in summary.items():
    print(f"\n{mode}:  {s['n_raw']:,} raw candidates")
    for name, n, eff in s["table"]:
        print(f"   {name:18s} {n:9,}  ({eff:6.1%})")
    if mode == "KPi":
        print(f"   -> tagged D0={s['n_d0']:,}  D0bar={s['n_d0bar']:,}")
    else:
        print(f"   -> tagged total={s['n_d0'] + s['n_d0bar']:,} "
              "(flavour split not printed, blinding)")

def _signal(mode):
    s = summary.get(mode)
    if not s or s["n_raw"] == 0:
        return 0.0, 0.0
    mm = s["arr"]["D0_MM"][np.abs(s["dm"] - DM_DSTAR) < 0.6]
    d = np.abs(mm - M_D0)
    sig = int(np.sum(d < 20.0))
    sb = int(np.sum((d > 35.0) & (d < 75.0)))
    return sig - sb * 0.5, np.sqrt(sig + sb * 0.25)


n_kpi, _ = _signal("KPi")
content_ok = True
print("\n[INFO] BR normalisation (dm-tagged, sideband-subtracted):")
for mode in ("KK", "PiPi"):
    n, e = _signal(mode)
    exp = n_kpi * BR[mode] / BR["KPi"]
    ratio = n / exp if exp > 0 else 0.0
    good = ratio > 0.1
    content_ok &= good
    print(f"[{'PASS' if good else 'WARN'}] {mode}: signal within 10x of BR "
          f"expectation  N = {n:,.0f}+-{e:.0f}, expected {exp:,.0f}, "
          f"obs/exp = {ratio:.4f}")
if not content_ok:
    print("[WARN] yield deficit vs branching ratios")

print("\n" + ("ALL CHECKS PASSED" if ok
              else "SOMETHING FAILED"))
sys.exit(0 if ok else 1)
