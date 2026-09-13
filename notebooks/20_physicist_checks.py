# extra checks, fiducial map, D0 vs D0bar shapes, duplicate events, secondary charm

from __future__ import annotations

import dataclasses
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uproot
from scipy.stats import chi2 as _chi2

from charm_acp import fitting as fit
from charm_acp import selection as sel
from charm_acp.config import (DM_FIT_HI, DM_FIT_LO, PLOTS_DIR, RESULTS_DIR,
                              production_files, production_spec)
from charm_acp.kinematics import DM_DSTAR, M_D0

plt.style.use(mplhep.style.LHCb2)
RESULTS_DIR.mkdir(exist_ok=True)
report = {}

SIGNAL = "dgauss"
CUTS = sel.NOMINAL_CUTS
CUTS_NOFID = dataclasses.replace(sel.NOMINAL_CUTS, soft_fiducial=False)
CUTS_NOIP = dataclasses.replace(sel.NOMINAL_CUTS, d0_ipchi2_max=np.inf,
                                d0_mass_win=np.inf)
LAYOUT = production_spec()[1]["layout"]

FILES = production_files()


def load(mode, extra=()):
    chunks = []
    want = None
    for path, pol in FILES:
        for tname, d0 in sel.trees(mode, LAYOUT):
            cols = list(dict.fromkeys(sel.branches(mode, d0) + list(extra)))
            for arr in uproot.iterate({str(path): tname}, cols, library="np",
                                      step_size="512 MB"):
                arr = sel._normalize(arr, d0)
                want = list(arr.keys())
                chunks.append(arr)
    if not chunks:
        return None
    return {k: np.concatenate([c[k] for c in chunks]) for k in want}


print("=== C. duplicate (run, event) pairs across files ===")
seen = {}
dup_total = 0
for path, pol in FILES:
    tname, d0 = sel.trees("KPi", LAYOUT)[0]
    t = uproot.open(path)[tname]
    run = t["runNumber"].array(library="np").astype(np.int64)
    evt = t["eventNumber"].array(library="np").astype(np.int64)
    if evt.size and int(evt.max()) >= 10**10:
        raise SystemExit("eventNumber >= 1e10 would collide the pair key")
    keys = np.unique(run * np.int64(10**10) + evt)
    seen[path.name] = set(keys.tolist())
for i, (n1, s1) in enumerate(seen.items()):
    for n2, s2 in list(seen.items())[i + 1:]:
        overlap = len(s1 & s2)
        dup_total += overlap
        print(f"  {n1} ∩ {n2}: {overlap} shared events")
verdict_c = "PASS" if dup_total == 0 else f"FAIL, {dup_total} shared events"
print(f"  -> {verdict_c}")
report["duplicates_across_files"] = {"shared_events": dup_total,
                                     "pass": dup_total == 0}

print("\n=== B. flavour-category shape compatibility (Kpi, free fits) ===")
kpi = load("KPi")
mask, _ = sel.cutflow(kpi, "KPi", CUTS)
dm = kpi["Dst_2010_plus_MM"] - kpi["D0_MM"]
tag = sel.flavour_tag(kpi)
win = mask & (dm > DM_FIT_LO) & (dm < DM_FIT_HI)

shape_out = {}
for name, sign in (("D0", 1), ("D0bar", -1)):
    r = fit.fit_dm(dm[win & (tag == sign)], signal=SIGNAL)
    shape_out[name] = r
    print(f"  {name:6s}: mu = {r.mu:.4f}  sigma = {r.sigma:.4f}  "
          f"N = {r.n_sig:,.0f}  valid={r.valid}")

r0, r1 = shape_out["D0"], shape_out["D0bar"]
e_mu = np.hypot(r0.sigma / np.sqrt(r0.n_sig), r1.sigma / np.sqrt(r1.n_sig))
dmu = r0.mu - r1.mu
print(f"  mu(D0) - mu(D0bar) = {dmu * 1000:+.1f} keV  (~{abs(dmu) / e_mu:.1f} "
      f"sigma_stat)   sigma ratio = {r0.sigma / r1.sigma:.4f}")
ok_b = abs(dmu) < 0.010 and abs(r0.sigma / r1.sigma - 1) < 0.02
print(f"  -> {'PASS, widths agree' if ok_b else 'INVESTIGATE, charge-dependent shape'}")
report["shape_compatibility"] = {
    "dmu_keV": dmu * 1000, "sigma_ratio": r0.sigma / r1.sigma,
    "signal_model": SIGNAL, "pass": bool(ok_b)}

print("\n=== A. soft-pion (pz, px) fiducial map (Kpi control) ===")
print("  fiducial cut released")
soft = sel.PARTS["KPi"]["soft"]
mask_nf, _ = sel.cutflow(kpi, "KPi", CUTS_NOFID)
win_nf = mask_nf & (dm > DM_FIT_LO) & (dm < DM_FIT_HI)
px = kpi[f"{soft}_PX"][win_nf] / 1000.0
pz = kpi[f"{soft}_PZ"][win_nf] / 1000.0
tg = tag[win_nf]
in_peak = np.abs(dm[win_nf] - DM_DSTAR) < 1.0

PZ_E = np.array([2.0, 4.0, 6.0, 9.0, 14.0, 40.0])
PX_E = np.linspace(-2.0, 2.0, 9)
A = np.full((len(PZ_E) - 1, len(PX_E) - 1), np.nan)
E = np.full_like(A, np.nan)
for i in range(len(PZ_E) - 1):
    for j in range(len(PX_E) - 1):
        inb = (in_peak & (pz >= PZ_E[i]) & (pz < PZ_E[i + 1])
               & (px >= PX_E[j]) & (px < PX_E[j + 1]))
        np_, nm = int(np.sum(inb & (tg == 1))), int(np.sum(inb & (tg == -1)))
        if np_ + nm < 2000:
            continue
        A[i, j] = (np_ - nm) / (np_ + nm)
        E[i, j] = np.sqrt((1 - A[i, j] ** 2) / (np_ + nm))

good = np.isfinite(A)
mean_A = np.nansum(A / E**2) / np.nansum(1 / E[good]**2)
extreme = good & (np.abs(A - mean_A) > np.maximum(3 * E, 0.02))
print(f"  populated bins: {int(good.sum())}   "
      f"flagged (|A - <A>| > max(3sigma, 2%)): {int(extreme.sum())}")
for i, j in zip(*np.where(extreme)):
    print(f"    pz [{PZ_E[i]:.0f},{PZ_E[i+1]:.0f}) GeV, "
          f"px [{PX_E[j]:+.1f},{PX_E[j+1]:+.1f}) GeV:  "
          f"A = {A[i,j]:+.3f} +- {E[i,j]:.3f}")
frac_flagged = float(np.sum(extreme * 1.0) / max(good.sum(), 1))
print(f"  -> {'no extreme regions at current sensitivity' if extreme.sum() == 0 else 'candidate fiducial exclusions found'}")

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(A * 100, origin="lower", aspect="auto", cmap="RdBu_r",
               extent=[PX_E[0], PX_E[-1], 0, len(PZ_E) - 1])
ax.set_yticks(np.arange(len(PZ_E) - 1) + 0.5)
ax.set_yticklabels([f"{PZ_E[i]:.0f}-{PZ_E[i+1]:.0f}" for i in range(len(PZ_E) - 1)])
ax.set_xlabel(r"soft pion $p_x$ [GeV]")
ax.set_ylabel(r"soft pion $p_z$ [GeV]")
fig.colorbar(im, ax=ax, label=r"$A_{\rm raw}(K\pi)$ [%]")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "fiducial_softpion_map.png", dpi=150, bbox_inches="tight")
print("  wrote fiducial_softpion_map.png")
report["fiducial_map"] = {"bins": int(good.sum()), "flagged": int(extreme.sum())}

print("\n=== D. secondary (D0-from-b) fraction, prompt cut released ===")
report["secondary_fraction"] = {}
for mode in ("KK", "PiPi", "KPi"):
    arr = kpi if mode == "KPi" else load(mode)
    if arr is None:
        print(f"  {mode:5s}: no candidates, skipped")
        continue
    mask_m, _ = sel.cutflow(arr, mode, CUTS_NOIP)
    dm_m = arr["Dst_2010_plus_MM"] - arr["D0_MM"]
    m = arr["D0_MM"]
    ip = arr["D0_IPCHI2_OWNPV"]
    dm_tag = mask_m & (np.abs(dm_m - DM_DSTAR) < 0.6)

    def sb_count(subset):
        d = np.abs(m[subset] - M_D0)
        sig = int(np.sum(d < 20.0))
        sb = int(np.sum((d > 35.0) & (d < 75.0)))
        return sig - 0.5 * sb, np.sqrt(sig + 0.25 * sb), sig, sb

    n_all, e_all, raw_all, _ = sb_count(dm_tag)
    n_sec, e_sec, raw_sec, raw_sec_sb = sb_count(dm_tag & (ip > 9.0))
    if n_all < 50 or e_all <= 0:
        print(f"  {mode:5s}: signal {n_all:>11,.0f}  too few to quote a "
              "secondary fraction, skipped")
        report["secondary_fraction"][mode] = {"f": None, "err": None,
                                              "n_signal": float(n_all)}
        continue
    if n_sec <= 0:
        ul = 0.5 * _chi2.ppf(0.90, 2 * (raw_sec + 1)) - 0.5 * raw_sec_sb
        f_ul = max(ul, 0.0) / n_all
        print(f"  {mode:5s}: signal {n_all:>11,.0f}  with IPchi2>9: "
              f"{n_sec:>9,.0f}   -> consistent with zero, "
              f"f < {f_ul:.1%} (90% CL)")
        report["secondary_fraction"][mode] = {
            "f": None, "err": None, "f_upper_limit_90CL": float(f_ul),
            "n_signal": float(n_all), "n_secondary": float(n_sec)}
        continue
    f = n_sec / n_all
    ef = np.sqrt(e_sec ** 2 + (f * e_all) ** 2) / n_all
    print(f"  {mode:5s}: signal {n_all:>11,.0f}  with IPchi2>9: {n_sec:>9,.0f}"
          f"   -> secondary-like fraction f = {f:.1%} +- {ef:.1%}")
    report["secondary_fraction"][mode] = {"f": float(f), "err": float(ef),
                                          "n_signal": float(n_all)}


(RESULTS_DIR / "physicist_checks.json").write_text(
    json.dumps(report, indent=2, default=float))
print("\nwrote physicist_checks.json")
