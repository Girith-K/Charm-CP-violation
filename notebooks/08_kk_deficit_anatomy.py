# counts KK and pipi signal by hand

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uproot

from charm_acp import selection as sel
from charm_acp.config import BR, PLOTS_DIR, production_files
from charm_acp.kinematics import DM_DSTAR, M_D0

plt.style.use(mplhep.style.LHCb2)


DM_WIN = 0.6
DM_SB = (2.0, 5.0)

SIG_WIN = 20.0
SB_WIN = (35.0, 75.0)

files = [p for p, _pol in production_files("v2")]
print(f"reading {len(files)} production file(s)\n")

rows = {}
for mode in ["KK", "PiPi", "KPi"]:
    p = sel.PARTS[mode]
    want = ["D0_MM", "Dst_2010_plus_MM", "D0_IPCHI2_OWNPV", "D0_TAU",
            f"{p['h1']}_PIDK", f"{p['h2']}_PIDK", f"{p['soft']}_PIDK",
            f"{p['h1']}_TRACK_GhostProb", f"{p['h2']}_TRACK_GhostProb",
            f"{p['soft']}_TRACK_GhostProb"]
    chunks = []
    for path in files:
        t = uproot.open(path)[sel.trees(mode, "v2")[0][0]]
        chunks.append(t.arrays(want, library="np"))
    a = {k: np.concatenate([c[k] for c in chunks]) for k in want}

    dm = a["Dst_2010_plus_MM"] - a["D0_MM"]
    m = a["D0_MM"]

    ghost = np.logical_and.reduce([
        a[f"{part}_TRACK_GhostProb"] < 0.3 for part in (p["h1"], p["h2"], p["soft"])])
    prompt = (a["D0_IPCHI2_OWNPV"] < 9.0) & (a["D0_TAU"] > 0.0)

    rows[mode] = dict(m=m, dm=dm, ghost=ghost, prompt=prompt, n=len(m))
    print(f"{mode:5s}: {len(m):>10,} candidates loaded")


def sideband_count(m, in_dm):
    mm = m[in_dm]
    sig = np.sum(np.abs(mm - M_D0) < SIG_WIN)
    d = np.abs(mm - M_D0)
    sb = np.sum((d > SB_WIN[0]) & (d < SB_WIN[1]))
    scale = (2 * SIG_WIN) / (2 * (SB_WIN[1] - SB_WIN[0]))
    bkg = sb * scale
    n = sig - bkg
    err = np.sqrt(sig + sb * scale**2)
    return n, err, sig, bkg


print("\n=== D0-mass peak of dm-tagged candidates (sideband subtracted) ===")
print(f"{'mode':6s} {'cuts':22s} {'N(dm-tag)':>10s} {'S+B':>9s} {'B':>9s} "
      f"{'signal':>12s} {'purity':>8s}")
res = {}
for mode in ["KPi", "KK", "PiPi"]:
    r = rows[mode]
    for label, extra in [("dm-tag + ghost", r["ghost"]),
                         ("  + prompt D0", r["ghost"] & r["prompt"])]:
        in_dm = (np.abs(r["dm"] - DM_DSTAR) < DM_WIN) & extra
        n, err, s, b = sideband_count(r["m"], in_dm)
        pur = n / s if s else 0.0
        print(f"{mode:6s} {label:22s} {int(in_dm.sum()):>10,} {s:>9,} {b:>9.0f} "
              f"{n:>8.0f}+-{err:<4.0f} {pur:>7.1%}")
        res[(mode, label)] = (n, err)
    print()

print("=== branching-ratio normalisation (prompt-cut yields) ===")
n_kpi = res[("KPi", "  + prompt D0")][0]
print(f"  N(Kpi)  = {n_kpi:,.0f}   <- normalisation")
for mode in ["KK", "PiPi"]:
    n, err = res[(mode, "  + prompt D0")]
    exp = n_kpi * BR[mode] / BR["KPi"]
    print(f"  N({mode:4s}) = {n:>10,.0f} +- {err:<6.0f}  expected {exp:>12,.0f}  "
          f"obs/exp = {n/exp:.4f}   (1/{exp/max(n,1):.0f})")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, mode in zip(axes, ["KPi", "KK", "PiPi"]):
    r = rows[mode]
    in_dm = (np.abs(r["dm"] - DM_DSTAR) < DM_WIN) & r["ghost"] & r["prompt"]
    ax.hist(r["m"][in_dm], bins=80, range=(1790, 1940), histtype="step",
            lw=1.6, color="navy")
    ax.axvline(M_D0, color="crimson", ls="--", lw=1)
    ax.set_xlabel(r"$m(h^+h^-)$ [MeV]")
    ax.set_ylabel("candidates / 1.9 MeV")
    ax.set_title(f"{mode}: $\\Delta m$-tagged + prompt  "
                 f"({int(in_dm.sum()):,} cand.)", fontsize=11)
fig.tight_layout()
out = PLOTS_DIR / "kk_deficit_d0mass_dmtagged.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nwrote {out.name}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, mode in zip(axes, ["KPi", "KK", "PiPi"]):
    r = rows[mode]
    sel_m = (np.abs(r["m"] - M_D0) < SIG_WIN) & r["ghost"] & r["prompt"]
    ax.hist(r["dm"][sel_m], bins=90, range=(140, 158), histtype="step",
            lw=1.6, color="darkgreen")
    ax.axvline(DM_DSTAR, color="crimson", ls="--", lw=1)
    ax.set_xlabel(r"$\Delta m$ [MeV]")
    ax.set_ylabel("candidates / 0.2 MeV")
    ax.set_title(f"{mode}: $m(D^0)$ window + prompt  "
                 f"({int(sel_m.sum()):,} cand.)", fontsize=11)
fig.tight_layout()
out = PLOTS_DIR / "kk_deficit_dm_spectra.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out.name}")
