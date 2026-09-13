# scans the selection cuts and looks at the stat error on the raw asymmetry

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uproot

from charm_acp import selection as sel
from charm_acp.config import PLOTS_DIR, production_files
from charm_acp.kinematics import DM_DSTAR, M_D0

plt.style.use(mplhep.style.LHCb2)
DM_WIN = 0.6
SIG_WIN = 20.0
SB_WIN = (35.0, 75.0)
SCALE = (2 * SIG_WIN) / (2 * (SB_WIN[1] - SB_WIN[0]))


FILES = [p for p, _pol in production_files("v2")]
print(f"reading {len(FILES)} file(s)\n")


def load(mode):
    p = sel.PARTS[mode]
    want = ["D0_MM", "Dst_2010_plus_MM", "D0_IPCHI2_OWNPV", "D0_TAU",
            f"{p['h1']}_PIDK", f"{p['h2']}_PIDK", f"{p['soft']}_PIDK",
            f"{p['soft']}_ID", f"{p['soft']}_PT",
            f"{p['h1']}_TRACK_GhostProb", f"{p['h2']}_TRACK_GhostProb",
            f"{p['soft']}_TRACK_GhostProb"]
    chunks = [uproot.open(f)[sel.trees(mode, "v2")[0][0]].arrays(want, library="np")
              for f in FILES]
    return {k: np.concatenate([c[k] for c in chunks]) for k in want}


def counts(m, mask):
    mm = m[mask]
    d = np.abs(mm - M_D0)
    s = int(np.sum(d < SIG_WIN))
    b = int(np.sum((d > SB_WIN[0]) & (d < SB_WIN[1])))
    return s - b * SCALE, np.sqrt(s + b * SCALE**2)


def asymmetry(m, mask, tag):
    n_p, s_p = counts(m, mask & (tag > 0))
    n_m, s_m = counts(m, mask & (tag < 0))
    N = n_p + n_m
    if N <= 0:
        return np.nan, np.nan, N
    A = (n_p - n_m) / N
    sA = np.hypot(2 * n_m / N**2 * s_p, 2 * n_p / N**2 * s_m)
    return A, sA, N


data = {m: load(m) for m in ("KK", "PiPi", "KPi")}
for m, a in data.items():
    print(f"{m:5s}: {len(a['D0_MM']):>12,} candidates")

results = {}
for mode in ("KK", "PiPi", "KPi"):
    a = data[mode]
    p = sel.PARTS[mode]
    m, dm = a["D0_MM"], a["Dst_2010_plus_MM"] - a["D0_MM"]
    tag = np.sign(a[f"{p['soft']}_ID"])
    dm_tag = np.abs(dm - DM_DSTAR) < DM_WIN

    ghost = np.logical_and.reduce([a[f"{q}_TRACK_GhostProb"] < 0.3
                                   for q in (p["h1"], p["h2"], p["soft"])])
    soft_pid = a[f"{p['soft']}_PIDK"] < 0.0
    soft_pt = a[f"{p['soft']}_PT"] > 200.0
    prompt = (a["D0_IPCHI2_OWNPV"] < 9.0) & (a["D0_TAU"] > 0.0)
    tau_ok = a["D0_TAU"] > 0.0

    def hard_pid(thr):
        ms = []
        for part in (p["h1"], p["h2"]):
            if part.startswith("K"):
                ms.append(a[f"{part}_PIDK"] > thr)
            else:
                ms.append(a[f"{part}_PIDK"] < 0.0)
        return np.logical_and.reduce(ms)

    ladder = [
        ("baseline (selection.py)", hard_pid(5.0) & soft_pid & ghost & prompt & soft_pt),
        ("  drop soft-pion PID", hard_pid(5.0) & ghost & prompt & soft_pt),
        ("  + kaon PIDK>0 (line)", hard_pid(0.0) & ghost & prompt & soft_pt),
        ("  + drop ghost cut", hard_pid(0.0) & prompt & soft_pt),
        ("  + drop prompt cut", hard_pid(0.0) & tau_ok & soft_pt),
        ("loosest (dm tag only)", np.ones(len(m), bool)),
    ]

    print(f"\n===== {mode} =====")
    print(f"{'selection':26s} {'N(dm-tag)':>10s} {'signal':>16s} "
          f"{'A_raw':>18s}")
    results[mode] = {}
    for label, cut in ladder:
        full = cut & dm_tag
        n, e = counts(m, full)
        A, sA, _ = asymmetry(m, full, tag)
        shown = (f"{A:>+9.3f} +-{sA:<7.3f}" if mode == "KPi"
                 else f"{'blinded':>9s} +-{sA:<7.3f}")
        print(f"{label:26s} {int(full.sum()):>10,} {n:>9.0f} +-{e:<5.0f} "
              f"{shown}")
        results[mode][label] = (n, e, A, sA)

print("\n=== projected sigma(dACP) = quadrature sum of sigma(A_raw) ===")
print(f"{'selection':26s} {'sigma(KK)':>10s} {'sigma(pipi)':>12s} "
      f"{'sigma(dACP)':>12s}")
for label in results["KK"]:
    s_kk = results["KK"][label][3]
    s_pp = results["PiPi"][label][3]
    print(f"{label:26s} {s_kk:>10.3f} {s_pp:>12.3f} "
          f"{np.hypot(s_kk, s_pp):>12.3f}")

labels = list(results["KK"].keys())
x = np.arange(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
for mode, color in (("KK", "crimson"), ("PiPi", "navy")):
    sig = [results[mode][l][0] for l in labels]
    err = [results[mode][l][1] for l in labels]
    ax1.errorbar(x, sig, yerr=err, marker="o", ms=6, lw=1.5, capsize=4,
                 color=color, label=mode)
    ax2.plot(x, [results[mode][l][3] for l in labels], marker="s", ms=6,
             lw=1.5, color=color, label=fr"$\sigma(A_{{\rm raw}})$ {mode}")
ax2.plot(x, [np.hypot(results["KK"][l][3], results["PiPi"][l][3]) for l in labels],
         marker="^", ms=7, lw=2, color="darkgreen",
         label=r"$\sigma(\Delta A_{CP})$")
for ax in (ax1, ax2):
    ax.set_xticks(x)
    ax.set_xticklabels([l.strip() for l in labels], rotation=30, ha="right",
                       fontsize=9)
    ax.legend(fontsize=10)
ax1.set_ylabel("sideband-subtracted signal")
ax2.set_ylabel("statistical error")
fig.tight_layout()
out = PLOTS_DIR / "phase4_pid_tradeoff.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nwrote {out.name}")

best = min(np.hypot(results["KK"][l][3], results["PiPi"][l][3])
           for l in results["KK"])
LHCB = 15.4e-4
print(f"\n  best sigma(dACP) on {len(FILES)} files = {best:.3f}")
for name, lumi_fb in [("full 2016 MagDown", 0.85), ("2016 both polarities", 1.70),
                      ("all Run 2 (2015-18)", 6.0)]:
    have = sum(float(np.sum(uproot.open(f)["GetIntegratedLuminosity/LumiTuple"]
                            ["IntegratedLuminosity"].array(library="np")))
               for f in FILES) / 1000.0
    proj = best / np.sqrt(lumi_fb / have)
    print(f"  scaled to {name:22s} ({lumi_fb:.2f}/fb): {proj:.4f}  "
          f"= {proj / LHCB:.0f}x the published effect ({LHCB:.5f})")
