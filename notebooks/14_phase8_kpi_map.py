# Kpi raw asymmetry in bins of pT and eta

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from scipy.stats import chi2 as chi2_dist

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit
from charm_acp import kinematics as kin
from charm_acp import selection as sel
from charm_acp.config import (DM_FIT_HI, DM_FIT_LO, GRID_ETA_EDGES,
                              GRID_PT_EDGES, PLOTS_DIR, RESULTS_DIR,
                              production_files, production_spec)

plt.style.use(mplhep.style.LHCb2)

MODE = "KPi"
CHUNK = "512 MB"
LAYOUT = production_spec()[1]["layout"]
PT_EDGES = np.array(GRID_PT_EDGES)
ETA_EDGES = np.array(GRID_ETA_EDGES)
MIN_PER_CAT = 500
SIGNAL = "dgauss"

RESULTS_DIR.mkdir(exist_ok=True)

print("=== loading Kpi (nominal selection, same as the measurement) ===")
keep = {k: [] for k in ("dm", "tag", "pt", "eta", "run", "evt", "vchi2",
                        "pol")}
POLS = sorted({pol for _, pol in production_files()})
POL_CODE = {p_: i for i, p_ in enumerate(POLS)}
n_raw = 0
for path, pol in production_files():
    for arr in sel.iter_mode(path, MODE, LAYOUT, step_size=CHUNK):
        n_raw += len(arr["D0_MM"])
        mask, _ = sel.cutflow(arr, MODE, sel.NOMINAL_CUTS)
        if not mask.any():
            continue
        px, py, pz = (arr[f"Dst_2010_plus_P{c}"][mask] for c in "XYZ")
        keep["dm"].append((arr["Dst_2010_plus_MM"] - arr["D0_MM"])[mask])
        keep["tag"].append(np.where(
            arr[f"{sel.PARTS[MODE]['soft']}_ID"][mask] > 0, 1, -1).astype(np.int8))
        keep["pt"].append(kin.pt(px, py))
        keep["eta"].append(kin.eta(px, py, pz))
        keep["run"].append(arr["runNumber"][mask])
        keep["evt"].append(arr["eventNumber"][mask])
        keep["vchi2"].append((arr["D0_ENDVERTEX_CHI2"]
                              / np.maximum(arr["D0_ENDVERTEX_NDOF"], 1))[mask])
        keep["pol"].append(np.full(int(mask.sum()), POL_CODE[pol],
                                   dtype=np.int8))
d = {k: np.concatenate(v) for k, v in keep.items()}

pick = sel.single_candidate_pick(d["run"], d["evt"], d["vchi2"])
d = {k: v[pick] for k, v in d.items()}
print(f"  {n_raw:,} raw -> {len(d['dm']):,} selected, single-candidate")

in_win = (d["dm"] > DM_FIT_LO) & (d["dm"] < DM_FIT_HI)


def cell_index(pt_e, eta_e):
    n_eta = len(eta_e) - 1
    ipt = np.digitize(d["pt"], pt_e) - 1
    ieta = np.digitize(d["eta"], eta_e) - 1
    ok = (in_win & (ipt >= 0) & (ipt < len(pt_e) - 1)
          & (ieta >= 0) & (ieta < n_eta))
    return np.where(ok, (ipt * n_eta + ieta) * 2 + (d["tag"] > 0), -1)


def fit_pair(dm_d0, dm_d0bar):
    by_cat = {"D0": dm_d0, "D0bar": dm_d0bar}
    if min(len(v) for v in by_cat.values()) < MIN_PER_CAT:
        return None
    fits, combined = fit.fit_categories(by_cat, share_shape=True,
                                        signal=SIGNAL)
    if not all(f.valid for f in fits.values()):
        return None
    A, sA = asy.raw_asymmetry(fits["D0"].n_sig, fits["D0bar"].n_sig,
                              fits["D0"].n_sig_err, fits["D0bar"].n_sig_err)
    return {"A": float(A), "sA": float(sA),
            "n_sig": float(fits["D0"].n_sig + fits["D0bar"].n_sig),
            "sigma_dm": float(combined.sigma), "mu": float(combined.mu)}


print("\n=== integrated over all kinematics (polarity pooled) ===")
integ = fit_pair(d["dm"][in_win & (d["tag"] == 1)],
                 d["dm"][in_win & (d["tag"] == -1)])
if integ is None:
    raise SystemExit("integrated Kpi fit below thresholds, no map at these "
                     "statistics")
print(f"  A_raw(Kpi, pooled) = {integ['A']:+.5f} +- {integ['sA']:.5f}   "
      f"(N_sig = {integ['n_sig']:,.0f})")

npt, neta = len(PT_EDGES) - 1, len(ETA_EDGES) - 1
A_map = np.full((npt, neta), np.nan)
E_map = np.full((npt, neta), np.nan)
N_map = np.full((npt, neta), np.nan)
S_map = np.full((npt, neta), np.nan)

print(f"\n=== (pT, eta) map: {npt} x {neta} bins ===")
print(f"{'pT [MeV]':>16s} {'eta':>12s} {'N_sig':>12s} {'sigma_dm':>9s} {'A_raw':>20s}")
cells = []
grp = cell_index(PT_EDGES, ETA_EDGES)
for i in range(npt):
    for j in range(neta):
        c = (i * neta + j) * 2
        r = fit_pair(d["dm"][grp == c + 1], d["dm"][grp == c])
        if r is None:
            print(f"{PT_EDGES[i]:7.0f}-{PT_EDGES[i+1]:<8.0f} "
                  f"{ETA_EDGES[j]:5.2f}-{ETA_EDGES[j+1]:<6.2f}  "
                  f"{'too few':>12s}")
            continue
        A_map[i, j], E_map[i, j] = r["A"], r["sA"]
        N_map[i, j], S_map[i, j] = r["n_sig"], r["sigma_dm"]
        cells.append(r)
        print(f"{PT_EDGES[i]:7.0f}-{PT_EDGES[i+1]:<8.0f} "
              f"{ETA_EDGES[j]:5.2f}-{ETA_EDGES[j+1]:<6.2f} "
              f"{r['n_sig']:>12,.0f} {r['sigma_dm']:>9.3f} "
              f"{r['A']:>+11.5f} +-{r['sA']:.5f}")

good = np.isfinite(A_map) & np.isfinite(E_map) & (E_map > 0)
vals, errs = A_map[good], E_map[good]
wavg, wavg_err = asy.weighted_average(vals, errs)
chi2 = float(np.sum(((vals - wavg) / errs) ** 2))
ndf = int(good.sum()) - 1

print("\n=== is A_raw(Kpi) flat across (pT, eta)? ===")
print(f"  bins used                : {int(good.sum())}")
print(f"  inverse-variance average : {wavg:+.5f} +- {wavg_err:.5f}")
print(f"  integrated (pooled fit)  : {integ['A']:+.5f} +- {integ['sA']:.5f}")
print(f"  spread of bin values     : {np.min(vals):+.5f} to {np.max(vals):+.5f}  "
      f"(range {np.max(vals) - np.min(vals):.5f})")
print(f"  chi2/ndf vs a constant   : {chi2:.1f}/{ndf} = {chi2/ndf:.1f}")
pval = float(chi2_dist.sf(chi2, ndf))
print(f"  p-value                  : {pval:.2e}")
if pval < 1e-3:
    print("  => not flat (p < 1e-3)")
elif pval < 0.05:
    print("  => not flat (p < 0.05)")
else:
    print("  => consistent with flat")


print("\n=== polarity cancellation, matched in (pT, eta) ===")

pol_mask = {p_: (d["pol"] == POL_CODE[p_]) for p_ in POLS}
integ_by_pol = {}
for p_ in POLS:
    m_ = in_win & pol_mask[p_]
    r_ = fit_pair(d["dm"][m_ & (d["tag"] == 1)], d["dm"][m_ & (d["tag"] == -1)])
    integ_by_pol[p_] = r_
    if r_:
        print(f"  integrated {p_:8s}: A_raw = {r_['A']:+.5f} +- {r_['sA']:.5f}")

polarity_residual = None
if all(integ_by_pol.values()) and len(POLS) == 2:
    (p1, p2) = POLS
    a_int, s_int = asy.polarity_average(
        integ_by_pol[p1]["A"], integ_by_pol[p1]["sA"],
        integ_by_pol[p2]["A"], integ_by_pol[p2]["sA"])
    a_odd = 0.5 * (integ_by_pol[p1]["A"] - integ_by_pol[p2]["A"])
    print(f"  A_even (integrated, equal weight) = {a_int:+.5f} +- {s_int:.5f}")
    print(f"  A_odd  (half-difference)          = {a_odd:+.5f}")

    matched_vals, matched_errs, n_cells = [], [], 0
    for i in range(npt):
        for j in range(neta):
            per_pol = []
            for p_ in POLS:
                cellm = (in_win & pol_mask[p_]
                         & (np.digitize(d["pt"], PT_EDGES) - 1 == i)
                         & (np.digitize(d["eta"], ETA_EDGES) - 1 == j))
                r_ = fit_pair(d["dm"][cellm & (d["tag"] == 1)],
                              d["dm"][cellm & (d["tag"] == -1)])
                if r_ is None:
                    per_pol = []
                    break
                per_pol.append((r_["A"], r_["sA"]))
            if len(per_pol) != 2:
                continue
            (a1, s1), (a2, s2) = per_pol
            av, ev = asy.polarity_average(a1, s1, a2, s2)
            matched_vals.append(av)
            matched_errs.append(ev)
            n_cells += 1

    if n_cells >= 2:
        a_match, s_match = asy.weighted_average(np.array(matched_vals),
                                                np.array(matched_errs))
        shift = float(a_match - a_int)
        print(f"  A_even (matched cell by cell)     = {a_match:+.5f} "
              f"+- {s_match:.5f}   [{n_cells} cells]")
        print(f"  residual shift                    = {shift:+.5f}  "
              f"({abs(shift) / s_int:.2f} x sigma_stat)")
        polarity_residual = {
            "A_even_integrated": float(a_int),
            "A_even_integrated_err": float(s_int),
            "A_odd_integrated": float(a_odd),
            "A_even_cell_matched": float(a_match),
            "A_even_cell_matched_err": float(s_match),
            "shift": shift,
            "n_cells": n_cells,
            "per_polarity_integrated": {
                p_: {"A": integ_by_pol[p_]["A"], "sA": integ_by_pol[p_]["sA"]}
                for p_ in POLS},
        }
    else:
        print("  too few cells fit in BOTH polarities, residual not measurable")
else:
    print("  needs exactly two polarities with a valid integrated fit")

print("\n=== binning stability ===")
print(f"  {'grid':>16s} {'bins':>6s} {'weighted average':>26s}")
print(f"  {'5x5 (nominal)':>16s} {int(good.sum()):>6d} "
      f"{wavg:>+15.5f} +-{wavg_err:.5f}")
for label, pt_e, eta_e in [
        ("3x3 coarse", np.array([2000., 4500., 8000., 30000.]),
         np.array([2.0, 3.0, 4.0, 5.0])),
        ("pT only", PT_EDGES, np.array([2.0, 5.0])),
        ("eta only", np.array([2000., 30000.]), ETA_EDGES)]:
    vs, es = [], []
    g = cell_index(pt_e, eta_e)
    n_eta = len(eta_e) - 1
    for i in range(len(pt_e) - 1):
        for j in range(n_eta):
            c = (i * n_eta + j) * 2
            r = fit_pair(d["dm"][g == c + 1], d["dm"][g == c])
            if r:
                vs.append(r["A"])
                es.append(r["sA"])
    if vs:
        a, e = asy.weighted_average(np.array(vs), np.array(es))
        print(f"  {label:>16s} {len(vs):>6d} {a:>+15.5f} +-{e:.5f}")

fig, (ax, axp) = plt.subplots(1, 2, figsize=(15, 5.5))

im = ax.imshow(A_map * 100, origin="lower", aspect="auto", cmap="RdBu_r",
               extent=[0, neta, 0, npt],
               vmin=np.nanmin(A_map) * 100, vmax=np.nanmax(A_map) * 100)
ax.set_xticks(np.arange(neta) + 0.5)
ax.set_xticklabels([f"{ETA_EDGES[j]:.2f}-\n{ETA_EDGES[j+1]:.2f}" for j in range(neta)],
                   fontsize=9)
ax.set_yticks(np.arange(npt) + 0.5)
ax.set_yticklabels([f"{PT_EDGES[i]/1000:g}-{PT_EDGES[i+1]/1000:g}" for i in range(npt)],
                   fontsize=9)
ax.set_xlabel(r"$\eta(D^{*})$")
ax.set_ylabel(r"$p_T(D^{*})$ [GeV]")
for i in range(npt):
    for j in range(neta):
        if np.isfinite(A_map[i, j]):
            ax.text(j + 0.5, i + 0.5, f"{A_map[i,j]*100:+.2f}",
                    ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label=r"$A_{\rm raw}$ [%]")

for j in range(neta):
    m = np.isfinite(A_map[:, j])
    if m.sum() < 2:
        continue
    centres = 0.5 * (PT_EDGES[:-1] + PT_EDGES[1:])[m] / 1000
    axp.errorbar(centres, A_map[m, j] * 100, yerr=E_map[m, j] * 100,
                 marker="o", ms=4, lw=1.2, capsize=3,
                 label=fr"$\eta$ {ETA_EDGES[j]:.2f}-{ETA_EDGES[j+1]:.2f}")
axp.axhline(integ["A"] * 100, color="k", ls="--", lw=1,
            label="pooled (yield-weighted)")
axp.set_xscale("log")
axp.set_xlabel(r"$p_T(D^{*})$ [GeV]")
axp.set_ylabel(r"$A_{\rm raw}(K\pi)$ [%]")
axp.legend(fontsize=8, ncol=2)
fig.tight_layout()
out = PLOTS_DIR / "phase8_kpi_asymmetry_map.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nwrote {out.name}")

payload = {
    "integrated": integ,
    "pt_edges": PT_EDGES.tolist(), "eta_edges": ETA_EDGES.tolist(),
    "A": np.where(np.isfinite(A_map), A_map, None).tolist(),
    "A_err": np.where(np.isfinite(E_map), E_map, None).tolist(),
    "n_sig": np.where(np.isfinite(N_map), N_map, None).tolist(),
    "weighted_average": {"value": float(wavg), "error": float(wavg_err)},
    "flatness": {"chi2": chi2, "ndf": ndf, "p_value": pval},
    "polarity_kinematic_residual": polarity_residual,
}
p = RESULTS_DIR / "phase8_kpi_map.json"
p.write_text(json.dumps(payload, indent=2, default=str))
print(f"wrote {p.name}")
