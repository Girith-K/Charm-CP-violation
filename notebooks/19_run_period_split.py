# checks the result does not drift with run number

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as chi2_dist

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit
from charm_acp import selection as sel
from charm_acp.config import (DM_FIT_HI, DM_FIT_LO, production_files,
                              production_spec)

CUTS = sel.NOMINAL_CUTS
LAYOUT = production_spec()[1]["layout"]
SIGNAL = "dgauss"
N_BLOCKS = 3


def mode_window(mode):
    return 155.0 if (LAYOUT == "v3" and mode != "KPi") else DM_FIT_HI


def load(mode):
    acc = {}
    for path, pol in production_files():
        a = acc.setdefault(pol, {k: [] for k in ("dm", "tag", "run", "evt",
                                                 "v")})
        for arr in sel.iter_mode(path, mode, LAYOUT):
            mask, _ = sel.cutflow(arr, mode, CUTS)
            a["dm"].append((arr["Dst_2010_plus_MM"] - arr["D0_MM"])[mask])
            a["tag"].append(sel.flavour_tag(arr)[mask])
            a["run"].append(arr["runNumber"][mask])
            a["evt"].append(arr["eventNumber"][mask])
            a["v"].append((arr["D0_ENDVERTEX_CHI2"]
                           / np.maximum(arr["D0_ENDVERTEX_NDOF"], 1))[mask])
    hi = mode_window(mode)
    out = {}
    for pol, a in acc.items():
        dm = np.concatenate(a["dm"])
        tag = np.concatenate(a["tag"])
        run = np.concatenate(a["run"])
        pick = sel.single_candidate_pick(run, np.concatenate(a["evt"]),
                                         np.concatenate(a["v"]))
        dm, tag, run = dm[pick], tag[pick], run[pick]
        win = (dm > DM_FIT_LO) & (dm < hi)
        out[pol] = (dm[win], tag[win], run[win])
    return out


def a_raw(dm, tag, hi):
    by_cat = {"D0": dm[tag > 0], "D0bar": dm[tag < 0]}
    if min(len(v) for v in by_cat.values()) < 30:
        return None
    nbins = int(round((hi - DM_FIT_LO) / 0.25))
    fits, _ = fit.fit_categories(by_cat, share_shape=True, hi=hi, nbins=nbins,
                                 signal=SIGNAL)
    if not all(r.valid and r.n_sig_err > 0 for r in fits.values()):
        return None
    if not all(fit.significant(r) for r in fits.values()):
        return None
    A, sA = asy.raw_asymmetry(fits["D0"].n_sig, fits["D0bar"].n_sig,
                              fits["D0"].n_sig_err, fits["D0bar"].n_sig_err)
    if not (np.isfinite(A) and np.isfinite(sA) and sA > 0):
        return None
    return A, sA


def graded_verdict(p):
    if p < 0.001:
        return "INCOMPATIBLE"
    if p < 0.05:
        return "marginal"
    return "compatible"


print("=== loading (nominal selection, split per polarity) ===")
data = {m: load(m) for m in ("KPi", "KK")}
for m, per_pol in data.items():
    for pol, (dm, tag, run) in sorted(per_pol.items()):
        print(f"  {m:4s} {pol:8s}: {len(dm):,} in fit window, "
              f"runs {int(run.min())}-{int(run.max())}")

for mode in ("KPi", "KK"):
    hi = mode_window(mode)
    for pol, (dm, tag, run) in sorted(data[mode].items()):
        full = a_raw(dm, tag, hi)
        if full is None:
            print(f"\n=== {mode} {pol}: combined fit below thresholds or "
                  "invalid, skipped ===")
            continue
        shown = (f"combined A_raw = {full[0]:+.4f} +- {full[1]:.4f}"
                 if mode == "KPi" else
                 f"combined sigma_A = {full[1]:.4f} (central value blinded)")
        print(f"\n=== {mode} {pol}:  {shown} ===")
        runs_sorted = np.sort(run)
        edges = [int(runs_sorted[0])] + \
                [int(runs_sorted[int(len(runs_sorted) * k / N_BLOCKS)])
                 for k in range(1, N_BLOCKS)] + [int(runs_sorted[-1]) + 1]
        print(f"  run-block edges (equal thirds within {pol}): {edges}")
        vals, errs = [], []
        for k in range(N_BLOCKS):
            inb = (run >= edges[k]) & (run < edges[k + 1])
            r = a_raw(dm[inb], tag[inb], hi)
            if r is None:
                print(f"  block {k + 1} [{edges[k]}-{edges[k + 1]}): too few, "
                      "skipped")
                continue
            A, sA = r
            vals.append(A)
            errs.append(sA)
            shown = (f"A_raw = {A:+.4f} +- {sA:.4f}" if mode == "KPi"
                     else f"sigma_A = {sA:.4f} (value blinded)")
            print(f"  block {k + 1} [{edges[k]}-{edges[k + 1]}): "
                  f"{shown}   n = {int(inb.sum()):,}")
        if len(vals) >= 2:
            vals, errs = np.asarray(vals), np.asarray(errs)
            wavg, _ = asy.weighted_average(vals, errs)
            chi2 = float(np.sum(((vals - wavg) / errs) ** 2))
            ndf = len(vals) - 1
            p = float(chi2_dist.sf(chi2, ndf))
            print(f"  chi2/ndf across blocks = {chi2:.1f}/{ndf}  "
                  f"(p = {p:.3f})  -> {graded_verdict(p)}")

