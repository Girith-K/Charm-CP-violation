# toy pull study on the raw asymmetry

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from scipy.stats import norm

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit
from charm_acp import gates as gt
from charm_acp import selection as sel
from charm_acp.config import (DM_FIT_HI, DM_FIT_LO, PLOTS_DIR, RESULTS_DIR,
                              SEED, production_files, production_spec)

plt.style.use(mplhep.style.LHCb2)

ap = argparse.ArgumentParser()
ap.add_argument("--toys", type=int, default=1000)
ap.add_argument("--max-fail-frac", type=float, default=0.05,
                help="fraction of toys allowed to fail before the gate does")
args = ap.parse_args()

CUTS = sel.NOMINAL_CUTS
LAYOUT = production_spec()[1]["layout"]
MIN_PER_CAT = 30
SIGNAL = "dgauss"
PARS = fit.shape_pars(SIGNAL)


def mode_edges(mode):
    hi = 155.0 if (LAYOUT == "v3" and mode != "KPi") else DM_FIT_HI
    nbins = int(round((hi - DM_FIT_LO) / 0.25))
    return np.linspace(DM_FIT_LO, hi, nbins + 1), hi

print("=== measuring the truth from data (nominal selection) ===")
truth = {}
for mode in ("KK", "PiPi", "KPi"):
    dm_all, tag_all, run_all_, evt_all, v_all = [], [], [], [], []
    for path, pol in production_files():
        for arr in sel.iter_mode(path, mode, LAYOUT):
            mask, _ = sel.cutflow(arr, mode, CUTS)
            dm_all.append((arr["Dst_2010_plus_MM"] - arr["D0_MM"])[mask])
            tag_all.append(sel.flavour_tag(arr)[mask])
            run_all_.append(arr["runNumber"][mask])
            evt_all.append(arr["eventNumber"][mask])
            v_all.append((arr["D0_ENDVERTEX_CHI2"]
                          / np.maximum(arr["D0_ENDVERTEX_NDOF"], 1))[mask])
    dm, tag = np.concatenate(dm_all), np.concatenate(tag_all)
    pick = sel.single_candidate_pick(np.concatenate(run_all_),
                                     np.concatenate(evt_all),
                                     np.concatenate(v_all))
    dm, tag = dm[pick], tag[pick]
    edges, hi = mode_edges(mode)
    win = (dm > DM_FIT_LO) & (dm < hi)
    by_cat = {"D0": dm[win & (tag > 0)], "D0bar": dm[win & (tag < 0)]}
    if min(len(v) for v in by_cat.values()) < MIN_PER_CAT:
        print(f"  {mode:5s} below statistics threshold, no toys for this mode")
        continue

    fits, combined = fit.fit_categories(by_cat, share_shape=True, hi=hi,
                                        nbins=len(edges) - 1, signal=SIGNAL)
    if not all(fit.significant(r) for r in fits.values()):
        print(f"  {mode:5s} truth fit invalid or signal below 3 sigma, "
              "no toys for this mode (fits on it are not used anywhere)")
        continue
    shared = {p: getattr(combined, p) for p in PARS}
    truth[mode] = {
        "shape": shared,
        "shape_by_cat": {c: {**shared, "mu": fits[c].mu}
                         for c in ("D0", "D0bar")},
        "n_sig": {c: fits[c].n_sig for c in ("D0", "D0bar")},
        "n_bkg": {c: fits[c].n_bkg for c in ("D0", "D0bar")},
        "dmu_truth": float(fits["D0"].mu - fits["D0bar"].mu),
        "edges": edges,
    }
    A, sA = asy.raw_asymmetry(fits["D0"].n_sig, fits["D0bar"].n_sig,
                              fits["D0"].n_sig_err, fits["D0bar"].n_sig_err)
    truth[mode]["A_data"] = (A, sA)
    n_tot = fits["D0"].n_sig + fits["D0bar"].n_sig
    dmu_kev = truth[mode]["dmu_truth"] * 1000.0
    if mode == "KPi":
        print(f"  {mode:5s} N(D0)={fits['D0'].n_sig:9.1f}  "
              f"N(D0bar)={fits['D0bar'].n_sig:9.1f}  "
              f"mu={combined.mu:.3f} sigma={combined.sigma:.3f}  "
              f"dmu(D0-D0bar)={dmu_kev:+.2f} keV")
    else:
        print(f"  {mode:5s} N_sig(total)={n_tot:9.1f}  "
              f"mu={combined.mu:.3f} sigma={combined.sigma:.3f}  "
              f"dmu={dmu_kev:+.2f} keV  (flavour split not printed, blinding)")

if "KPi" in truth:
    n_kpi = truth["KPi"]["n_sig"]["D0"] + truth["KPi"]["n_sig"]["D0bar"]
    for mode in list(truth):
        if mode == "KPi":
            continue
        n_mode = truth[mode]["n_sig"]["D0"] + truth[mode]["n_sig"]["D0bar"]
        g = gt.composition_gate(n_mode, n_kpi, mode)
        if not g["passed"]:
            pct = "n/a" if g["ratio"] is None else f"{g['ratio']:.3%}"
            print(f"  {mode:5s} refused by the composition gate "
                  f"({pct} of the BR-scaled Kpi expectation), no toys")
            del truth[mode]

if not truth:
    print("no mode has enough statistics to gate on, refusing to pass")
    sys.exit(1)

rng = np.random.default_rng(SEED + 13)


def expected(n_sig, n_bkg, shape, edges):
    lo, hi = float(edges[0]), float(edges[-1])
    sig = fit.sig_cdf(SIGNAL, edges, shape["mu"], shape["sigma"], shape,
                      lo, hi)
    cdf = n_sig * sig + n_bkg * fit._bkg_cdf(edges, shape["a"], shape["b"],
                                             lo, hi)
    return np.diff(cdf)


def run_toys(mode, n_toys):
    t = truth[mode]
    edges = t["edges"]
    pred = {c: expected(t["n_sig"][c], t["n_bkg"][c], t["shape_by_cat"][c],
                        edges)
            for c in ("D0", "D0bar")}
    A_true = ((t["n_sig"]["D0"] - t["n_sig"]["D0bar"])
              / (t["n_sig"]["D0"] + t["n_sig"]["D0bar"]))

    pulls, errs, deltas, failed = [], [], [], 0
    for _ in range(n_toys):
        counts = {c: rng.poisson(pred[c]) for c in ("D0", "D0bar")}

        res, comb = fit.fit_categories_binned(counts, edges, signal=SIGNAL)
        if res is None or not comb.valid:
            failed += 1
            continue
        if not all(r.valid and r.n_sig_err > 0 for r in res.values()):
            failed += 1
            continue

        A, sA = asy.raw_asymmetry(res["D0"].n_sig, res["D0bar"].n_sig,
                                  res["D0"].n_sig_err, res["D0bar"].n_sig_err)
        if not np.isfinite(A) or not np.isfinite(sA) or sA <= 0:
            failed += 1
            continue
        pulls.append((A - A_true) / sA)
        errs.append(sA)
        deltas.append(A - A_true)

    return (np.asarray(pulls), np.asarray(errs), np.asarray(deltas),
            failed, A_true)


print(f"\n=== {args.toys} toys per mode, through fit_categories ===")
out = {}
gate_ok = True
for mode in truth:
    pulls, errs, deltas, failed, A_true = run_toys(mode, args.toys)
    n = len(pulls)
    if n < max(50, args.toys // 4):
        print(f"\n  --- {mode}: only {n} usable toys, gate FAILS")
        gate_ok = False
        continue
    mean, mean_e = float(np.mean(pulls)), float(np.std(pulls) / np.sqrt(n))
    width, width_e = float(np.std(pulls)), float(np.std(pulls) / np.sqrt(2 * n))
    rms_delta = float(np.std(deltas))
    med_err = float(np.median(errs))
    fail_frac = failed / float(args.toys)
    out[mode] = dict(pulls=pulls, mean=mean, mean_err=mean_e, width=width,
                     rms=rms_delta, med_err=med_err, failed=failed,
                     failed_frac=fail_frac)

    print(f"\n  --- {mode} ---")
    print(f"  usable toys      : {n}/{args.toys}  ({failed} failed, "
          f"{fail_frac:.1%})")
    print(f"  pull mean        : {mean:+.3f} +- {mean_e:.3f}     (want 0)")
    print(f"  pull width       : {width:.3f} +- {width_e:.3f}     (want 1)")
    print(f"  true scatter of A: {rms_delta:.4f}")
    honest = abs(width - 1) < 0.1
    print(f"  median quoted err: {med_err:.4f}   "
          f"-> errors {'OK' if honest else 'MIS-SIZED'}"
          f" (scatter/quoted = {rms_delta / med_err:.3f})")
    bias_sig = abs(mean) / mean_e
    biased = bias_sig >= 3
    print(f"  bias significance: {bias_sig:.1f} sigma "
          f"-> {'no evidence of bias' if not biased else 'BIASED'}")
    converged = fail_frac <= args.max_fail_frac
    verdict = ("OK" if converged else
               "TOO MANY")
    print(f"  toy failure rate : {fail_frac:.1%} "
          f"(limit {args.max_fail_frac:.0%}) -> {verdict}")
    gate_ok &= honest and not biased and converged

if all(m in out for m in ("KK", "PiPi")):
    print("\n=== consequence for dACP ===")
    scale = {m: out[m]["width"] for m in ("KK", "PiPi")}
    s_kk = truth["KK"]["A_data"][1] * scale["KK"]
    s_pp = truth["PiPi"]["A_data"][1] * scale["PiPi"]
    print(f"  quoted sigma(A_raw): KK {truth['KK']['A_data'][1]:.4f}, "
          f"pipi {truth['PiPi']['A_data'][1]:.4f}")
    print(f"  pull-width scaling : KK x{scale['KK']:.3f}, "
          f"pipi x{scale['PiPi']:.3f}")
    print(f"  corrected sigma    : KK {s_kk:.4f}, pipi {s_pp:.4f}")

RESULTS_DIR.mkdir(exist_ok=True)
gate_json = {m: {**{k: r[k] for k in ("mean", "mean_err", "width", "rms",
                                      "med_err", "failed", "failed_frac")},
                 "n_toys": args.toys,
                 "dmu_truth_keV": truth[m]["dmu_truth"] * 1000.0}
             for m, r in out.items()}
gate_json["gate_ok"] = bool(gate_ok)
gate_json["max_fail_frac"] = args.max_fail_frac
gate_json["estimator"] = ("shared shape, per-flavour mean free "
                          "(fit_categories_binned, share_mean=False)")
(RESULTS_DIR / "pull_gate.json").write_text(json.dumps(gate_json, indent=2))
print(f"\nwrote pull_gate.json")

if out:
    fig, axes = plt.subplots(1, len(out), figsize=(6.5 * len(out), 5),
                             squeeze=False)
    x = np.linspace(-4, 4, 200)
    for ax, mode in zip(axes[0], out):
        r = out[mode]
        n = len(r["pulls"])
        ax.hist(r["pulls"], bins=40, range=(-4, 4), histtype="step", lw=1.8,
                color="navy", label="toys")
        ax.plot(x, n * (8 / 40) * norm.pdf(x), "--", color="crimson", lw=1.5,
                label="unit Gaussian")
        ax.set_xlabel(r"pull $=(A_{\rm fit}-A_{\rm true})/\sigma_A$")
        ax.set_ylabel("toys")
        ax.set_title(mode, fontsize=11)
        ax.legend(fontsize=9)
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out_png = PLOTS_DIR / "asymmetry_pull_gate.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png.name}")

if not gate_ok:
    print("\nGATE FAILED, the chain stops here")
    sys.exit(1)
print("\nGATE PASSED")
