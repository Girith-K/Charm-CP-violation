# the main script, selection, dm fits and the blinded dACP for each polarity

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uproot

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit
from charm_acp import gates as gt
from charm_acp import kinematics as kin
from charm_acp import selection as sel
from charm_acp.config import (DM_FIT_HI, DM_FIT_LO, GRID_ETA_EDGES,
                              GRID_PT_EDGES, PLOTS_DIR, RESULTS_DIR,
                              provenance, production_files, production_spec)

plt.style.use(mplhep.style.LHCb2)

MODES = ["KK", "PiPi", "KPi"]
CATS = ["D0", "D0bar"]
MIN_FIT_COMBINED = 150
MIN_FIT_CATEGORY = 30
CHUNK = "512 MB"

PT_EDGES = np.array(GRID_PT_EDGES)
ETA_EDGES = np.array(GRID_ETA_EDGES)

NOM = sel.NOMINAL_CUTS

p = argparse.ArgumentParser()
p.add_argument("--no-single-cand", action="store_true",
               help="keep every candidate instead of one per event")
p.add_argument("--ipchi2-max", type=float, default=NOM.d0_ipchi2_max,
               help="D0 IPchi2 upper cut, the prompt requirement")
p.add_argument("--pid-k-min", type=float, default=NOM.pid_k_min,
               help="offline kaon PIDK cut on top of the stripping PID")
p.add_argument("--soft-pid", action="store_true",
               help="apply the PID cut to the soft pion as well")
p.add_argument("--no-fiducial", action="store_true",
               help="disable the soft pion magnet edge fiducial exclusion")
p.add_argument("--signal-shape", default="dgauss",
               choices=["gauss", "dgauss", "tgauss"],
               help="signal model, the nominal is dgauss")
p.add_argument("--tag", default="prod", help="label used in output filenames")
args = p.parse_args()

CUTS = dataclasses.replace(sel.NOMINAL_CUTS, d0_ipchi2_max=args.ipchi2_max,
                           pid_k_min=args.pid_k_min, soft_pid=args.soft_pid,
                           soft_fiducial=not args.no_fiducial)
NON_NOMINAL = (CUTS != sel.NOMINAL_CUTS
               or args.no_single_cand
               or args.signal_shape != "dgauss")
if NON_NOMINAL and args.tag == "prod":
    raise SystemExit("non-nominal configuration with --tag prod would "
                     "overwrite the production results, pass an explicit --tag")
PROD_NAME, PROD = production_spec()
LAYOUT = PROD["layout"]
FILES = production_files()
SHAPE = args.signal_shape

print(f"production {PROD_NAME} (layout {LAYOUT}), signal shape {SHAPE}")
print(f"selection: kaon PIDK>{CUTS.pid_k_min}, soft-pion PID "
      f"{'ON' if CUTS.soft_pid else 'OFF'}, D0 IPchi2<{CUTS.d0_ipchi2_max}, "
      f"fiducial {'ON' if CUTS.soft_fiducial else 'OFF'}\n")

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

POLS = sorted({pol for _, pol in FILES})
lumi = {}
print(f"=== dataset: {len(FILES)} file(s) ===")
for path, pol in FILES:
    li = float(np.sum(uproot.open(path)["GetIntegratedLuminosity/LumiTuple"]
                      ["IntegratedLuminosity"].array(library="np")))
    lumi[pol] = lumi.get(pol, 0.0) + li
    print(f"  {path.name}  {pol:8s}  {path.stat().st_size / 1e9:5.2f} GB  "
          f"{li:8.3f} /pb")
lumi_total = sum(lumi.values())
print(f"  total {lumi_total:.3f} /pb   "
      + "   ".join(f"{k} {v:.3f}" for k, v in sorted(lumi.items())) + "\n")

cut_names = ["all candidates"] + [n for n, _ in sel.CUT_SEQUENCE]
data = {}
cutflows = {}
for mode in MODES:
    for pol in POLS:
        keep = {k: [] for k in ("dm", "tag", "pt", "eta", "run", "evt", "vchi2")}
        table_tot = np.zeros(len(cut_names), dtype=np.int64)
        n_raw = 0
        for path, fpol in FILES:
            if fpol != pol:
                continue
            for arr in sel.iter_mode(path, mode, LAYOUT, step_size=CHUNK):
                n_raw += len(arr["D0_MM"])
                mask, table = sel.cutflow(arr, mode, CUTS)
                table_tot += np.array([row[1] for row in table], dtype=np.int64)
                if not mask.any():
                    continue
                px, py, pz = (arr[f"Dst_2010_plus_P{c}"][mask] for c in "XYZ")
                keep["dm"].append((arr["Dst_2010_plus_MM"] - arr["D0_MM"])[mask])
                keep["tag"].append(sel.flavour_tag(arr)[mask])
                keep["pt"].append(kin.pt(px, py))
                keep["eta"].append(kin.eta(px, py, pz))
                keep["run"].append(arr["runNumber"][mask])
                keep["evt"].append(arr["eventNumber"][mask])
                keep["vchi2"].append((arr["D0_ENDVERTEX_CHI2"]
                                      / np.maximum(arr["D0_ENDVERTEX_NDOF"],
                                                   1))[mask])
        d = {k: (np.concatenate(v) if v else np.array([]))
             for k, v in keep.items()}
        n_before = len(d["dm"])
        if n_before and not args.no_single_cand:
            pick = sel.single_candidate_pick(d["run"], d["evt"], d["vchi2"])
            d = {k: v[pick] for k, v in d.items()}
        n_after = len(d["dm"])
        data[mode, pol] = d
        cutflows[f"{mode}_{pol}"] = [(name, int(n))
                                     for name, n in zip(cut_names, table_tot)]
        print(f"=== {mode} {pol} ===")
        print(f"  raw candidates        {n_raw:>12,}")
        for name, n in cutflows[f"{mode}_{pol}"]:
            print(f"  {name:20s}  {n:>12,}")
        print(f"  single-candidate      {n_after:>12,}   "
              f"(removed {n_before - n_after:,} = "
              f"{(n_before - n_after) / max(n_before, 1):.2%} multiples, "
              "tie-break seeded random)\n")


def mode_hi(mode):
    return 155.0 if (LAYOUT == "v3" and mode != "KPi") else DM_FIT_HI


def fit_araw(d, mode):
    hi = mode_hi(mode)
    win = (d["dm"] > DM_FIT_LO) & (d["dm"] < hi)
    by_cat = {"D0": d["dm"][win & (d["tag"] == 1)],
              "D0bar": d["dm"][win & (d["tag"] == -1)]}
    n_tot = sum(len(v) for v in by_cat.values())
    if n_tot < MIN_FIT_COMBINED or \
            min(len(v) for v in by_cat.values()) < MIN_FIT_CATEGORY:
        return None
    nbins = int(round((hi - DM_FIT_LO) / 0.25))
    fits, combined = fit.fit_categories(by_cat, share_shape=True, signal=SHAPE,
                                        hi=hi, nbins=nbins)
    if not (combined.valid and
            all(fit.significant(r) for r in fits.values())):
        return None
    A, sA = asy.raw_asymmetry(fits["D0"].n_sig, fits["D0bar"].n_sig,
                              fits["D0"].n_sig_err, fits["D0bar"].n_sig_err)

    out = {"fits": fits, "combined": combined, "A": float(A), "sA": float(sA),
           "gof": {"combined": combined.gof(),
                   **{c: fits[c].gof() for c in by_cat}},
           "lr_significance": {c: fit.lr_significance(fits[c])
                               for c in by_cat}}

    counts = {c: np.histogram(v, bins=nbins, range=(DM_FIT_LO, hi))[0]
              for c, v in by_cat.items()}
    edges = np.linspace(DM_FIT_LO, hi, nbins + 1)
    try:
        joint = fit.joint_asymmetry(counts, edges, signal=SHAPE)
    except (RuntimeError, ValueError):
        joint = None
    if joint is not None:
        Aj, sAj, jres, jfit = joint
        rho = (jfit.cov_nsig("D0", "D0bar")
               / max(jres["D0"].n_sig_err * jres["D0bar"].n_sig_err, 1e-300))
        out["joint"] = {"A": float(Aj), "sA": float(sAj),
                        "rho_yields": float(rho),
                        "chi2_ndf": jfit.chi2_ndf, "ndf": jfit.ndf,
                        "p_value": jfit.pvalue,
                        "sigma_ratio_vs_nominal": float(sAj / sA) if sA else None}
    return out


print("=== dm fits (shape shared between the two flavour categories) ===")
res = {}
for mode in MODES:
    for pol in POLS:
        r = fit_araw(data[mode, pol], mode)
        res[mode, pol] = r
        if r is None:
            print(f"  {mode} {pol}: below fit thresholds or invalid fit, SKIP")
            continue
        c = r["combined"]
        pur = c.n_sig / max(c.n_sig + c.n_bkg, 1.0)
        g = r["gof"]["combined"]
        print(f"  {mode} {pol}: mu={c.mu:.3f}  sigma={c.sigma:.3f}  "
              f"N_sig={c.n_sig:,.0f}  purity {pur:.1%}  valid={c.valid}")
        pv = "n/a" if g["p_value"] is None else f"{g['p_value']:.2e}"
        print(f"      goodness of fit: chi2/ndf = {g['chi2_lr']:.1f}/{g['ndf']}"
              f" = {g['chi2_ndf']:.2f}   p = {pv}   "
              f"max|pull| = {g['max_abs_pull']:.1f}")
        if g["chi2_ndf"] and g["chi2_ndf"] > 2.0:
            print("      WARNING: chi2/ndf > 2")
        for cat in ("D0", "D0bar"):
            lr = r["lr_significance"].get(cat)
            gc = r["gof"][cat]
            print(f"      {cat:5s}: LR significance "
                  + ("n/a" if lr is None else f"{lr:8.1f} sigma")
                  + f"   chi2/ndf = {gc['chi2_lr']:.1f}/{gc['ndf']} "
                    f"= {gc['chi2_ndf']:.2f}")
        if "joint" in r:
            j = r["joint"]
            print(f"      simultaneous-fit cross-check: A = {j['A']:+.6f} "
                  f"+- {j['sA']:.6f}  (nominal {r['A']:+.6f} "
                  f"+- {r['sA']:.6f})")
            print(f"      yield correlation rho = {j['rho_yields']:+.3f}; "
                  f"sigma ratio joint/nominal = "
                  f"{j['sigma_ratio_vs_nominal']:.3f}; "
                  f"joint chi2/ndf = {j['chi2_ndf']:.2f}")


def polarity_average(mode):
    vals = [(res[mode, pol]["A"], res[mode, pol]["sA"])
            for pol in POLS if res[mode, pol]]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    if len(vals) != 2:
        raise SystemExit(f"{mode}: {len(vals)} polarity subsamples, the "
                         "equal-weight average is defined for exactly 2")
    (a1, s1), (a2, s2) = vals
    return asy.polarity_average(a1, s1, a2, s2)


print("\n=== raw asymmetries ===")
araw_out = {}
composition = {}
if any(res["KPi", pol] for pol in POLS):
    kpi = {}
    for pol in POLS:
        if res["KPi", pol]:
            a, s = res["KPi", pol]["A"], res["KPi", pol]["sA"]
            kpi[pol] = {"value": a, "error": s}
            print(f"  A_raw(KPi {pol:8s}) = {a:+.4f} +- {s:.4f}")
    avg = polarity_average("KPi")
    kpi["average"] = {"value": avg[0], "error": avg[1]}
    print(f"  A_raw(KPi average ) = {avg[0]:+.4f} +- {avg[1]:.4f}")
    if len(kpi) == 3:
        d1, d2 = (res["KPi", p_]["A"] for p_ in POLS)
        s12 = float(np.hypot(*(res["KPi", p_]["sA"] for p_ in POLS)))
        kpi["polarity_split_sigma"] = abs(d1 - d2) / s12
        n_chunks = {p_: sum(1 for _, pp in FILES if pp == p_) for p_ in POLS}
        chunk_note = "/".join(f"{p_} x{n_chunks[p_]}" for p_ in POLS)
        print(f"  polarity split      = {d1 - d2:+.4f} +- {s12:.4f}  "
              f"({abs(d1 - d2) / s12:.1f} sigma, chunks {chunk_note})")
    araw_out["KPi"] = kpi

def polarities_used(mode):
    return {pol for pol in POLS if res[mode, pol]}


for mode in ("KK", "PiPi"):
    avg = polarity_average(mode)
    if avg is None:
        araw_out[mode] = None
        continue
    pols_m = polarities_used(mode)
    n_mode = sum(res[mode, p_]["combined"].n_sig for p_ in pols_m)
    n_kpi = sum(res["KPi", p_]["combined"].n_sig for p_ in pols_m
                if res["KPi", p_])
    g = gt.composition_gate(n_mode, n_kpi, mode)
    composition[mode] = g
    if not g["passed"]:
        pct = "n/a" if g["ratio"] is None else f"{g['ratio']:.2%}"
        print(f"  A_raw({mode:4s}) not quoted, N_sig is {pct} of "
              "the BR expectation from Kpi")
        araw_out[mode] = {"blinded_average": None,
                          "suppressed_reason": g["reason"],
                          "yield_over_expected": g["ratio"],
                          "composition_gate": g}
        continue
    ba, bs = asy.blind_raw(avg[0], avg[1], mode)
    araw_out[mode] = {"blinded_average": {"value": ba, "error": bs}}
    print(f"  A_raw({mode:4s}) BLINDED = {ba:+.4f} +- {bs:.4f}   "
          "(per-mode offset, never printed raw)")

result = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "provenance": provenance(),
    "production": PROD_NAME,
    "layout": LAYOUT,
    "signal_shape": SHAPE,
    "files": [[path.name, pol] for path, pol in FILES],
    "luminosity_pb": {**lumi, "total": lumi_total},
    "single_candidate": not args.no_single_cand,
    "cuts": {"pid_k_min": CUTS.pid_k_min, "soft_pid": CUTS.soft_pid,
             "d0_ipchi2_max": CUTS.d0_ipchi2_max,
             "soft_fiducial": CUTS.soft_fiducial},
    "cutflow": cutflows,
    "fits": {f"{mode}_{pol}": ({"n_sig_total": float(r["combined"].n_sig),
                                "sigma_dm": float(r["combined"].sigma),
                                "mu": float(r["combined"].mu),
                                "valid": bool(r["combined"].valid),
                                "gof": r["gof"],
                                "lr_significance": r["lr_significance"],
                                "joint_cross_check": r.get("joint")}
                               if r else None)
             for (mode, pol), r in res.items()},
    "A_raw": araw_out,
}
for pol in POLS:
    if res["KPi", pol]:
        f_ = res["KPi", pol]["fits"]
        result["fits"][f"KPi_{pol}"]["by_cat"] = {
            c: {"n_sig": float(f_[c].n_sig), "n_sig_err": float(f_[c].n_sig_err)}
            for c in CATS}

d_int = None
cp_quotable = all(araw_out.get(m) and araw_out[m].get("blinded_average")
                  for m in ("KK", "PiPi"))
pol_gate = gt.polarity_set_gate(
    {m: polarities_used(m) for m in ("KK", "PiPi")}, POLS)
result_gates = {"composition": composition, "polarity": pol_gate}
result["gates"] = result_gates
if cp_quotable and not pol_gate["passed"]:
    print("\n  dACP not formed: KK and PiPi did not fit in the same, complete"
          f" polarity set ({pol_gate['used']})")
    cp_quotable = False
if not cp_quotable:
    reason = gt.gate_summary(list(composition.values()) + [pol_gate])
    print(f"\n  dACP not formed. Gate: {reason}")
if cp_quotable:
    a_kk = polarity_average("KK")
    a_pp = polarity_average("PiPi")
    d_int, s_int = asy.delta_acp(*a_kk, *a_pp)
    bd, bs = asy.blind_delta_acp(d_int, s_int)
    print(f"\n  dACP (integrated, BLINDED) = {bd:+.4f} +- {bs:.4f}")
    result["delta_acp_blinded"] = {"value": bd, "error": bs,
                                   "method": "integrated, polarity averaged"}

    print("\n=== Phase 8: (pT, eta)-binned dACP ===")
    per_bin = {}
    for pol in POLS:
        if not (res["KK", pol] and res["PiPi", pol]):
            continue
        for ipt in range(len(PT_EDGES) - 1):
            for ieta in range(len(ETA_EDGES) - 1):
                per_mode = {}
                for mode in ("KK", "PiPi"):
                    d = data[mode, pol]
                    hi_m = mode_hi(mode)
                    inbin = ((d["pt"] >= PT_EDGES[ipt])
                             & (d["pt"] < PT_EDGES[ipt + 1])
                             & (d["eta"] >= ETA_EDGES[ieta])
                             & (d["eta"] < ETA_EDGES[ieta + 1])
                             & (d["dm"] > DM_FIT_LO) & (d["dm"] < hi_m))
                    by_cat = {"D0": d["dm"][inbin & (d["tag"] == 1)],
                              "D0bar": d["dm"][inbin & (d["tag"] == -1)]}
                    if min(len(v) for v in by_cat.values()) < MIN_FIT_CATEGORY:
                        per_mode = None
                        break
                    comb = res[mode, pol]["combined"]
                    shape = {q: getattr(comb, q)
                             for q in fit.shape_pars(SHAPE) if q != "mu"}
                    ok = True
                    fits_bin = {}
                    for c, v in by_cat.items():
                        rb = fit.fit_dm(v, hi=hi_m,
                                        nbins=int(round((hi_m - DM_FIT_LO)
                                                        / 0.25)),
                                        start={"n_sig": 0.3 * len(v),
                                               "n_bkg": 0.7 * len(v),
                                               "mu": comb.mu},
                                        fixed_shape=shape, signal=SHAPE)
                        if not fit.significant(rb):
                            ok = False
                            break
                        fits_bin[c] = rb
                    if not ok:
                        per_mode = None
                        break
                    per_mode[mode] = fits_bin
                if per_mode is None:
                    continue
                a_kk = asy.raw_asymmetry(
                    per_mode["KK"]["D0"].n_sig, per_mode["KK"]["D0bar"].n_sig,
                    per_mode["KK"]["D0"].n_sig_err,
                    per_mode["KK"]["D0bar"].n_sig_err)
                a_pp = asy.raw_asymmetry(
                    per_mode["PiPi"]["D0"].n_sig,
                    per_mode["PiPi"]["D0bar"].n_sig,
                    per_mode["PiPi"]["D0"].n_sig_err,
                    per_mode["PiPi"]["D0bar"].n_sig_err)
                d_b, s_b = asy.delta_acp(*a_kk, *a_pp)
                per_bin.setdefault((ipt, ieta), {})[pol] = (float(d_b),
                                                            float(s_b))

    vals, errs, n_single = [], [], 0
    for b, by_pol in per_bin.items():
        if len(by_pol) == 2:
            (d1, s1), (d2, s2) = by_pol.values()
            vals.append(0.5 * (d1 + d2))
            errs.append(0.5 * float(np.hypot(s1, s2)))
        else:
            n_single += 1
    n_used = len(vals)

    if n_used >= 2:
        d_corr, s_corr = asy.weighted_average(vals, errs)
        bdc, bsc = asy.blind_delta_acp(d_corr, s_corr)
        print(f"  bins used: {n_used} with both polarities, "
              f"{n_single} single-polarity dropped")
        print(f"  dACP ((pT,eta)-corrected, BLINDED) = {bdc:+.4f} +- {bsc:.4f}")
        result["delta_acp_blinded_binned"] = {"value": bdc, "error": bsc,
                                              "n_bins": n_used,
                                              "n_bins_single_pol_dropped":
                                                  n_single,
                                              "method": "binned, polarity "
                                                        "averaged per bin"}
        result["nominal_method"] = ("integrated, polarity averaged "
                                    "(binned-integrated as systematic)")
    else:
        print(f"  only {n_used} both-polarity bin(s) fit, "
              f"{n_single} single-polarity dropped")
        result["delta_acp_blinded_binned"] = None
        result["nominal_method"] = "integrated, polarity averaged"
else:
    print("\n  dACP: KK and/or pipi below fit thresholds in every polarity")
    result["delta_acp_blinded"] = None
    result["delta_acp_blinded_binned"] = None
    result["nominal_method"] = None

out_json = RESULTS_DIR / f"{args.tag}_results.json"
out_json.write_text(json.dumps(result, indent=2))
print(f"\nwrote {out_json.relative_to(RESULTS_DIR.parent)}")

for pol in POLS:
    if not res["KPi", pol]:
        continue
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [3, 1],
                                          "hspace": 0.05, "wspace": 0.10})
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    for j, cat in enumerate(CATS):
        r = res["KPi", pol]["fits"][cat]
        ax, axp = axes[0, j], axes[1, j]
        centers = 0.5 * (r.edges[1:] + r.edges[:-1])
        w = r.edges[1] - r.edges[0]
        ax.errorbar(centers, r.counts, yerr=np.sqrt(r.counts), fmt="k.", ms=4,
                    label="Data")
        ax.plot(centers, r.pred, "-", lw=2, label="Total fit")
        bkg_cdf = fit._BKG_CDFS[r.bkg]
        ax.plot(centers, r.n_bkg * np.diff(bkg_cdf(r.edges, r.a, r.b,
                                                   r.edges[0], r.edges[-1])),
                "--", lw=1.5, label="Background")
        tag = r"$\pi_s^+$ ($D^0$)" if cat == "D0" else r"$\pi_s^-$ ($\bar{D}^0$)"
        ax.set_title(f"KPi {pol}, {tag}", fontsize=11)
        ax.legend(fontsize=9)
        axp.axhline(0, color="gray", lw=1)
        axp.bar(centers, r.pulls, width=w, color="steelblue")
        axp.set_ylim(-5, 5)
        axp.set_xlabel(r"$\Delta m$ [MeV]")
        if j == 0:
            ax.set_ylabel(f"Candidates / {w:.2f} MeV", fontsize=13)
            axp.set_ylabel("Pull", fontsize=13)
    out = PLOTS_DIR / f"{args.tag}_dm_fit_KPi_{pol}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")

for mode in ("KK", "PiPi"):
    for pol in POLS:
        if not res[mode, pol]:
            continue
        c = res[mode, pol]["combined"]
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1],
                                                   "hspace": 0.05})
        centers = 0.5 * (c.edges[1:] + c.edges[:-1])
        w = c.edges[1] - c.edges[0]
        ax.errorbar(centers, c.counts, yerr=np.sqrt(c.counts), fmt="k.", ms=4,
                    label="Data")
        ax.plot(centers, c.pred, "-", lw=2, label="Total fit")
        ax.set_title(f"{mode} {pol}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylabel(f"Candidates / {w:.2f} MeV", fontsize=12)
        axp.axhline(0, color="gray", lw=1)
        axp.bar(centers, c.pulls, width=w, color="steelblue")
        axp.set_ylim(-5, 5)
        axp.set_xlabel(r"$\Delta m$ [MeV]")
        axp.set_ylabel("Pull", fontsize=12)
        out = PLOTS_DIR / f"{args.tag}_dm_fit_{mode}_{pol}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}")
