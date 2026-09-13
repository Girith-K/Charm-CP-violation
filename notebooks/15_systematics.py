# systematics, reruns the measurement with each cut and fit choice changed

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone

import numpy as np

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit
from charm_acp import gates as gt
from charm_acp import selection as sel
from charm_acp.config import (DM_FIT_HI, DM_FIT_LO, DM_NBINS, RESULTS_DIR,
                              blind_scale, production_files, production_spec,
                              provenance)

CP_MODES = ("KK", "PiPi")
KPI = "KPi"
ALL_MODES = CP_MODES + (KPI,)
MIN_CAT = 30

RESULTS_DIR.mkdir(exist_ok=True)

PROD_NAME, PROD_SPEC = production_spec()
LAYOUT = PROD_SPEC["layout"]
HAS_PROBNN = LAYOUT == "v3"
BASELINE = dataclasses.asdict(sel.NOMINAL_CUTS)

META = {"production": PROD_NAME, "blind_scale": blind_scale(),
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "provenance": provenance()}

ENVELOPE = sel.Cuts(pid_k_min=0.0, pid_pi_max=BASELINE["pid_pi_max"],
                    soft_pid=False, ghost_max=0.5, d0_mass_win=25.0,
                    d0_ipchi2_max=25.0, d0_tau_min=BASELINE["d0_tau_min"],
                    soft_pt_min=200.0, soft_fiducial=False)


def keep_columns(mode):
    p = sel.PARTS[mode]
    cols = ["Dst_2010_plus_MM", "Dst_2010_plus_ID", "D0_MM",
            "D0_IPCHI2_OWNPV", "D0_TAU", "D0_ENDVERTEX_CHI2",
            "D0_ENDVERTEX_NDOF", "runNumber", "eventNumber"]
    for part in (p["h1"], p["h2"], p["soft"]):
        cols += [f"{part}_PIDK", f"{part}_TRACK_GhostProb"]
        if HAS_PROBNN:
            cols += [f"{part}_{sel.TUNE}_ProbNNk",
                     f"{part}_{sel.TUNE}_ProbNNpi"]
    cols += [f"{p['soft']}_PT", f"{p['soft']}_PX", f"{p['soft']}_PZ"]
    return list(dict.fromkeys(cols))


def load(mode, pol):
    cols = keep_columns(mode)
    chunks = []
    for path, fpol in production_files():
        if fpol != pol:
            continue
        for arr in sel.iter_mode(path, mode, LAYOUT):
            mask = np.logical_and.reduce([
                fn(arr, mode, ENVELOPE) for _name, fn in sel.CUT_SEQUENCE])
            if not mask.any():
                continue
            chunks.append({k: arr[k][mask] for k in cols})
    if not chunks:
        return None
    return {k: np.concatenate([c[k] for c in chunks]) for k in cols}


POLS = sorted({pol for _, pol in production_files()})
DATA = {}
for m in ALL_MODES:
    for p_ in POLS:
        DATA[m, p_] = load(m, p_)
        n = len(DATA[m, p_]["D0_MM"]) if DATA[m, p_] else 0
        print(f"loaded {m:5s} {p_:8s}: {n:>12,} (envelope-selected)")
print()


def resolve_window(mode, lo, hi, nbins):
    if LAYOUT == "v3" and mode != KPI:
        width = (hi - lo) / nbins
        hi = min(hi, 155.0)
        nbins = max(1, int(round((hi - lo) / width)))
    return lo, hi, nbins


def measure_modes(modes, cuts_kw=None, lo=DM_FIT_LO, hi=DM_FIT_HI,
                  nbins=DM_NBINS, single_cand=True, shape_shift=None,
                  signal="dgauss", bkg="threshold", share_mean=False,
                  share_width=True, single_cand_by="vchi2",
                  estimator="two_step"):
    kw = dict(BASELINE)
    if cuts_kw:
        kw.update(cuts_kw)
    cuts = sel.Cuts(**kw)

    out = {}
    for mode in modes:
        w_lo, w_hi, w_nb = resolve_window(mode, lo, hi, nbins)
        per_pol, n_by_pol = [], []
        for pol in POLS:
            arr = DATA[mode, pol]
            if arr is None:
                continue
            mask, _ = sel.cutflow(arr, mode, cuts)
            if single_cand and mask.any():
                idx = np.flatnonzero(mask)
                vchi2 = (arr["D0_ENDVERTEX_CHI2"][idx]
                         / np.maximum(arr["D0_ENDVERTEX_NDOF"][idx], 1))
                pick = sel.single_candidate_pick(arr["runNumber"][idx],
                                                 arr["eventNumber"][idx],
                                                 vchi2, by=single_cand_by)
                mask = np.zeros_like(mask)
                mask[idx[pick]] = True
            dm = arr["Dst_2010_plus_MM"] - arr["D0_MM"]
            tag = sel.flavour_tag(arr)
            win = mask & (dm > w_lo) & (dm < w_hi)
            by_cat = {"D0": dm[win & (tag > 0)], "D0bar": dm[win & (tag < 0)]}
            if min(len(v) for v in by_cat.values()) < MIN_CAT:
                continue

            if estimator == "joint":
                edges = np.linspace(w_lo, w_hi, w_nb + 1)
                counts = {c: np.histogram(v, bins=w_nb,
                                          range=(w_lo, w_hi))[0]
                          for c, v in by_cat.items()}
                try:
                    got = fit.joint_asymmetry(
                        counts, edges, signal=signal, bkg=bkg,
                        share_mean=share_mean, share_width=share_width)
                except (RuntimeError, ValueError):
                    got = None
                if got is None:
                    continue
                A_, s_, jres, _jfit = got
                if not all(r.n_sig_err > 0 for r in jres.values()):
                    continue
                per_pol.append((A_, s_))
                n_by_pol.append(sum(r.n_sig for r in jres.values()))
                continue

            res, combined = fit.fit_categories(
                by_cat, share_shape=True, lo=w_lo, hi=w_hi, nbins=w_nb,
                signal=signal, bkg=bkg, share_mean=share_mean,
                shape_shift=shape_shift, share_width=share_width)
            if not combined.valid:
                continue
            if not all(fit.significant(r) for r in res.values()):
                continue
            per_pol.append(asy.raw_asymmetry(
                res["D0"].n_sig, res["D0bar"].n_sig,
                res["D0"].n_sig_err, res["D0bar"].n_sig_err))
            n_by_pol.append(float(combined.n_sig))
        if len(per_pol) < len(POLS):
            return None
        if len(per_pol) != 2:
            raise SystemExit(f"{mode}: {len(per_pol)} polarity subsamples,"
                             " the equal-weight average needs exactly 2")
        (a1, s1), (a2, s2) = per_pol
        A, sA = asy.polarity_average(a1, s1, a2, s2)
        out[mode] = {"A": float(A), "sA": float(sA),
                     "n_sig": float(sum(n_by_pol))}
    return out


def dacp_with_gate(**kwargs):
    A = measure_modes(CP_MODES + (KPI,), **kwargs)
    if A is None:
        return None, {"measurable": False,
                      "reason": "a mode did not fit in every polarity"}
    checks = [gt.composition_gate(A[m]["n_sig"], A[KPI]["n_sig"], m)
              for m in CP_MODES]
    report = {"measurable": all(g["passed"] for g in checks),
              "reason": gt.gate_summary(checks),
              "composition": {g["mode"]: g for g in checks}}
    if not report["measurable"]:
        return None, report
    d, s = asy.delta_acp(A["KK"]["A"], A["KK"]["sA"],
                         A["PiPi"]["A"], A["PiPi"]["sA"])
    return asy.blind_delta_acp(d, s), report


def measure_dacp(**kwargs):
    return dacp_with_gate(**kwargs)[0]


def measure_kpi(**kwargs):
    A = measure_modes((KPI,), **kwargs)
    return None if A is None else (A[KPI]["A"], A[KPI]["sA"])


SIGMA_REL = "sigma_rel"

VARIATIONS = [
    ("selection", "kaon PIDK > 5 (tight)", dict(cuts_kw=dict(pid_k_min=5.0))),
    ("selection", "kaon PIDK > 3", dict(cuts_kw=dict(pid_k_min=3.0))),
    ("selection", "soft-pion PID on", dict(cuts_kw=dict(soft_pid=True))),
    ("selection", "ghost prob < 0.2", dict(cuts_kw=dict(ghost_max=0.2))),
    ("selection", "ghost prob < 0.5", dict(cuts_kw=dict(ghost_max=0.5))),
    ("selection", "soft-pion pT > 300", dict(cuts_kw=dict(soft_pt_min=300.0))),
    ("selection", "D0 mass win 20 MeV", dict(cuts_kw=dict(d0_mass_win=20.0))),
    ("selection", "fiducial cut off",
     dict(cuts_kw=dict(soft_fiducial=False))),
    ("selection", "fiducial pz edge 3 GeV",
     dict(cuts_kw=dict(fid_pz_min=3.0, fid_pz_mid=5.0))),
    ("selection", "fiducial pz edge 5 GeV",
     dict(cuts_kw=dict(fid_pz_min=5.0, fid_pz_mid=7.0))),
    ("selection", "fiducial px edge 1.5 GeV",
     dict(cuts_kw=dict(fid_px_max=1.5))),
    ("secondary charm", "D0 IPchi2 < 4 (tighter)",
     dict(cuts_kw=dict(d0_ipchi2_max=4.0))),
    ("secondary charm", "D0 IPchi2 < 25 (looser)",
     dict(cuts_kw=dict(d0_ipchi2_max=25.0))),
    ("fit model", "sigma(dm) +10%", dict(shape_shift={SIGMA_REL: +0.10})),
    ("fit model", "sigma(dm) -10%", dict(shape_shift={SIGMA_REL: -0.10})),
    ("fit model", "shared flavour mean", dict(share_mean=True)),
    ("fit model", "free per-flavour width", dict(share_width=False)),
    ("fit model", "single Gaussian signal", dict(signal="gauss")),
    ("fit model", "triple Gaussian (LHCb shape)", dict(signal="tgauss")),
    ("fit model", "powexp background", dict(bkg="powexp")),
    ("fit model", "window lo +0.5 MeV", dict(lo=140.5, hi=157.5)),
    ("fit model", "window lo +1.0 MeV", dict(lo=141.0, hi=158.0)),
    ("fit model", "bin width 0.5 MeV", dict(nbins=36)),
    ("fit model", "bin width 0.2 MeV", dict(nbins=90)),
    ("candidates", "keep multiple candidates", dict(single_cand=False)),
    ("candidates", "random candidate choice",
     dict(single_cand_by="random")),
]

if HAS_PROBNN:
    VARIATIONS[2:2] = [
        ("selection", "ProbNNk > 0.2",
         dict(cuts_kw=dict(probnn_k_min=0.2))),
        ("selection", "ProbNNk > 0.4",
         dict(cuts_kw=dict(probnn_k_min=0.4))),
        ("selection", "ProbNNpi > 0.2",
         dict(cuts_kw=dict(probnn_pi_min=0.2))),
    ]

VARIATIONS.append(
    ("fit model", "simultaneous (joint) fit", dict(estimator="joint")))

def resolution(mode):
    vals = []
    for pol in POLS:
        arr = DATA[mode, pol]
        if arr is None:
            continue
        mask, _ = sel.cutflow(arr, mode, sel.Cuts(**BASELINE))
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        vchi2 = (arr["D0_ENDVERTEX_CHI2"][idx]
                 / np.maximum(arr["D0_ENDVERTEX_NDOF"][idx], 1))
        pick = sel.single_candidate_pick(arr["runNumber"][idx],
                                         arr["eventNumber"][idx], vchi2)
        keep = np.zeros_like(mask)
        keep[idx[pick]] = True
        dm = (arr["Dst_2010_plus_MM"] - arr["D0_MM"])
        _lo, hi_m, nb_m = resolve_window(mode, DM_FIT_LO, DM_FIT_HI, DM_NBINS)
        win = keep & (dm > DM_FIT_LO) & (dm < hi_m)
        if int(win.sum()) >= 2 * MIN_CAT:
            r = fit.fit_dm(dm[win], hi=hi_m, nbins=nb_m, signal="dgauss")
            if r.valid:
                vals.append(r.sigma)
    return float(np.mean(vals)) if vals else None


SIGMA = {m: resolution(m) for m in ALL_MODES}
for m, s_ in SIGMA.items():
    print(f"nominal dm resolution {m:5s}: "
          + (f"{s_:.4f} MeV" if s_ else "not measurable"))
print()


def scan(name, fn, nominal, sigma_ref, fmt="{:+.4f}"):
    v0 = nominal[0]
    print(f"{'source':16s} {'variation':26s} {name:>14s} {'shift':>10s}"
          f" {'Barlow':>7s}")
    print(" " * 68 + "sigma")
    rows = []
    for source, label, kwargs in VARIATIONS:
        kwargs = dict(kwargs)
        shift = kwargs.get("shape_shift")
        if shift and SIGMA_REL in shift:
            if not sigma_ref:
                print(f"{source:16s} {label:26s} {'NO SIGMA REF':>14s}")
                rows.append({"source": source, "label": label, "ok": False})
                continue
            kwargs["shape_shift"] = {"sigma": shift[SIGMA_REL] * sigma_ref}
        r = fn(**kwargs)
        if r is None:
            print(f"{source:16s} {label:26s} {'FAILED':>14s}")
            rows.append({"source": source, "label": label, "ok": False})
            continue
        v, s = r
        shift = v - v0
        bsig = asy.barlow_significance(shift, nominal[1], s)
        rows.append({"source": source, "label": label, "ok": True,
                     "value": v, "stat": s, "shift": shift,
                     "barlow_sigma": asy.barlow_sigma(nominal[1], s),
                     "barlow_significance": bsig})
        bs = "  model " if bsig is None else f"{bsig:6.1f}"
        print(f"{source:16s} {label:26s} {fmt.format(v):>14s} "
              f"{shift:>+10.5f}  {bs}")
    return rows


DISTINCT_FIT_EFFECTS = (
    ("background shape", ("powexp background",)),
    ("signal shape", ("single Gaussian signal",
                      "triple Gaussian (LHCb shape)")),
    ("flavour width sharing", ("free per-flavour width",)),
    ("flavour mean sharing", ("shared flavour mean",)),
    ("estimator", ("simultaneous (joint) fit",)),
    ("resolution scale", ("sigma(dm) +10%", "sigma(dm) -10%")),
    ("binning and window", ("window lo +0.5 MeV", "window lo +1.0 MeV",
                            "bin width 0.5 MeV", "bin width 0.2 MeV")),
)


def budget(rows, stat, extra=None, barlow_min=2.0):
    by_source, kept = {}, {}
    for r in rows:
        if not r["ok"]:
            continue
        by_source.setdefault(r["source"], []).append(abs(r["shift"]))
        bs = r.get("barlow_significance")
        if bs is None or bs >= barlow_min:
            kept.setdefault(r["source"], []).append(abs(r["shift"]))
    terms = {source: max(shifts) for source, shifts in by_source.items()}
    terms.update(extra or {})
    total = float(np.sqrt(sum(v ** 2 for v in terms.values())))

    for k, v in terms.items():
        print(f"  {k:42s} {v:.5f}")
    print(f"  {'TOTAL (quadrature, NOMINAL rule)':42s} {total:.5f}")

    kept_terms = {src: max(v) for src, v in kept.items()}
    kept_terms.update(extra or {})
    kept_total = float(np.sqrt(sum(v ** 2 for v in kept_terms.values())))
    dropped = [r["label"] for r in rows if r["ok"]
               and r.get("barlow_significance") is not None
               and r["barlow_significance"] < barlow_min]
    print(f"  {'  cross-check, Barlow-filtered total':42s} {kept_total:.5f}"
          f"   ({len(dropped)} shift(s) below {barlow_min} sigma dropped)")

    shift_by_label = {r["label"]: abs(r["shift"]) for r in rows if r["ok"]}
    distinct = {}
    for name, labels in DISTINCT_FIT_EFFECTS:
        vals = [shift_by_label[l] for l in labels if l in shift_by_label]
        if vals:
            distinct[name] = max(vals)
    if distinct:
        fit_quad = float(np.sqrt(sum(v ** 2 for v in distinct.values())))
        alt = dict(terms)
        alt["fit model"] = fit_quad
        alt_total = float(np.sqrt(sum(v ** 2 for v in alt.values())))
        print(f"  {'  cross-check, distinct fit effects in quad':42s} "
              f"{fit_quad:.5f}")
        print(f"  {'  -> total with that fit-model term':42s} "
              f"{alt_total:.5f}")
    else:
        fit_quad, alt_total, distinct = None, None, {}

    n_ok = sum(1 for r in rows if r["ok"])
    n_noop = sum(1 for r in rows if r["ok"] and r["shift"] == 0.0)
    print(f"  {'variations that converged':42s} {n_ok}/{len(rows)}")
    if n_noop:
        print(f"  {'  of which moved nothing at all (no-ops)':42s} {n_noop}")
    if total > 0:
        print(f"  stat / syst = {stat / total:.2f}  ->  "
              f"{'statistically limited' if stat > total else 'systematically limited'}")
        flips = [name for name, t in
                 (("Barlow-filtered", kept_total),
                  ("distinct fit effects", alt_total))
                 if t and (stat > total) != (stat > t)]
        print(f"  range over the three rules: "
              f"{min(x for x in (total, kept_total, alt_total) if x):.5f} "
              f"to {max(x for x in (total, kept_total, alt_total) if x):.5f}")
    else:
        print("  no variation converged, the budget below is not a bound")

    cross = {"barlow_filtered_total": kept_total,
             "barlow_dropped": dropped,
             "barlow_min_sigma": barlow_min,
             "distinct_fit_effects": distinct,
             "distinct_fit_total": fit_quad,
             "total_with_distinct_fit_term": alt_total,
             "n_no_op_variations": n_noop}
    return terms, total, cross


def pull_gate_bias(mode, stat):
    gate_path = RESULTS_DIR / "pull_gate.json"
    if not gate_path.exists():
        return {}
    gate = json.loads(gate_path.read_text())
    entry = gate.get(mode)
    if not isinstance(entry, dict) or "mean" not in entry:
        return {}
    return {"fit bias (pull mean, nb.13)": abs(entry["mean"]) * stat}


out = {**META, "baseline_selection": BASELINE,
       "dm_resolution_MeV": {m: SIGMA[m] for m in ALL_MODES},
       "estimator": "shared shape with per-flavour mean free "
                    "(fit_categories, share_mean=False)"}

print("=" * 74)
print("Kpi CONTROL ASYMMETRY")
print("=" * 74)
kpi_nom = measure_kpi()
if kpi_nom is None:
    print("Kpi did not fit in every polarity, no control systematic")
    out["kpi"] = None
else:
    k0, ks0 = kpi_nom
    print(f"NOMINAL A_raw(Kpi) = {k0:+.5f} +- {ks0:.5f} (stat)\n")
    kpi_rows = scan("A_raw(Kpi)", measure_kpi, kpi_nom, SIGMA[KPI], "{:+.5f}")
    print("\n=== Kpi systematics table ===")
    extra = pull_gate_bias("KPi", ks0)
    map_path = RESULTS_DIR / "phase8_kpi_map.json"
    if map_path.exists():
        mp = json.loads(map_path.read_text())
        a_int = mp.get("integrated", {}).get("A")
        a_map = mp.get("weighted_average", {}).get("value")
        if a_int is not None and a_map is not None:
            extra["kinematic binning (map-integrated, nb.14)"] = \
                abs(float(a_map) - float(a_int))
        resid = mp.get("polarity_kinematic_residual", {}).get("shift")
        if resid is not None:
            extra["polarity cancellation residual (nb.14)"] = abs(float(resid))
    kpi_terms, kpi_syst, kpi_cross = budget(kpi_rows, ks0, extra)
    print(f"\n*** A_raw(Kpi) = {k0:+.5f} +- {ks0:.5f} (stat) "
          f"+- {kpi_syst:.5f} (syst) ***")
    print(f"    = ({k0 * 100:+.4f} +- {ks0 * 100:.4f} +- {kpi_syst * 100:.4f}) %")
    out["kpi"] = {"nominal": {"value": k0, "stat": ks0},
                  "variations": kpi_rows, "systematics": kpi_terms,
                  "syst_total": kpi_syst, "cross_checks": kpi_cross,
                  "note": "equal-weight polarity average"}

print("\n" + "=" * 74)
print("dACP  (BLINDED)")
print("=" * 74)
nominal, gate_report = dacp_with_gate()
if nominal is None:
    for mode, g in (gate_report.get("composition") or {}).items():
        pct = "n/a" if g["ratio"] is None else f"{g['ratio']:.3%}"
        print(f"  {mode:5s}: N_sig = {g['observed']:,.0f} against a BR-scaled "
              f"Kpi expectation of {g['expected']:,.0f}  ->  {pct} "
              f"(gate needs {g['threshold']:.0%})")
    print(f"\n  dACP REFUSED: {gate_report.get('reason')}")
    out.update({"nominal": None, "variations": [], "systematics": {},
                "syst_total": None, "gate": gate_report,
                "note": "dACP refused by the composition gate"})
else:
    d0, s0 = nominal
    out["gate"] = gate_report
    print(f"NOMINAL blinded dACP = {d0:+.4f} +- {s0:.4f}")
    print("nominal selection: prompt IPchi2<9, fiducial ON, stripping PID\n")
    cp_sigma = [SIGMA[m] for m in CP_MODES if SIGMA[m]]
    rows = scan("blinded dACP", measure_dacp, nominal,
                float(np.mean(cp_sigma)) if cp_sigma else None)
    print("\n=== dACP systematics table ===")
    extra = {}
    gate_path = RESULTS_DIR / "pull_gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        means = [abs(v["mean"]) for k, v in gate.items()
                 if isinstance(v, dict) and "mean" in v and k in CP_MODES]
        if means:
            extra["fit bias (pull mean, nb.13)"] = max(means) * s0
    prod_path = RESULTS_DIR / "prod_results.json"
    if prod_path.exists():
        prod = json.loads(prod_path.read_text())
        di = prod.get("delta_acp_blinded")
        db = prod.get("delta_acp_blinded_binned")
        if di and db:
            extra["kinematic weighting (binned-integrated)"] = \
                abs(db["value"] - di["value"])
    terms, syst, cross = budget(rows, s0, extra)
    print(f"\n*** blinded dACP = {d0:+.4f} +- {s0:.4f} (stat) "
          f"+- {syst:.4f} (syst) ***")
    out.update({"nominal": {"blinded": d0, "stat": s0}, "variations": rows,
                "systematics": terms, "syst_total": syst,
                "cross_checks": cross})

p = RESULTS_DIR / "systematics.json"
p.write_text(json.dumps(out, indent=2, default=float))
print(f"\nwrote {p.name}")
sys.exit(0)
