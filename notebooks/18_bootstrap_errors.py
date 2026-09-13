# bootstrap check on the stat errors from the fits

from __future__ import annotations

import argparse

import numpy as np

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit
from charm_acp import selection as sel
from charm_acp.config import (DM_FIT_HI, DM_FIT_LO, SEED, production_files,
                              production_spec)

ap = argparse.ArgumentParser()
ap.add_argument("--boots", type=int, default=200)
args = ap.parse_args()

CUTS = sel.NOMINAL_CUTS
LAYOUT = production_spec()[1]["layout"]
SIGNAL = "dgauss"
rng = np.random.default_rng(SEED + 18)


def mode_window(mode):
    hi = 155.0 if (LAYOUT == "v3" and mode != "KPi") else DM_FIT_HI
    return hi, int(round((hi - DM_FIT_LO) / 0.25))


any_mode = False
for mode in ("KK", "PiPi", "KPi"):
    HI, NBINS = mode_window(mode)
    dm_l, tag_l, run_l, evt_l, v_l = [], [], [], [], []
    for path, pol in production_files():
        for arr in sel.iter_mode(path, mode, LAYOUT):
            mask, _ = sel.cutflow(arr, mode, CUTS)
            dm_l.append((arr["Dst_2010_plus_MM"] - arr["D0_MM"])[mask])
            tag_l.append(sel.flavour_tag(arr)[mask])
            run_l.append(arr["runNumber"][mask])
            evt_l.append(arr["eventNumber"][mask])
            v_l.append((arr["D0_ENDVERTEX_CHI2"]
                        / np.maximum(arr["D0_ENDVERTEX_NDOF"], 1))[mask])
    dm, tag = np.concatenate(dm_l), np.concatenate(tag_l)
    pick = sel.single_candidate_pick(np.concatenate(run_l),
                                     np.concatenate(evt_l),
                                     np.concatenate(v_l))
    dm, tag = dm[pick], tag[pick]
    win = (dm > DM_FIT_LO) & (dm < HI)
    dm, tag = dm[win], tag[win]
    n = len(dm)
    if n < 60 or min(int((tag > 0).sum()), int((tag < 0).sum())) < 30:
        print(f"{mode}: below statistics threshold, skipped")
        continue
    any_mode = True

    by_cat = {"D0": dm[tag > 0], "D0bar": dm[tag < 0]}
    fits, _ = fit.fit_categories(by_cat, share_shape=True, hi=HI, nbins=NBINS,
                                 signal=SIGNAL)
    if not all(fit.significant(r) for r in fits.values()):
        print(f"{mode}: signal below 3 sigma, fits on it are not used, skipped")
        continue
    A0, sA0 = asy.raw_asymmetry(fits["D0"].n_sig, fits["D0bar"].n_sig,
                                fits["D0"].n_sig_err, fits["D0bar"].n_sig_err)

    vals = []
    for _ in range(args.boots):
        idx = rng.integers(0, n, n)
        b_dm, b_tag = dm[idx], tag[idx]
        cats = {"D0": b_dm[b_tag > 0], "D0bar": b_dm[b_tag < 0]}
        if min(len(v) for v in cats.values()) < 30:
            continue
        try:
            f, _ = fit.fit_categories(cats, share_shape=True, hi=HI,
                                      nbins=NBINS, signal=SIGNAL)
        except Exception:
            continue
        if not all(r.valid and r.n_sig_err > 0 for r in f.values()):
            continue
        A, _ = asy.raw_asymmetry(f["D0"].n_sig, f["D0bar"].n_sig,
                                 f["D0"].n_sig_err, f["D0bar"].n_sig_err)
        if np.isfinite(A):
            vals.append(A)

    vals = np.asarray(vals)
    if len(vals) < max(10, args.boots // 4):
        print(f"{mode:5s}: only {len(vals)}/{args.boots} usable replicas, "
              "no verdict")
        continue
    ratio = float(vals.std() / sA0)
    shown = (f"A_raw = {A0:+.4f} +- {sA0:.4f} (fit, polarity pooled)"
             if mode == "KPi"
             else f"sigma_A = {sA0:.4f} (fit, central value blinded)")
    print(f"{mode:5s}: n={n:,}  {shown}   "
          f"bootstrap spread = {vals.std():.4f}  ({len(vals)}/{args.boots} ok)")
    print(f"       bootstrap/fit error ratio = {ratio:.3f}  "
          f"{'OK (agree within 15%)' if abs(ratio - 1) < 0.15 else 'DISAGREE'}")

if not any_mode:
    raise SystemExit("no mode had enough statistics to bootstrap")
