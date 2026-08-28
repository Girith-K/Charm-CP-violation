# End to end dACP rehearsal on the v2 test production

import os

# v2 only notebook, so pin the production before charm_acp is imported: the
# blinding salt and offset scale must be v2's, not the v3 default in config
os.environ["CHARM_ACP_PRODUCTION"] = "v2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uproot
from scipy.stats import norm

from charm_acp import asymmetry as asy
from charm_acp import fitting as fit
from charm_acp import selection as sel
from charm_acp.config import DM_FIT_HI, DM_FIT_LO, PLOTS_DIR, TESTPROD_V2

plt.style.use(mplhep.style.LHCb2)

MIN_EVENTS_FOR_FIT = 200

f = uproot.open(TESTPROD_V2)
results = {}

for mode in ["KK", "PiPi", "KPi"]:
    t = f[sel.trees(mode, "v2")[0][0]]
    arr = t.arrays(sel.branches(mode), library="np")
    mask, table = sel.cutflow(arr, mode)

    print(f"\n=== {mode}: cutflow ===")
    for name, n, eff in table:
        print(f"  {name:18s} {n:6d}  ({eff:6.1%} of previous)")

    cats = sel.split_categories(arr, mask, mode)
    dm = arr["Dst_2010_plus_MM"] - arr["D0_MM"]
    results[mode] = {"arr": arr, "cats": cats, "dm": dm,
                     "n_d0": int(cats["D0"].sum()), "n_d0bar": int(cats["D0bar"].sum())}
    if mode == "KPi":
        print(f"  tagged: D0={results[mode]['n_d0']}, D0bar={results[mode]['n_d0bar']}")
    else:
        print(f"  tagged: total={results[mode]['n_d0'] + results[mode]['n_d0bar']} "
              "(flavour split not printed, blinding)")

print("\n=== per-category Δm fits ===")
fit_out = {}
for mode, r in results.items():
    fit_out[mode] = {}
    for cat_name, cat_mask in r["cats"].items():
        dm = r["dm"][cat_mask]
        dm = dm[(dm > DM_FIT_LO) & (dm < DM_FIT_HI)]
        if len(dm) < MIN_EVENTS_FOR_FIT:
            print(f"  {mode}/{cat_name}: {len(dm)} events, SKIP fit (test slice too small)")
            fit_out[mode][cat_name] = None
            continue
        res = fit.fit_dm(dm)
        fit_out[mode][cat_name] = res
        if mode == "KPi":
            print(f"  {mode}/{cat_name}: N_sig = {res.n_sig:7.1f} +- {res.n_sig_err:5.1f}  "
                  f"(mu={res.mu:.2f}, sigma={res.sigma:.3f}, valid={res.valid})")
        else:
            print(f"  {mode}/{cat_name}: fit done (mu={res.mu:.2f}, "
                  f"sigma={res.sigma:.3f}, valid={res.valid}, "
                  "yield not printed, blinding)")

print("\n=== asymmetries ===")
kpi = fit_out["KPi"]
if kpi["D0"] and kpi["D0bar"]:
    A, sA = asy.raw_asymmetry(kpi["D0"].n_sig, kpi["D0bar"].n_sig,
                              kpi["D0"].n_sig_err, kpi["D0bar"].n_sig_err)
    print(f"  A_raw(KPi control) = {A:+.4f} +- {sA:.4f}")
    print("  (null test: pure nuisance level; consistent with ~0-1% expected)")

have_signal = fit_out["KK"]["D0"] and fit_out["KK"]["D0bar"] \
    and fit_out["PiPi"]["D0"] and fit_out["PiPi"]["D0bar"]
if have_signal:
    kk, pp = fit_out["KK"], fit_out["PiPi"]
    Akk, skk = asy.raw_asymmetry(kk["D0"].n_sig, kk["D0bar"].n_sig,
                                 kk["D0"].n_sig_err, kk["D0bar"].n_sig_err)
    App, spp = asy.raw_asymmetry(pp["D0"].n_sig, pp["D0bar"].n_sig,
                                 pp["D0"].n_sig_err, pp["D0bar"].n_sig_err)
    d, sd = asy.delta_acp(Akk, skk, App, spp)
    bd, bsd = asy.blind_delta_acp(d, sd)
    print(f"  BLINDED dACP = {bd:+.4f} +- {bsd:.4f}   <-- offset stays until Phase 10")
else:
    print("  dACP: waiting on full production for KK/PiPi statistics "
          "(machinery in place: asy.blind_delta_acp).")

for cat_name in ["D0", "D0bar"]:
    res = kpi[cat_name]
    if res is None:
        continue
    centers = 0.5 * (res.edges[1:] + res.edges[:-1])
    width = res.edges[1] - res.edges[0]
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax.errorbar(centers, res.counts, yerr=np.sqrt(res.counts), fmt="k.", ms=4, label="Data")
    ax.plot(centers, res.pred, "-", lw=2, label="Total fit")
    bkg = res.n_bkg * np.diff(fit._bkg_cdf(res.edges, res.a, res.b))
    ax.plot(centers, bkg, "--", lw=1.5, label="Background")
    tag = r"$\pi_s^+$ (D$^0$)" if cat_name == "D0" else r"$\pi_s^-$ ($\bar{D}^0$)"
    ax.set_title(tag, fontsize=11)
    ax.set_ylabel(f"Candidates / {width:.2f} MeV")
    ax.legend()
    axp.axhline(0, color="gray", lw=1)
    axp.bar(centers, res.pulls, width=width, color="steelblue")
    axp.set_ylim(-5, 5)
    axp.set_xlabel(r"$\Delta m$ [MeV]")
    axp.set_ylabel("Pull")
    out = PLOTS_DIR / f"testprod_v2_kpi_fit_{cat_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out.name}")

print("\n=== toy pull study (300 toys, Kpi-like stats) ===")
ref = kpi["D0"]
true_pars = dict(n_sig=ref.n_sig, n_bkg=ref.n_bkg, mu=ref.mu,
                 sigma=ref.sigma, a=ref.a, b=ref.b)
study = fit.toy_pull_study(true_pars, n_toys=300)
print(f"  pull mean  = {study['mean']:+.3f} +- {study['mean_err']:.3f}   (want ~0)")
print(f"  pull width = {study['width']:.3f} +- {study['width_err']:.3f}   (want ~1)")
print(f"  failed fits: {study['n_failed']}/{study['n_toys']}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(study["pulls"], bins=30, range=(-4, 4), histtype="step", lw=1.8, color="navy")
x = np.linspace(-4, 4, 200)
n_ok = len(study["pulls"])
ax.plot(x, n_ok * (8 / 30) * norm.pdf(x), "--", color="crimson", lw=1.5,
        label="unit Gaussian")
ax.set_xlabel(r"pull = $(N_{fit} - N_{true})\,/\,\sigma_{fit}$")
ax.set_ylabel("toys")
ax.legend()
fig.savefig(PLOTS_DIR / "toy_pulls_dm.png", dpi=150, bbox_inches="tight")
print("wrote toy_pulls_dm.png")
