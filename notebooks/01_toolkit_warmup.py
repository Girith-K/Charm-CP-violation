# warm up, fits the D0 mass peak on the masterclass data

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import norm

from charm_acp.config import DATA_DIR, PLOTS_DIR
from charm_acp.kinematics import M_D0

tree = uproot.open(DATA_DIR / "archive" / "MasterclassData.root")["DecayTree"]
arr = tree.arrays(["D0_MM", "D0_TAU", "D0_PT", "D0_MINIPCHI2"], library="np")
mass = arr["D0_MM"]
print(f"loaded {len(mass)} candidates; "
      f"mass range [{mass.min():.1f}, {mass.max():.1f}] MeV")

LO, HI, NBINS = 1815.0, 1915.0, 100
counts, edges = np.histogram(mass, bins=NBINS, range=(LO, HI))

XC = 0.5 * (LO + HI)
W = HI - LO


def bkg_cdf(x, slope):
    return ((x - LO) + 0.5 * slope * ((x - XC) ** 2 - (LO - XC) ** 2)) / W


def model(x, n_sig, n_bkg, mu, sigma, slope):
    return n_sig * norm.cdf(x, mu, sigma) + n_bkg * bkg_cdf(x, slope)


cost = ExtendedBinnedNLL(counts, edges, model)
m = Minuit(cost, n_sig=50_000, n_bkg=40_000, mu=1865.0, sigma=8.0, slope=0.0)
m.limits["n_sig", "n_bkg"] = (0, None)
m.limits["sigma"] = (0.1, 50)
m.limits["mu"] = (LO, HI)
m.migrad()
m.hesse()

assert m.valid, "fit did not converge, inspect m.fmin"
n_sig, n_sig_err = m.values["n_sig"], m.errors["n_sig"]
mu_fit, sigma_fit = m.values["mu"], m.values["sigma"]
print(f"fit valid: N_sig = {n_sig:.0f} +- {n_sig_err:.0f}")
print(f"           mu    = {mu_fit:.2f} MeV   (PDG D0: {M_D0:.2f} MeV)")
print(f"           sigma = {sigma_fit:.2f} MeV  (detector resolution)")

plt.style.use(mplhep.style.LHCb2)
centers = 0.5 * (edges[1:] + edges[:-1])
pred = np.diff(model(edges, *m.values))
pred_sig = np.diff(n_sig * norm.cdf(edges, mu_fit, sigma_fit))
pred_bkg = pred - pred_sig
pulls = (counts - pred) / np.sqrt(np.where(pred > 0, pred, 1.0))

fig, (ax, axp) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)
ax.errorbar(centers, counts, yerr=np.sqrt(counts), fmt="k.", ms=4, label="Data")
ax.plot(centers, pred, "-", lw=2, label="Total fit")
ax.plot(centers, pred_bkg, "--", lw=1.5, label="Background")
ax.set_ylabel(f"Candidates / {W / NBINS:.0f} MeV")
ax.legend()

axp.axhline(0, color="gray", lw=1)
axp.bar(centers, pulls, width=W / NBINS, color="steelblue")
axp.set_ylim(-5, 5)
axp.set_xlabel(r"$m(K^-\pi^+)$ [MeV]")
axp.set_ylabel("Pull")

fig.savefig(PLOTS_DIR / "warmup_d0_kpi_fit.png", dpi=150, bbox_inches="tight")
print("wrote plots/warmup_d0_kpi_fit.png")
