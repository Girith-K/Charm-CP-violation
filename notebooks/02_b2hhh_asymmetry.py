# warm up, raw charge asymmetry in B to KKK from the B2HHH open data

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import norm

from charm_acp import kinematics as kin
from charm_acp.asymmetry import raw_asymmetry
from charm_acp.config import DATA_DIR, PLOTS_DIR

PROB_K_MIN = 0.70
PROB_PI_MAX = 0.50
FIT_LO, FIT_HI, NBINS = 5150.0, 5450.0, 100

branches = [f"H{i}_{v}" for i in (1, 2, 3)
            for v in ("PX", "PY", "PZ", "ProbK", "ProbPi", "Charge", "isMuon")]
tree = uproot.open(DATA_DIR / "archive" / "B2HHH_MagnetUp.root")["DecayTree"]
a = tree.arrays(branches, library="np")
n0 = len(a["H1_PX"])
print(f"loaded {n0:,} candidates")


def track_is_kaon(i):
    return (
        (a[f"H{i}_ProbK"] > PROB_K_MIN)
        & (a[f"H{i}_ProbPi"] < PROB_PI_MAX)
        & (a[f"H{i}_isMuon"] == 0)
    )

masks = [
    ("no muons",  (a["H1_isMuon"] == 0) & (a["H2_isMuon"] == 0) & (a["H3_isMuon"] == 0)),
    ("3x kaon PID", track_is_kaon(1) & track_is_kaon(2) & track_is_kaon(3)),
]
sel = np.ones(n0, dtype=bool)
print("cutflow:")
print(f"  {'(all)':16s} {n0:>9,}")
for name, mk in masks:
    sel &= mk
    print(f"  {name:16s} {sel.sum():>9,}")

p4 = []
for i in (1, 2, 3):
    px, py, pz = a[f"H{i}_PX"][sel], a[f"H{i}_PY"][sel], a[f"H{i}_PZ"][sel]
    p4.append((px, py, pz, kin.energy(px, py, pz, kin.M_K)))
m_kkk = kin.mass_from_four_momentum(*kin.sum_four_momentum(p4))
print(f"m(KKK): mean {m_kkk.mean():.0f} MeV, "
      f"in-window fraction {((m_kkk > FIT_LO) & (m_kkk < FIT_HI)).mean():.2%}")

q_b = (a["H1_Charge"] + a["H2_Charge"] + a["H3_Charge"])[sel]
assert set(np.unique(q_b)) <= {-1, 1}, "unexpected total charge"

W = FIT_HI - FIT_LO


def bkg_cdf(x, tau):
    return (1.0 - np.exp(-(x - FIT_LO) / tau)) / (1.0 - np.exp(-W / tau))


def model(x, n_sig, n_bkg, mu, sigma, tau):
    return n_sig * norm.cdf(x, mu, sigma) + n_bkg * bkg_cdf(x, tau)


def fit_sample(mass, label):
    counts, edges = np.histogram(mass, bins=NBINS, range=(FIT_LO, FIT_HI))
    m = Minuit(ExtendedBinnedNLL(counts, edges, model),
               n_sig=counts.sum() * 0.5, n_bkg=counts.sum() * 0.5,
               mu=5279.0, sigma=20.0, tau=200.0)
    m.limits["n_sig", "n_bkg"] = (0, None)
    m.limits["mu"] = (5250, 5310)
    m.limits["sigma"] = (5, 60)
    m.limits["tau"] = (30, 3000)
    m.migrad()
    m.hesse()
    assert m.valid, f"{label}: fit did not converge"
    print(f"  {label}: N_sig = {m.values['n_sig']:8.0f} +- {m.errors['n_sig']:.0f}"
          f"   mu = {m.values['mu']:.1f}  sigma = {m.values['sigma']:.1f}")
    return m.values["n_sig"], m.errors["n_sig"], m, counts, edges


print("fits (Gaussian + exponential bkg, extended binned NLL):")
res = {}
for q, label in [(-1, "B-"), (+1, "B+")]:
    res[q] = fit_sample(m_kkk[q_b == q], label)

nM, sM = res[-1][0], res[-1][1]
nP, sP = res[+1][0], res[+1][1]
A, sigA = raw_asymmetry(nM, nP, sM, sP)
print(f"\nA_raw(KKK) = {A:+.4f} +- {sigA:.4f}   (stat only)")
print("published LHCb A_CP(B->KKK) = -0.036 +- 0.004")

plt.style.use(mplhep.style.LHCb2)
fig, axes = plt.subplots(
    2, 2, figsize=(14, 7), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.18},
)
for col, (q, title) in enumerate([(-1, r"$B^-$"), (+1, r"$B^+$")]):
    _, _, m, counts, edges = res[q]
    centers = 0.5 * (edges[1:] + edges[:-1])
    pred = np.diff(model(edges, *m.values))
    pred_bkg = np.diff(m.values["n_bkg"] * bkg_cdf(edges, m.values["tau"]))
    pulls = (counts - pred) / np.sqrt(np.where(pred > 0, pred, 1.0))

    ax, axp = axes[0][col], axes[1][col]
    ax.errorbar(centers, counts, yerr=np.sqrt(counts), fmt="k.", ms=3, label="Data")
    ax.plot(centers, pred, "-", lw=2, label="Total fit")
    ax.plot(centers, pred_bkg, "--", lw=1.5, label="Background")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(f"Candidates / {W / NBINS:.0f} MeV" if col == 0 else "")
    ax.legend(fontsize=10)
    axp.axhline(0, color="gray", lw=1)
    axp.bar(centers, pulls, width=W / NBINS, color="steelblue")
    axp.set_ylim(-5, 5)
    axp.set_xlabel(r"$m(K^+K^-K^\pm)$ [MeV]")
    axp.set_ylabel("Pull" if col == 0 else "")

fig.savefig(PLOTS_DIR / "b2hhh_kkk_asymmetry.png", dpi=150, bbox_inches="tight")
print("wrote plots/b2hhh_kkk_asymmetry.png")
