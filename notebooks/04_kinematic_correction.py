# Validate the (pT, eta) correction on synthetic data with a known injected bias

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np

from charm_acp import asymmetry as asy
from charm_acp.config import PLOTS_DIR, SEED

plt.style.use(mplhep.style.LHCb2)
rng = np.random.default_rng(SEED)

ACP_KK, ACP_PP = 0.005, -0.003
DACP_TRUE = ACP_KK - ACP_PP


def A_nuis(eta):
    return 0.02 * (eta - 3.0)


N_KK, N_PP = 4_000_000, 1_500_000


def simulate(n, acp, eta_mean):
    eta = np.clip(rng.normal(eta_mean, 0.4, n), 2.0, 4.5)
    p_d0 = 0.5 * (1.0 + acp + A_nuis(eta))
    tag = np.where(rng.random(n) < p_d0, 1, -1)
    return eta, tag


eta_kk, tag_kk = simulate(N_KK, ACP_KK, eta_mean=2.8)
eta_pp, tag_pp = simulate(N_PP, ACP_PP, eta_mean=3.3)


def yields(eta, tag, mask=None):
    if mask is None:
        mask = np.ones_like(tag, dtype=bool)
    n_d0 = np.sum((tag == 1) & mask)
    n_d0bar = np.sum((tag == -1) & mask)
    return n_d0, n_d0bar, np.sqrt(n_d0), np.sqrt(n_d0bar)


A_kk, sA_kk = asy.raw_asymmetry(*yields(eta_kk, tag_kk))
A_pp, sA_pp = asy.raw_asymmetry(*yields(eta_pp, tag_pp))
dacp_naive, s_naive = asy.delta_acp(A_kk, sA_kk, A_pp, sA_pp)

ETA_EDGES = np.linspace(2.0, 4.5, 11)
ib_kk = asy.bin_index_2d(np.ones_like(eta_kk), eta_kk, [0, 2], ETA_EDGES)
ib_pp = asy.bin_index_2d(np.ones_like(eta_pp), eta_pp, [0, 2], ETA_EDGES)

nb = len(ETA_EDGES) - 1
cols = {k: np.zeros(nb) for k in
        ("nkk_d0", "nkk_d0b", "skk_d0", "skk_d0b",
         "npp_d0", "npp_d0b", "spp_d0", "spp_d0b")}
per_bin_A_kk, per_bin_A_pp = np.full(nb, np.nan), np.full(nb, np.nan)
for b in range(nb):
    a = yields(eta_kk, tag_kk, ib_kk == b)
    p = yields(eta_pp, tag_pp, ib_pp == b)
    cols["nkk_d0"][b], cols["nkk_d0b"][b], cols["skk_d0"][b], cols["skk_d0b"][b] = a
    cols["npp_d0"][b], cols["npp_d0b"][b], cols["spp_d0"][b], cols["spp_d0b"][b] = p
    per_bin_A_kk[b] = asy.raw_asymmetry(*a)[0]
    per_bin_A_pp[b] = asy.raw_asymmetry(*p)[0]

dacp_corr, s_corr = asy.binned_delta_acp(
    cols["nkk_d0"], cols["nkk_d0b"], cols["skk_d0"], cols["skk_d0b"],
    cols["npp_d0"], cols["npp_d0b"], cols["spp_d0"], cols["spp_d0b"])

print(f"true ΔA_CP                = {DACP_TRUE:+.5f}")
print(f"naive (integrated)        = {dacp_naive:+.5f} ± {s_naive:.5f}   "
      f"bias = {dacp_naive - DACP_TRUE:+.5f}  "
      f"({abs(dacp_naive - DACP_TRUE)/s_naive:.1f}σ off)")
print(f"corrected (η-binned)      = {dacp_corr:+.5f} ± {s_corr:.5f}   "
      f"bias = {dacp_corr - DACP_TRUE:+.5f}  "
      f"({abs(dacp_corr - DACP_TRUE)/s_corr:.1f}σ off)")

centers = 0.5 * (ETA_EDGES[1:] + ETA_EDGES[:-1])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(centers, per_bin_A_kk, "o-", label=r"$A_{\rm raw}(KK)$", color="crimson")
ax1.plot(centers, per_bin_A_pp, "s-", label=r"$A_{\rm raw}(\pi\pi)$", color="navy")
ax1.plot(centers, per_bin_A_kk - per_bin_A_pp, "^--", color="green",
         label=r"per-bin difference")
ax1.axhline(DACP_TRUE, color="gray", ls=":", label=r"true $\Delta A_{CP}$")
ax1.set_xlabel(r"$\eta$ bin")
ax1.set_ylabel("asymmetry")
ax1.legend(fontsize=10)

labels = ["naive\n(integrated)", "corrected\n(η-binned)"]
vals = [dacp_naive, dacp_corr]
errs = [s_naive, s_corr]
ax2.errorbar([0, 1], vals, yerr=errs, fmt="o", ms=10, capsize=6, color="black")
ax2.axhline(DACP_TRUE, color="crimson", ls="--", label=r"true $\Delta A_{CP}$ = %.3f" % DACP_TRUE)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(labels)
ax2.set_xlim(-0.5, 1.5)
ax2.set_ylabel(r"$\Delta A_{CP}$")
ax2.legend(fontsize=10)

fig.tight_layout()
fig.savefig(PLOTS_DIR / "phase8_kinematic_correction_demo.png", dpi=150, bbox_inches="tight")
print("wrote phase8_kinematic_correction_demo.png")
