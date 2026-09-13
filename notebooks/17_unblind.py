# takes the blinding off, run once at the end and never again

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep
import numpy as np

from charm_acp import asymmetry as asy
from charm_acp.config import PLOTS_DIR, RESULTS_DIR, production_spec

plt.style.use(mplhep.style.LHCb2)

LHCB = {"value": -18.2e-4, "error": 3.3e-4,
        "label": r"LHCb 2019 $\pi$-tagged (PRL 122, 211803)"}
LHCB_COMB = {"value": -15.4e-4, "error": 2.9e-4,
             "label": r"LHCb 2019 combination (Run 1+2)"}

ap = argparse.ArgumentParser()
ap.add_argument("--i-understand-this-is-final", action="store_true")
args = ap.parse_args()

marker = RESULTS_DIR / "UNBLINDED.json"
if marker.exists():
    try:
        prev = json.loads(marker.read_text(encoding="utf-8-sig"))
        detail = (f"on {prev['unblinded_utc']}, the result is "
                  f"dACP = {prev['delta_acp']:+.4f} +- {prev['stat']:.4f} "
                  f"(stat) +- {prev['syst']:.4f} (syst)")
    except Exception:
        detail = "and the marker file is unreadable"
    raise SystemExit(f"already unblinded {detail}")

if not args.i_understand_this_is_final:
    raise SystemExit(
        "unblinding is one way, freeze selection, fit model and systematics "
        "first, then rerun with --i-understand-this-is-final")

syst = json.loads((RESULTS_DIR / "systematics.json").read_text())
prod = json.loads((RESULTS_DIR / "prod_results.json").read_text())

live = production_spec()[0]
if syst.get("production") != prod.get("production") \
        or syst.get("production") != live:
    raise SystemExit(
        f"production mismatch: systematics.json={syst.get('production')}, "
        f"prod_results.json={prod.get('production')}, live={live}. "
        "Unblinding with the wrong offset would give a wrong final number, "
        "regenerate the results under one production first.")

if not syst.get("nominal"):
    raise SystemExit("no blinded dACP exists in systematics.json, nothing to "
                     "unblind at these statistics")

blinded = syst["nominal"]["blinded"]
stat = syst["nominal"]["stat"]
s_tot = syst["syst_total"]

value = asy.unblind(blinded)
offset = blinded - value

err_comb = float(np.hypot(stat, s_tot))
diff = value - LHCB["value"]
sig = diff / float(np.hypot(err_comb, LHCB["error"]))
zero_sig = value / err_comb

print("=" * 66)
print("UNBLINDED RESULT")
print("=" * 66)
print(f"  dACP = {value:+.4f} +- {stat:.4f} (stat) +- {s_tot:.4f} (syst)")
print(f"       = {value:+.4f} +- {err_comb:.4f} (total)")
print(f"  hidden offset was {offset:+.5f}")
print(f"  vs zero          : {zero_sig:+.1f} sigma")
print(f"  vs LHCb pi-tag   : {LHCB['value']:+.5f} +- {LHCB['error']:.5f}  "
      f"-> difference {diff:+.4f} = {sig:+.1f} sigma")
print(f"  context, combo   : {LHCB_COMB['value']:+.5f} +- "
      f"{LHCB_COMB['error']:.5f}  (combination, for reference)")

record = {
    "unblinded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "delta_acp": value, "stat": stat, "syst": s_tot, "total": err_comb,
    "offset_removed": offset,
    "significance_vs_zero": float(zero_sig),
    "significance_vs_lhcb": float(sig),
    "lhcb_reference": LHCB,
    "inputs": {"files": prod["files"], "luminosity_pb": prod["luminosity_pb"],
               "systematics": syst["systematics"]},
}
marker.write_text(json.dumps(record, indent=2))
print(f"\nwrote {marker.name}  (its existence blocks any second unblinding)")

fig, ax = plt.subplots(figsize=(8, 4.5))
pts = [("this analysis\n(open data)", value, err_comb, "crimson"),
       (LHCB["label"] + "\n(5.9 fb$^{-1}$)", LHCB["value"], LHCB["error"],
        "navy"),
       (LHCB_COMB["label"], LHCB_COMB["value"], LHCB_COMB["error"], "gray")]
for i, (label, v, e, c) in enumerate(pts):
    ax.errorbar([v], [i], xerr=[e], fmt="o", ms=8, capsize=5, color=c, lw=2)
ax.axvline(0, color="gray", lw=1, ls=":")
ax.set_yticks(range(len(pts)))
ax.set_yticklabels([p[0] for p in pts], fontsize=10)
ax.set_ylim(-0.6, len(pts) - 0.4)
ax.set_xlabel(r"$\Delta A_{CP}$")
ax.set_title("Unblinded result vs the LHCb observation", fontsize=12)
fig.tight_layout()
out = PLOTS_DIR / "final_dacp_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out.name}")
