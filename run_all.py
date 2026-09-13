# runs the whole chain in order, stops at the first failure

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from charm_acp.config import PRODUCTION, production_files

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
NB = HERE / "notebooks"
RESULTS = HERE / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true",
                help="skip per-file validation and run 300 toys instead of 1000")
args = ap.parse_args()

PIPELINE_ARGS = ["--tag", "prod"]

STAGES = [
    ("recon", NB / "06_file_recon.py", []),
    *[(f"validate {p.name}", NB / "05_validate_production_file.py", [str(p)])
      for p, _pol in production_files()],
    ("pipeline", NB / "09_production_pipeline.py", PIPELINE_ARGS),
    ("pull gate", NB / "13_asymmetry_pull_gate.py",
     ["--toys", "300" if args.fast else "1000"]),
    ("bootstrap", NB / "18_bootstrap_errors.py",
     ["--boots", "50" if args.fast else "200"]),
    ("kpi map", NB / "14_phase8_kpi_map.py", []),
    ("run stability", NB / "19_run_period_split.py", []),
    ("referee checks", NB / "20_physicist_checks.py", []),
    ("systematics", NB / "15_systematics.py", []),
]
if args.fast:
    STAGES = [s for s in STAGES if not s[0].startswith("validate")]

RESULTS.mkdir(exist_ok=True)
LOG = (RESULTS / "run_all_log.txt").open("w", encoding="utf-8")


def emit(line=""):
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


t_all = time.time()
for name, script, extra in STAGES:
    emit(f"\n{'=' * 74}\n>>> STAGE: {name}  ({script.name})\n{'=' * 74}")
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(script), *extra], cwd=HERE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        emit(line.rstrip("\n"))
    rc = proc.wait()
    dt = time.time() - t0
    if rc != 0:
        emit(f"\n!!! stage '{name}' FAILED (exit {rc}) after {dt:.0f}s, stopping")
        LOG.close()
        sys.exit(rc)
    emit(f"--- stage '{name}' OK in {dt:.0f}s")

emit(f"\n{'=' * 74}\n>>> SUMMARY  (total {time.time() - t_all:.0f}s)\n{'=' * 74}")

prod = json.loads((RESULTS / "prod_results.json").read_text())
syst = json.loads((RESULTS / "systematics.json").read_text())
kpi_map = json.loads((RESULTS / "phase8_kpi_map.json").read_text())

pv = prod.get("provenance") or {}
if pv.get("git_commit"):
    dirty = " (WORKING TREE DIRTY)" if pv.get("git_dirty") else ""
    emit(f"code           : v{pv.get('code_version')} @ "
         f"{pv['git_commit'][:12]}{dirty}")
emit(f"production     : {prod['production']} (layout {prod['layout']})")
emit(f"files          : " + ", ".join(f"{n} ({p})" for n, p in prod["files"]))
lum = prod["luminosity_pb"]
emit("luminosity     : " + "  ".join(f"{k} {v:.3f}/pb"
                                      for k, v in lum.items()))
gofs = []
for key, f in (prod.get("fits") or {}).items():
    if not f or not f.get("gof"):
        continue
    g = f["gof"].get("combined") or {}
    if g.get("chi2_ndf") is not None:
        gofs.append((key, g["chi2_ndf"], g.get("p_value"),
                     g.get("max_abs_pull")))
if gofs:
    emit("fit quality    : " + "  ".join(
        f"{k} chi2/ndf={r:.2f}" for k, r, _p, _mp in gofs))
    worst = max(gofs, key=lambda t: t[1])
    if worst[1] > 2.0:
        emit(f"                 WORST {worst[0]}: chi2/ndf = {worst[1]:.2f}, "
             f"max|pull| = {worst[3]:.1f}")

a = prod["A_raw"]
kpi = a.get("KPi") or {}
for pol, v in kpi.items():
    if isinstance(v, dict) and "value" in v and pol != "average":
        emit(f"A_raw(KPi {pol:8s}): {v['value']:+.4f} +- {v['error']:.4f}")
if "average" in kpi:
    v = kpi["average"]
    k_syst = (syst.get("kpi") or {}).get("syst_total")
    if k_syst:
        emit(f"A_raw(KPi average) : {v['value']:+.4f} +- {v['error']:.4f} "
             f"(stat) +- {k_syst:.4f} (syst)")
        emit(f"                   = ({v['value'] * 100:+.2f} "
             f"+- {v['error'] * 100:.2f} +- {k_syst * 100:.2f}) %")
    else:
        emit(f"A_raw(KPi average) : {v['value']:+.4f} +- {v['error']:.4f} "
             "(stat only)")
if "polarity_split_sigma" in kpi:
    emit(f"polarity split     : {kpi['polarity_split_sigma']:.1f} sigma "
          "(MagDown vs MagUp)")
for mode in ("KK", "PiPi"):
    v = a.get(mode)
    if v and v.get("blinded_average"):
        b = v["blinded_average"]
        emit(f"A_raw({mode:4s}) BLINDED : {b['value']:+.4f} +- {b['error']:.4f}")
emit(f"null test      : Kpi map chi2/ndf vs flat = "
      f"{kpi_map['flatness']['chi2']:.0f}/{kpi_map['flatness']['ndf']}")
if prod.get("delta_acp_blinded"):
    d = prod["delta_acp_blinded"]
    emit(f"dACP (blinded) : {d['value']:+.4f} +- {d['error']:.4f} "
          f"(stat, integrated)")
if prod.get("delta_acp_blinded_binned"):
    d = prod["delta_acp_blinded_binned"]
    emit(f"               : {d['value']:+.4f} +- {d['error']:.4f} "
          f"(stat, (pT,eta)-binned, {d['n_bins']} bins)")
gate = syst.get("gate") or {}
comp = (prod.get("gates") or {}).get("composition") or {}
if syst.get("nominal") and prod.get("delta_acp_blinded"):
    n = syst["nominal"]
    emit(f"FINAL (blinded): dACP = {n['blinded']:+.4f} +- {n['stat']:.4f} "
          f"(stat) +- {syst['syst_total']:.4f} (syst)   "
          f"[method: {prod.get('nominal_method')}]")
    emit("\nresult stays blinded until notebooks/17_unblind.py is run.")
else:
    emit("FINAL          : no dACP, composition gate refused KK and PiPi")
    for mode, g in comp.items():
        if g and not g.get("passed"):
            pct = "n/a" if g.get("ratio") is None else f"{g['ratio']:.3%}"
            emit(f"                 {mode:5s}: N_sig = {g['observed']:,.0f} "
                 f"vs BR-scaled expectation {g['expected']:,.0f}  = {pct} "
                 f"(gate needs {g['threshold']:.0%})")
    if gate.get("reason"):
        emit(f"                 systematics: {gate['reason']}")
    if syst.get("nominal"):
        emit("                 WARNING: systematics.json still has a dACP nominal")

LOG.close()
