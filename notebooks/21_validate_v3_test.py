# checks the v3 test files, both polarities

from __future__ import annotations

import numpy as np
import uproot

from charm_acp import kinematics as kin
from charm_acp.config import BR, DATA_DIR
from charm_acp.kinematics import DM_DSTAR, M_D0, M_K

TREES = {
    "DstpD02KK":   dict(d0="D0",  h1="Kplus",  h2="Kminus",  soft="piplus",
                        kaons=("Kplus", "Kminus"), expect_dst=+413),
    "DstmD02KK":   dict(d0="D_0", h1="Kplus",  h2="Kminus",  soft="piplus",
                        kaons=("Kplus", "Kminus"), expect_dst=-413),
    "DstpD02PiPi": dict(d0="D0",  h1="piplus", h2="piminus", soft="piplus_0",
                        kaons=(), expect_dst=+413),
    "DstmD02PiPi": dict(d0="D_0", h1="piplus", h2="piminus", soft="piplus_0",
                        kaons=(), expect_dst=-413),
    "DstD02KPi":   dict(d0="D0",  h1="Kminus", h2="piplus",  soft="piplus_0",
                        kaons=("Kminus",), expect_dst=0),
}

FILES = [("MagDown (job0)", DATA_DIR / "archive" / "v3test_job0_magdown.root"),
         ("MagUp   (job1)", DATA_DIR / "archive" / "v3test_job1_magup.root")]

for _label, _path in FILES:
    if "archive" not in _path.parts:
        raise SystemExit("21 runs only on the archived test jobs, its "
                         "per-charge entry prints would leak on production "
                         "chunks")

overall_ok = True
n_checks = 0


def check(label, cond, detail=""):
    global overall_ok, n_checks
    n_checks += 1
    if not cond:
        overall_ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}  {detail}")


def sideband_signal(m, mask):
    mm = m[mask]
    d = np.abs(mm - M_D0)
    sig = int(np.sum(d < 20.0))
    sb = int(np.sum((d > 35.0) & (d < 75.0)))
    return sig - 0.5 * sb, np.sqrt(max(sig + 0.25 * sb, 1))


for pol_label, path in FILES:
    if not path.exists():
        print(f"missing {path}, skipped")
        continue
    print(f"\n{'=' * 70}\n{pol_label}: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)\n{'=' * 70}")
    f = uproot.open(path)

    try:
        lumi = float(np.sum(f["GetIntegratedLuminosity/LumiTuple"]
                            ["IntegratedLuminosity"].array(library="np")))
        print(f"  [INFO] integrated luminosity: {lumi:.4f} /pb")
    except Exception:
        lumi = None
        print("  [INFO] no LumiTuple found (checked GetIntegratedLuminosity/LumiTuple)")

    data = {}
    for tname, s in TREES.items():
        tpath = f"{tname}/DecayTree"
        try:
            t = f[tpath]
        except Exception:
            check(f"{tname}: tree present", False, f"'{tpath}' NOT FOUND")
            continue
        keys = set(t.keys())
        live = sum(1 for _, b in t.items() if getattr(b, "num_baskets", 0) > 0)
        check(f"{tname}: tree present, branches live",
              t.num_entries > 0 and live == len(keys),
              f"{t.num_entries:,} entries, {live}/{len(keys)} live")

        d0 = s["d0"]
        need = [f"Dst_2010_plus_ID", f"Dst_2010_plus_MM", f"{d0}_MM",
                f"{s['soft']}_ID"]
        if not all(k in keys for k in need):
            check(f"{tname}: core branches named as expected", False,
                  f"missing {[k for k in need if k not in keys]}")
            continue
        arr = t.arrays(need, library="np")
        dst = arr["Dst_2010_plus_ID"]
        dm = arr["Dst_2010_plus_MM"] - arr[f"{d0}_MM"]
        data[tname] = dict(m=arr[f"{d0}_MM"], dm=dm, n=len(dm), keys=keys,
                           tree=t, dst=dst)

        n_p, n_m = int(np.sum(dst == 413)), int(np.sum(dst == -413))
        if s["expect_dst"] == +413:
            check(f"{tname}: PURE D*+ (the v1 check)", n_m == 0 and n_p > 0,
                  f"D*+={n_p:,}  D*-={n_m:,}")
        elif s["expect_dst"] == -413:
            check(f"{tname}: PURE D*- (the D-bar-form mechanism)",
                  n_p == 0 and n_m > 0, f"D*+={n_p:,}  D*-={n_m:,}")
        else:
            check(f"{tname}: BOTH D* charges", n_p > 0 and n_m > 0,
                  f"D*+={n_p:,}  D*-={n_m:,}  (ratio {n_p / max(n_m, 1):.2f})")

        soft_id = arr[f"{s['soft']}_ID"]
        check(f"{tname}: soft-pion charge == D* charge",
              bool(np.all(np.sign(soft_id) == np.sign(dst))))

        if len(dm):
            in_pk = float(np.mean(np.abs(dm - DM_DSTAR) < 1.0))
            check(f"{tname}: dm peaks at {DM_DSTAR:.2f}", in_pk > 0.05,
                  f"{in_pk:.1%} within +-1 MeV; range "
                  f"[{dm.min():.2f}, {dm.max():.2f}] MeV")
            ceiling = 155.5 if tname != "DstD02KPi" else 165.5
            tail = float(np.mean(dm > ceiling))
            check(f"{tname}: dm tail above the line cut ({ceiling}) is "
                  "small (<1%)", tail < 0.01,
                  f"tail {tail:.2%}, max dm = {dm.max():.2f} (post-fit "
                  "migration past the stripping cut is expected at the "
                  "permille level)")

        pnk = f"{s['h1']}_MC15TuneV1_ProbNNk"
        has_probnn = pnk in keys or f"{s['h1']}_ProbNNk" in keys
        check(f"{tname}: ProbNN branches present (ANNPID)", has_probnn,
              "" if has_probnn else f"looked for {pnk}")
        if s["kaons"] and len(dm) and has_probnn:
            fracs = []
            for kq in s["kaons"]:
                kb = (f"{kq}_MC15TuneV1_ProbNNk"
                      if f"{kq}_MC15TuneV1_ProbNNk" in keys
                      else f"{kq}_ProbNNk")
                if kb in keys:
                    pk = t[kb].array(library="np")
                    fracs.append(float(np.mean(pk > 0.1)))
            frac = min(fracs) if fracs else 0.0
            if tname != "DstD02KPi":
                check(f"{tname}: ALL kaons pass line cut ProbNNk>0.1",
                      frac > 0.99, f"lowest kaon fraction {frac:.2%}")
            else:
                print(f"  [INFO] {tname}: kaon ProbNNk>0.1 fraction {frac:.2%} "
                      "(no such cut on the RS line, informational)")

        n_tos = sum(1 for k in keys if k.endswith("_TOS") or k.endswith("_TIS"))
        check(f"{tname}: TISTOS branches present", n_tos > 0,
              f"{n_tos} TIS/TOS branches")

    for tname in ("DstpD02KK", "DstmD02KK"):
        if tname not in data:
            continue
        s = TREES[tname]
        t = data[tname]["tree"]
        need = [f"{q}_P{c}" for q in (s["h1"], s["h2"]) for c in "XYZ"] \
            + [f"{s['d0']}_MM"]
        if not all(k in data[tname]["keys"] for k in need):
            continue
        a = t.arrays(need, library="np")
        if len(a[f"{s['d0']}_MM"]) == 0:
            continue
        mkk = kin.invariant_mass(a["Kplus_PX"], a["Kplus_PY"], a["Kplus_PZ"], M_K,
                                 a["Kminus_PX"], a["Kminus_PY"], a["Kminus_PZ"], M_K)
        dev = np.abs(mkk - a[f"{s['d0']}_MM"])
        med = float(np.median(dev))
        check(f"{tname}: kaon-hypothesis mass closure (bulk)", med < 0.5,
              f"median {med * 1000:.0f} keV, max {dev.max():.2f} MeV, n={len(dev)}")

    print(f"\n  --- H. branching-fraction normalisation ({pol_label}) ---")
    if "DstD02KPi" not in data or data["DstD02KPi"]["n"] == 0:
        check("KPi control has candidates", False)
        continue
    yields = {}
    for mode, members in (("KPi", ["DstD02KPi"]),
                          ("KK", ["DstpD02KK", "DstmD02KK"]),
                          ("PiPi", ["DstpD02PiPi", "DstmD02PiPi"])):
        m_all = np.concatenate([data[t]["m"] for t in members if t in data]) \
            if any(t in data for t in members) else np.array([])
        dm_all = np.concatenate([data[t]["dm"] for t in members if t in data]) \
            if any(t in data for t in members) else np.array([])
        n_cand = len(m_all)
        in_dm = np.abs(dm_all - DM_DSTAR) < 0.6
        n_sig, e_sig = sideband_signal(m_all, in_dm)
        yields[mode] = (n_cand, n_sig, e_sig)
        print(f"  {mode:5s}: candidates {n_cand:>8,}   dm-tagged signal "
              f"{n_sig:7.1f} +- {e_sig:.1f}")

    n_kpi_sig = yields["KPi"][1]
    if n_kpi_sig > 0:
        for mode in ("KK", "PiPi"):
            n, e = yields[mode][1], yields[mode][2]
            exp = n_kpi_sig * BR[mode] / BR["KPi"]
            ratio = n / exp if exp > 0 else 0.0
            check(f"{mode}: signal within 10x of BR expectation "
                  f"(v2 was ~1/1200)", ratio > 0.1,
                  f"N={n:.0f}+-{e:.0f}, expected {exp:.0f}, obs/exp = {ratio:.3f}"
                  f"  (v1 benchmark: KK 0.19, PiPi 0.59)")
        for mode in ("KK", "PiPi"):
            r_cand = yields[mode][0] / max(yields["KPi"][0], 1)
            print(f"  [INFO] {mode}/KPi candidate ratio = {r_cand:.4f}  "
                  f"(v1 CP line: KK 0.0196, PiPi 0.0218; v2 broken: 0.004)")

print(f"\n{'=' * 70}")
if n_checks == 0:
    raise SystemExit(
        "VERDICT: NOTHING CHECKED, no archived test job found, inconclusive")
print(f"VERDICT ({n_checks} checks):",
      "ALL CHECKS PASSED" if overall_ok else "AT LEAST ONE FAILURE")
raise SystemExit(0 if overall_ok else 1)
