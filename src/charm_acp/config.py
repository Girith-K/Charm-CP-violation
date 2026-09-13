# seed, paths, list of data files, fit window, pT eta grid, blinding and run info

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

SEED = 20260716

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
PLOTS_DIR = REPO / "plots"
RESULTS_DIR = REPO / "results"

TESTPROD_V2 = DATA_DIR / "archive" / "testprod_v2_job0.root"

PRODUCTIONS = {
    "v2": {
        "layout": "v2",
        "dir": DATA_DIR / "v2",
        "files": {
            "00406477_00000001_1.dvntuple.root": "MagDown",
            "00406477_00000002_1.dvntuple.root": "MagDown",
            "00406477_00000014_1.dvntuple.root": "MagDown",
        },
        "sha256": {},
    },
    "v3": {
        "layout": "v3",
        "dir": DATA_DIR / "v3",
        "files": {
            "00412869_00000004_1.dvntuple.root": "MagDown",
            "00412869_00000015_1.dvntuple.root": "MagDown",
            "00412870_00000012_1.dvntuple.root": "MagUp",
            "00412870_00000013_1.dvntuple.root": "MagUp",
        },
        "sha256": {
            "00412869_00000004_1.dvntuple.root":
                "6902bc763abcda9389af60ab73374dbb1e823aaf6aadad74b8c03c0318b1e62d",
            "00412869_00000015_1.dvntuple.root":
                "5baddf89790818e8d885318640a010c0664edb4888fa200072a82cde0c8ea7a5",
            "00412870_00000012_1.dvntuple.root":
                "07f515b8943c7ee0ab43cb6b3b723aabc7ce0af489f2ef94257b063f82a8dbc3",
            "00412870_00000013_1.dvntuple.root":
                "926216d3bb84744e5e5909a83ad6e3ee12ca1afda973102108abb51d8b6d0dc1",
        },
    },
}

PRODUCTION = os.environ.get("CHARM_ACP_PRODUCTION", "v3")


def production_spec(name=None):
    name = name or PRODUCTION
    if name not in PRODUCTIONS:
        raise SystemExit(f"unknown production '{name}', have {sorted(PRODUCTIONS)}")
    return name, PRODUCTIONS[name]


def file_digest(path, algo="sha256", chunk=16 << 20):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def production_files(name=None, verify=None):
    name, spec = production_spec(name)
    stray = sorted(p.name for p in DATA_DIR.glob("*.dvntuple.root"))
    if stray:
        raise SystemExit(
            f"files in data/ not listed in config.PRODUCTIONS: {stray}")
    if verify is None:
        verify = os.environ.get("CHARM_ACP_VERIFY", "") not in ("", "0")
    pinned = spec.get("sha256") or {}
    out = []
    for fname, polarity in spec["files"].items():
        path = spec["dir"] / fname
        if not path.exists():
            raise SystemExit(f"manifest file missing on disk: {path}")
        if verify and fname in pinned:
            got = file_digest(path)
            if got != pinned[fname]:
                raise SystemExit(
                    f"checksum mismatch for {fname}\n"
                    f"  expected {pinned[fname]}\n  found    {got}")
        out.append((path, polarity))
    return out


def input_manifest(name=None, with_digest=False):
    name, spec = production_spec(name)
    pinned = spec.get("sha256") or {}
    rows = []
    for fname, polarity in spec["files"].items():
        path = spec["dir"] / fname
        row = {"file": fname, "polarity": polarity,
               "bytes": path.stat().st_size if path.exists() else None,
               "sha256_pinned": pinned.get(fname)}
        if with_digest and path.exists():
            row["sha256_observed"] = file_digest(path)
        rows.append(row)
    return {"production": name, "layout": spec["layout"], "files": rows}


DM_FIT_LO = 140.0
DM_FIT_HI = 158.0
DM_NBINS = 72

GRID_PT_EDGES = (2000., 3500., 5000., 7000., 10000., 30000.)
GRID_ETA_EDGES = (2.0, 2.75, 3.25, 3.75, 4.25, 5.0)

PDG_EDITION = "2024 (Phys. Rev. D 110, 030001)"
BR = {"KPi": 3.947e-2, "KK": 4.08e-3, "PiPi": 1.454e-3}

COMPOSITION_MIN_FRACTION = 0.30

BLIND_SCALE = {"v2": 0.02, "v3": 0.25}


def blind_scale(name=None):
    return BLIND_SCALE[name or PRODUCTION]


def blind_passphrase():
    f = DATA_DIR / ".blind_passphrase"
    if not f.exists():
        raise SystemExit(
            "data/.blind_passphrase not found")
    return f.read_text(encoding="utf-8").strip()


def _git(*argv):
    try:
        out = subprocess.run(["git", "-C", str(REPO), *argv],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _pkg_versions():
    out = {}
    for name in ("numpy", "scipy", "iminuit", "uproot", "matplotlib"):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        out[name] = getattr(mod, "__version__", "?")
    return out


def provenance(name=None, with_digest=False):
    from . import __version__

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "code_version": __version__,
        "git_commit": head,
        "git_describe": _git("describe", "--always", "--dirty", "--tags"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _pkg_versions(),
        "seed": SEED,
        "pdg_edition": PDG_EDITION,
        "blind_scale": blind_scale(name),
        "inputs": input_manifest(name, with_digest=with_digest),
    }
