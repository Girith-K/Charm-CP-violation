# Seeds, paths, data manifest, fit window and blinding configuration

from __future__ import annotations

import os
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
    },
}

PRODUCTION = os.environ.get("CHARM_ACP_PRODUCTION", "v3")


def production_spec(name=None):
    name = name or PRODUCTION
    if name not in PRODUCTIONS:
        raise SystemExit(f"unknown production '{name}', have {sorted(PRODUCTIONS)}")
    return name, PRODUCTIONS[name]


def production_files(name=None):
    name, spec = production_spec(name)
    stray = sorted(p.name for p in DATA_DIR.glob("*.dvntuple.root"))
    if stray:
        raise SystemExit(
            f"unmanifested production files sitting in data/: {stray}. "
            "Move them into a production folder and list them in "
            "config.PRODUCTIONS, nothing is ever picked up by glob.")
    out = []
    for fname, polarity in spec["files"].items():
        path = spec["dir"] / fname
        if not path.exists():
            raise SystemExit(f"manifest file missing on disk: {path}")
        out.append((path, polarity))
    return out


DM_FIT_LO = 140.0
DM_FIT_HI = 158.0
DM_NBINS = 72

# D0 branching fractions, PDG 2024 edition (Phys. Rev. D 110, 030001), pinned
# so the BR-normalisation gate is reproducible across reruns. The 2026 edition
# gives 3.936e-2 / 4.07e-3 / 1.451e-3, a drift of <=0.3% that is immaterial
# to a gate quoted in factors of ten. Single source: import from here
PDG_EDITION = "2024 (Phys. Rev. D 110, 030001)"
BR = {"KPi": 3.947e-2, "KK": 4.08e-3, "PiPi": 1.454e-3}

BLIND_SCALE = {"v2": 0.02, "v3": 0.25}


def blind_scale(name=None):
    return BLIND_SCALE[name or PRODUCTION]


def blind_passphrase():
    f = DATA_DIR / ".blind_passphrase"
    if not f.exists():
        raise SystemExit(
            "data/.blind_passphrase not found. Create the file with the "
            "analysis passphrase on one line, it is gitignored and never "
            "enters the repository.")
    return f.read_text(encoding="utf-8").strip()
