from __future__ import annotations

from pathlib import Path
SEED = 20260716

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
PLOTS_DIR = REPO / "plots"

TESTPROD_V2 = DATA_DIR / "testprod_v2_job0.root"

# Production ntuples from the Ntupling Service, named <prod-id>_<job>_1.dvntuple
# .root. Glob rather than a list, so dropping a newly downloaded file into data/
# extends the dataset with no code change. The pipeline reports which files it
# used and the summed luminosity.Check names before running
PRODUCTION_GLOB = "*.dvntuple.root"
RESULTS_DIR = REPO / "results"


def production_files():
    """Sorted list of production ntuples currently present in data/"""
    return sorted(DATA_DIR.glob(PRODUCTION_GLOB))

# Fit window runs from just above the charged pion mass, the kinematic
# threshold of Delta m, up to just below the stripping cut (the D2hh lines keep
# dm < 160). Staying inside both edges avoids modelling the cut turn ons
DM_FIT_LO = 140.0  # MeV
DM_FIT_HI = 158.0  # MeV
DM_NBINS = 72      # 0.25 MeV bins

# The asymmetry difference is blinded with a deterministic offset derived from
# this passphrase, see asymmetry.blind_offset. The passphrase is public in the
# repo, what protects us is discipline, nobody computes the offset by hand
# before Phase 10. Unblinding means calling unblind() explicitly
BLIND_PASSPHRASE = "charm-acp-2026-dacp"
BLIND_SCALE = 0.02  # offset drawn uniformly in [-BLIND_SCALE, +BLIND_SCALE]
