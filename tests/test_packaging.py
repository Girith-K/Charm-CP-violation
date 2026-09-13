# tests for the version numbers and the run info

import json
import re
from pathlib import Path

import pytest

import charm_acp
from charm_acp import config

REPO = Path(__file__).resolve().parents[1]


def test_version_agrees_across_every_declaration():
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    assert m, "pyproject.toml has no version"
    assert m.group(1) == charm_acp.__version__

    cff = REPO / "CITATION.cff"
    if cff.exists():
        m = re.search(r'^version:\s*"?([^"\n]+)"?', cff.read_text(encoding="utf-8"), re.M)
        assert m, "CITATION.cff has no version"
        assert m.group(1).strip() == charm_acp.__version__


def test_provenance_carries_what_is_needed_to_reproduce():
    p = config.provenance()
    for key in ("code_version", "git_commit", "python", "packages", "seed",
                "pdg_edition", "inputs"):
        assert key in p
    assert p["code_version"] == charm_acp.__version__
    assert p["seed"] == config.SEED
    assert isinstance(p["packages"], dict) and "numpy" in p["packages"]
    assert json.dumps(p)


def test_provenance_survives_without_git():
    p = config.provenance()
    assert p["git_commit"] is None or re.fullmatch(r"[0-9a-f]{40}",
                                                   p["git_commit"])


def test_input_manifest_lists_every_manifested_file():
    man = config.input_manifest("v3")
    names = {row["file"] for row in man["files"]}
    assert names == set(config.PRODUCTIONS["v3"]["files"])
    assert man["layout"] == "v3"


def test_v3_inputs_have_pinned_checksums():
    pinned = config.PRODUCTIONS["v3"]["sha256"]
    assert set(pinned) == set(config.PRODUCTIONS["v3"]["files"])
    for digest in pinned.values():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_one_kinematic_grid_for_the_whole_analysis():
    assert len(config.GRID_PT_EDGES) >= 3
    assert len(config.GRID_ETA_EDGES) >= 3
    assert list(config.GRID_PT_EDGES) == sorted(config.GRID_PT_EDGES)
    assert list(config.GRID_ETA_EDGES) == sorted(config.GRID_ETA_EDGES)


def test_composition_threshold_is_a_single_source():
    from charm_acp import gates
    n_kpi = 1e6
    exp = gates.br_expected_yield(n_kpi, "KK")
    g = gates.composition_gate(config.COMPOSITION_MIN_FRACTION * exp, n_kpi,
                               "KK")
    assert g["threshold"] == config.COMPOSITION_MIN_FRACTION
    assert g["passed"] is True


def test_file_digest_matches_hashlib(tmp_path):
    import hashlib
    f = tmp_path / "x.bin"
    payload = b"charm-acp" * 5000
    f.write_bytes(payload)
    assert config.file_digest(f) == hashlib.sha256(payload).hexdigest()
