# tests for the composition and polarity checks

import pytest

from charm_acp import gates
from charm_acp.config import BR, COMPOSITION_MIN_FRACTION


def test_br_expected_scales_by_branching_fraction():
    exp = gates.br_expected_yield(1_000_000.0, "KK")
    assert exp == pytest.approx(1_000_000.0 * BR["KK"] / BR["KPi"])


def test_gate_refuses_a_reflection_dominated_yield():
    n_kpi = 2_735_903.0
    expected = gates.br_expected_yield(n_kpi, "KK")
    g = gates.composition_gate(0.0013 * expected, n_kpi, "KK")
    assert g["passed"] is False
    assert g["ratio"] < 0.01
    assert "BR-scaled" in g["reason"]


def test_gate_accepts_a_healthy_yield():
    n_kpi = 2_735_903.0
    expected = gates.br_expected_yield(n_kpi, "KK")
    g = gates.composition_gate(0.9 * expected, n_kpi, "KK")
    assert g["passed"] is True and g["reason"] is None
    assert g["ratio"] == pytest.approx(0.9, rel=1e-9)


def test_gate_boundary_is_inclusive_at_the_threshold():
    n_kpi = 1_000_000.0
    expected = gates.br_expected_yield(n_kpi, "PiPi")
    at = gates.composition_gate(COMPOSITION_MIN_FRACTION * expected, n_kpi,
                                "PiPi")
    just_below = gates.composition_gate(
        0.999 * COMPOSITION_MIN_FRACTION * expected, n_kpi, "PiPi")
    assert at["passed"] is True
    assert just_below["passed"] is False


def test_gate_handles_a_missing_control():
    g = gates.composition_gate(100.0, 0.0, "KK")
    assert g["passed"] is False and g["ratio"] is None


def test_unknown_mode_raises_rather_than_silently_passing():
    with pytest.raises(KeyError):
        gates.br_expected_yield(1000.0, "KsKs")


def test_polarity_gate_requires_the_same_complete_set():
    both = {"MagDown", "MagUp"}
    ok = gates.polarity_set_gate({"KK": both, "PiPi": both}, both)
    assert ok["passed"] is True

    partial = gates.polarity_set_gate({"KK": {"MagDown"}, "PiPi": both}, both)
    assert partial["passed"] is False
    assert partial["used"]["KK"] == ["MagDown"]


def test_gate_summary_collapses_reasons():
    n_kpi = 1_000_000.0
    bad = gates.composition_gate(1.0, n_kpi, "KK")
    good = gates.composition_gate(gates.br_expected_yield(n_kpi, "PiPi"),
                                  n_kpi, "PiPi")
    assert gates.gate_summary([good]) is None
    msg = gates.gate_summary([bad, good])
    assert msg is not None and msg.startswith("KK:")
