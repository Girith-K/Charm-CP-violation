# checks that decide if a mode can be quoted

from __future__ import annotations

from .config import BR, COMPOSITION_MIN_FRACTION


def br_expected_yield(n_control, mode, control="KPi", br=None):
    br = br or BR
    if mode not in br or control not in br:
        raise KeyError(f"no branching fraction for {mode!r} or {control!r}")
    return float(n_control) * br[mode] / br[control]


def composition_gate(n_mode, n_control, mode, control="KPi",
                     threshold=COMPOSITION_MIN_FRACTION, br=None):
    expected = br_expected_yield(n_control, mode, control, br)
    ratio = float(n_mode) / expected if expected > 0 else None
    if ratio is None:
        return {"mode": mode, "passed": False, "ratio": None,
                "expected": expected, "observed": float(n_mode),
                "threshold": threshold,
                "reason": f"no {control} control yield to normalise against"}
    passed = ratio >= threshold
    return {
        "mode": mode,
        "passed": bool(passed),
        "ratio": ratio,
        "expected": expected,
        "observed": float(n_mode),
        "threshold": threshold,
        "reason": None if passed else
                  f"yield below {threshold:.0%} of the BR-scaled "
                  f"{control} expectation",
    }


def polarity_set_gate(used_by_mode, required):
    required = set(required)
    sets = {m: set(s) for m, s in used_by_mode.items()}
    passed = all(s == required for s in sets.values())
    return {
        "passed": bool(passed),
        "required": sorted(required),
        "used": {m: sorted(s) for m, s in sets.items()},
        "reason": None if passed else
                  "modes did not fit in the same complete polarity set",
    }


def gate_summary(gates):
    bad = [g for g in gates if not g.get("passed")]
    if not bad:
        return None
    return "; ".join(f"{g.get('mode', 'polarity')}: {g['reason']}" for g in bad)
