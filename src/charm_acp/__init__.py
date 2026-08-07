"""charm_acp, direct CP violation in charm (ΔA_CP) from LHCb open data.

kinematics, invariant mass, Δm, four-vector helpers
selection, PID / quality / prompt cuts, cutflow
fitting, Δm signal+background model, fit driver, toy/pull tools
asymmetry, raw asymmetry + error, ΔA_CP combination, (pT, η) binning
"""

from . import asymmetry, fitting, kinematics, selection  # noqa: F401

__all__ = ["kinematics", "selection", "fitting", "asymmetry"]
__version__ = "0.1.0"
