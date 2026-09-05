"""Explain a missing optional dependency instead of showing a bare traceback.

Solving the study needs only Pyomo, Ipopt and numpy. Rebuilding the converted
networks additionally needs OpenDSS and powerio, which are the two pieces most
often absent: the usual cause is running from conda's base environment rather
than the study's own. A plain ``ModuleNotFoundError`` names the module but not
the step that was skipped, so it is caught here and answered.
"""

from __future__ import annotations

import importlib
import sys

# What each module is for, and how it arrives, so the message can say both.
_PURPOSE = {
    "opendssdirect": "read the feeder electrics back out of OpenDSS",
    "powerio": "parse the OpenDSS feeder into BMOPF JSON",
}


def require(module: str):
    """Import ``module``, or exit explaining which setup step is missing."""
    try:
        return importlib.import_module(module)
    except ImportError:
        purpose = _PURPOSE.get(module, "run this step")
        print(
            f"\n{module} is not installed in the environment now running "
            f"({sys.executable}).\n"
            f"It is needed to {purpose}, and it comes with the study "
            "environment:\n\n"
            "    conda env create -f environment.yml\n"
            "    conda activate tsbi-opf\n\n"
            "If that environment is already created, activating it is the "
            "step that is missing;\n"
            "a prompt reading (base) means it is not active.\n\n"
            "Solving the study does not need this module at all. The converted "
            "networks are\n"
            "tracked, so this works in any environment with Pyomo and Ipopt:\n\n"
            "    python run_curtailment_study.py --case 4bus_120v\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
