#!/usr/bin/env python3
"""Compare this model's power flow against OpenDSS on every maintained feeder.

The study's own checks establish that a solution is self-consistent: residuals
converge and constraints hold. They cannot establish that the network was
translated correctly. This script does that, by solving the same feeder twice
and comparing bus voltages:

  OpenDSS            reads data/dss/<case>/network/Master.dss directly
  this model         reads data/bmopf_json/<case>/network/bmopf.json

Both are solved without PV, so the comparison isolates the network translation
rather than the inverter model. The converter does not carry OpenDSS PVSystem
objects into BMOPF, so any it finds are disabled before OpenDSS solves.

Voltages are compared in volts rather than per unit. The two tools do not
always choose the same per-unit base: where a source declares pu=1.03, this
model may carry the energised magnitude as the base while OpenDSS keeps the
nominal, and the same physical voltage then reads as 1.000 here and 1.030
there. Volts remove that ambiguity.

    python converters/validate_against_opendss.py
    python converters/validate_against_opendss.py --case M1 --tolerance 1e-3
"""

from __future__ import annotations

import argparse
import cmath
import copy
import json
import math
import sys
from pathlib import Path

import pyomo.environ as pe

# This module lives in converters/, so the repository root is one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Buses this pipeline adds that OpenDSS keeps inside its Vsource object.
SOURCE_INTERNAL_SUFFIX = "_source_internal"

# The agreement this repository publishes, used as a regression bar.
PUBLISHED_AGREEMENT = REPO_ROOT / "results" / "opendss_validation.json"

# How much worse than the published agreement a feeder may get before it counts
# as a regression, and the error below which the comparison is not worth making.
BASELINE_HEADROOM = 2.0
BASELINE_FLOOR = 1e-6
sys.path.insert(0, str(REPO_ROOT))

from converters._requires import require  # noqa: E402
from converters.generate_curtailment_scenarios import (  # noqa: E402
    FeederLibrary,
)
from src.bmopf_parser import load_bmopf_network  # noqa: E402
from src.network_simulator import NetworkSimulator  # noqa: E402

dss = require("opendssdirect")


def opendss_voltages(case: str) -> dict[tuple[str, str], complex]:
    """Solve the feeder in OpenDSS and return the phasor volts of each node."""
    master = REPO_ROOT / "data" / "dss" / case / "network" / "Master.dss"
    dss.Command("Clear")
    dss.Command(f'Redirect "{master}"')

    # The converter does not represent PVSystem objects, so remove them here
    # too. Leaving them in would compare a feeder with PV against one without.
    disabled = 0
    if dss.PVsystems.Count():
        for name in dss.PVsystems.AllNames():
            if name.lower() == "none":
                continue
            dss.Command(f"PVSystem.{name}.enabled=no")
            disabled += 1
        dss.Command("Solve")
    if disabled:
        print(f"    disabled {disabled} OpenDSS PVSystem objects", flush=True)

    dss.Command("Set ControlMode=OFF")
    dss.Command("Solve")
    if not dss.Solution.Converged():
        raise RuntimeError(f"OpenDSS did not converge on {case}")

    out: dict[tuple[str, str], complex] = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        nodes = dss.Bus.Nodes()
        polar = dss.Bus.VMagAngle()
        for index, node in enumerate(nodes):
            magnitude = float(polar[2 * index])
            degrees = float(polar[2 * index + 1])
            out[(bus.lower(), str(node))] = cmath.rect(magnitude, math.radians(degrees))
    return out


def model_voltages(case: str) -> dict[tuple[str, str], complex]:
    """Solve the converted network here and return volts by node."""
    path = REPO_ROOT / "data" / "bmopf_json" / case / "network" / "bmopf.json"
    document = json.loads(path.read_text())
    network = load_bmopf_network(document)
    network.solver_settings = copy.deepcopy(
        {
            "name": "ipopt",
            "linear_solver": "mumps",
            "mumps_pivot_tolerance": 1e-3,
            "tol": 1e-9,
            "constraint_violation_tolerance": 1e-5,
            "max_iter": 5000,
            "honor_original_bounds": "yes",
            "mu_init": 0.1,
            "mu_strategy": "adaptive",
        }
    )
    simulator = NetworkSimulator(network)
    ok, message, _, _ = simulator.solve(timeout=300.0, tee=False)
    if not ok:
        raise RuntimeError(f"this model did not converge on {case}: {message}")

    out: dict[tuple[str, str], float] = {}
    for bus in network.buses:
        name = getattr(bus, "NodeName", None)
        phase = getattr(bus, "NodePhase", None)
        idx = getattr(bus, "int_bus_id", None)
        if name is None or phase is None or idx is None:
            continue
        phasor_pu = complex(
            float(pe.value(simulator.model.ipopt_vr_list[idx])),
            float(pe.value(simulator.model.ipopt_vi_list[idx])),
        )
        base = float(getattr(bus, "v_base_v", 0.0) or 0.0)
        if base <= 0.0:
            continue
        out[(str(name).lower(), str(phase))] = phasor_pu * base
    return out


def compare(case: str, tolerance: float) -> tuple[bool, dict]:
    """Solve both ways and summarise the agreement."""
    print(f"  {case}", flush=True)
    reference = opendss_voltages(case)
    ours = model_voltages(case)

    shared = sorted(set(reference) & set(ours))
    if not shared:
        raise RuntimeError(f"no bus names matched between the two models on {case}")

    # The two tools need not share a time reference, and a rotation applied to
    # every node alike is not a difference in the solution. One global rotation
    # is removed, so what is left is per-node: a wrong phase shift across a
    # transformer, or a swapped phase, still shows up.
    anchor = max(shared, key=lambda key: abs(reference[key]))
    rotation = cmath.rect(
        1.0, cmath.phase(reference[anchor]) - cmath.phase(ours[anchor])
    )

    errors = []
    for key in shared:
        a, b = reference[key], ours[key] * rotation
        if abs(a) <= 1e-9:
            continue
        errors.append((abs(b - a) / abs(a), key, a, b))
    errors.sort(reverse=True)

    worst, worst_key, worst_dss, worst_ours = errors[0]
    mean = sum(e[0] for e in errors) / len(errors)
    # A node OpenDSS energises and this model does not is a piece of feeder that
    # went missing, which no voltage comparison over the rest would reveal.
    missing = sorted(set(reference) - set(ours))
    # The source Thevenin branch adds terminals OpenDSS keeps inside its
    # Vsource object; anything else extra is unexplained.
    extra = sorted(
        key
        for key in set(ours) - set(reference)
        if not key[0].endswith(SOURCE_INTERNAL_SUFFIX)
    )
    summary = {
        "case": case,
        "compared_nodes": len(errors),
        "missing_from_model": len(missing),
        "unexplained_extra": len(extra),
        "mean_relative_error": mean,
        "max_relative_error": worst,
        "worst_node": f"{worst_key[0]}.{worst_key[1]}",
        "worst_opendss_v": abs(worst_dss),
        "worst_model_v": abs(worst_ours),
    }
    ok = worst <= tolerance and not missing and not extra
    print(
        f"    {len(errors)} nodes compared   mean {mean:.3e}   max {worst:.3e}"
        f"   {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"    worst at {summary['worst_node']}: "
        f"OpenDSS {abs(worst_dss):.2f} V at "
        f"{math.degrees(cmath.phase(worst_dss)):.2f} deg vs model "
        f"{abs(worst_ours):.2f} V at "
        f"{math.degrees(cmath.phase(worst_ours)):.2f} deg",
        flush=True,
    )
    if missing or extra:
        print(
            f"    unmatched nodes: {len(missing)} energised by OpenDSS and "
            f"absent here, {len(extra)} here and unexplained",
            flush=True,
        )
    return ok, summary


def main(argv: list[str] | None = None) -> int:
    """Compare the selected feeders and report where they disagree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="maximum acceptable relative difference in bus voltage magnitude "
        "(default: 1e-2, the one percent used for power-flow cross-validation "
        "in this project's earlier comparator)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write a JSON summary")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="do not compare against the published agreement in "
        f"{PUBLISHED_AGREEMENT.name}",
    )
    args = parser.parse_args(argv)

    available = FeederLibrary().converted()
    cases = args.case or available
    unknown = [case for case in cases if case not in available]
    if unknown:
        print(
            f"No converted feeder named {', '.join(unknown)}; "
            f"found {', '.join(available) or 'none'}",
            file=sys.stderr,
        )
        return 2
    print(
        f"Comparing {len(cases)} feeder(s) against OpenDSS, tolerance "
        f"{args.tolerance:g}\n"
    )
    # A fixed tolerance only catches an error large enough to reach it, and on
    # a lightly loaded feeder a sizeable modelling error moves the voltages far
    # less than that. The published agreement is therefore the second bar: a
    # conversion that gets materially worse than the recorded evidence is a
    # regression even while it still passes the tolerance.
    baseline = {}
    if not args.no_baseline and PUBLISHED_AGREEMENT.is_file():
        baseline = {
            entry["case"]: entry
            for entry in json.loads(PUBLISHED_AGREEMENT.read_text())
        }

    summaries, failures = [], []
    for case in cases:
        try:
            ok, summary = compare(case, args.tolerance)
        except (RuntimeError, OSError) as exc:
            print(f"    ERROR: {exc}", flush=True)
            failures.append(case)
            continue
        summaries.append(summary)
        recorded = baseline.get(case)
        if recorded is not None:
            was = float(recorded["max_relative_error"])
            now = float(summary["max_relative_error"])
            if now > max(was * BASELINE_HEADROOM, BASELINE_FLOOR):
                ok = False
                print(
                    f"    REGRESSED against the published agreement: "
                    f"{was:.3e} -> {now:.3e}",
                    flush=True,
                )
        if not ok:
            failures.append(case)
        print(flush=True)

    if args.out and failures:
        # The summary doubles as the regression bar, so a failing run must not
        # overwrite it with the worse numbers it just produced.
        print(
            f"Not writing {args.out}: it records the agreement this "
            "conversion is measured against, and this run did not agree."
        )
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"Wrote {args.out}")

    if failures:
        print(
            f"{len(cases) - len(failures)}/{len(cases)} feeders agree within "
            f"{args.tolerance:g}; failed: {', '.join(failures)}"
        )
        return 1
    print(f"All {len(cases)} feeders agree with OpenDSS within {args.tolerance:g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
