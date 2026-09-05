#!/usr/bin/env python3
"""Run the 120 PV inverter curtailment JSON scenarios.

The runner reads each scenario's inverter type, controls, per-inverter
weights, objective norm, line ratings and solver settings from that file, and
the voltage band it is held to from its feeder's study design. It does not
load an OpenDSS twin or mutate a shared case at run time.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# Ipopt's max_cpu_time budget is CPU time, not wall time, and the study already
# parallelises across scenarios, so a multithreaded BLAS inside every worker
# spends that budget on thread contention instead of iterations and cases time
# out. Pin the maths libraries before numpy loads so the behaviour is identical
# in the conda environment and in a plain pip one. An explicit setting wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from converters.generate_curtailment_scenarios import FeederLibrary
from src.bmopf_parser import ScenarioSelectionError, StudyInputs
from src.network_plotter import NetworkPlotter
from src.network_simulator import CurtailmentStudy, ScenarioSolver

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = REPO_ROOT / "data" / "bmopf_json"
DEFAULT_OUT = REPO_ROOT / "results" / "curtailment_matrix.json"
DEFAULT_PLOT_OUT = REPO_ROOT / "results" / "curtailment_by_mode.png"
DEFAULT_TABLE_OUT = REPO_ROOT / "results" / "runtime_table.csv"

P_MODES = CurtailmentStudy.P_MODES
Q_MODES = CurtailmentStudy.Q_MODES
STRATEGIES = CurtailmentStudy.NORMS
DEFAULT_WORKERS = max(1, os.cpu_count() or 1)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="study root containing <case>/scenarios/*.json",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="a case under the study root; repeat for several (default: all)",
    )
    parser.add_argument("--P_mode", "--p-mode", choices=P_MODES)
    parser.add_argument("--Q_mode", "--q-mode", choices=Q_MODES)
    parser.add_argument("--curtailment_strategy", "--norm", choices=STRATEGIES)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel solver processes "
        f"(default: {DEFAULT_WORKERS}, every core on this machine)",
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--solver",
        "--linear-solver",
        dest="linear_solver",
        choices=ScenarioSolver.LINEAR_SOLVERS,
        default=None,
        help="Ipopt linear solver (default: "
        f"{ScenarioSolver.DEFAULT_LINEAR_SOLVER}, which is what the scenarios "
        "state and what the published results use). The HSL solvers and the "
        "others are loaded at run time, so they work only where they are "
        "installed; --check lists the ones this machine has, and a run stops "
        "before solving if the one asked for is missing.",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed the optimization with the reference solve's bus voltages "
        "(default: on). The voltage band does not depend on this, so turning "
        "it off changes only the path to the optimum, not the optimum itself. "
        "Some cases need it to converge at all.",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Stream the Ipopt log to the console. Forces one worker, since "
        "parallel solves would interleave their output.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--plot-output",
        "--output",
        type=Path,
        default=None,
        help="Write the curtailment heatmap here (default: beside --out)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Render an existing matrix JSON without running optimization",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Matrix JSON for --plot-only (default: --out)",
    )
    parser.add_argument(
        "--plot-norm",
        choices=("ALL", *STRATEGIES),
        default="ALL",
        help="Norm rows to render with --plot-only (default: ALL)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the curtailment heatmap; useful for solver-only runs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report whether this environment can run the study, and exit",
    )
    return parser


def _report_module(module: str, note: str = "") -> bool:
    """Print one module's version, and return whether it could be imported."""
    try:
        found = importlib.import_module(module)
    except ImportError:
        print(f"  {module:<14}absent    {note}")
        return False
    print(f"  {module:<14}{getattr(found, '__version__', 'present')}")
    return True


def linear_solver_report(name: str) -> tuple[bool, str]:
    """Report whether Ipopt here can solve with one linear solver, and why not.

    Which linear solvers work is a property of the machine rather than of the
    Ipopt version, because HSL and Pardiso are loaded at run time. So a tiny
    problem is solved and Ipopt's own statement of the solver it used is read
    back: that catches both a failure to load and a silent substitution.
    """
    import pyomo.environ as pe
    from pyomo.common.errors import ApplicationError

    model = pe.ConcreteModel()
    model.x = pe.Var(bounds=(0.0, 2.0), initialize=0.5)
    model.floor = pe.Constraint(expr=model.x >= 0.5)
    model.objective = pe.Objective(expr=(model.x - 0.25) ** 2)
    opt = pe.SolverFactory("ipopt")
    opt.options["linear_solver"] = name
    log = io.StringIO()
    # Pyomo logs a failed solve at ERROR; here a failure is the answer being
    # measured, so it is collected rather than printed.
    pyomo_logger = logging.getLogger("pyomo")
    previous_level = pyomo_logger.level
    pyomo_logger.setLevel(logging.CRITICAL)
    try:
        with (
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            results = opt.solve(model, tee=True, load_solutions=False)
        termination = str(results.solver.termination_condition)
    except (ApplicationError, OSError, RuntimeError, ValueError) as exc:
        return (False, f"{type(exc).__name__}: {exc}".splitlines()[0][:160])
    finally:
        pyomo_logger.setLevel(previous_level)

    written = log.getvalue()
    if termination not in ("optimal", "locallyOptimal"):
        for line in written.splitlines():
            if line.startswith("EXIT:"):
                return (False, line.strip())
        return (False, f"Ipopt reported {termination}")
    # Ipopt names the solver it is running with; MUMPS adds its version.
    if f"running with linear solver {name}" not in written.lower():
        return (False, "Ipopt did not use the requested linear solver")
    return (True, "")


def check_environment(root: Path) -> int:
    """Report what this environment can and cannot do, and why.

    Meant to be run before a session rather than discovering a missing piece
    mid-command. Solving needs Pyomo, Ipopt and numpy; rebuilding the converted
    networks additionally needs OpenDSS and powerio, and is only necessary
    after a feeder changes.
    """
    print(f"  {'python':<14}{sys.version.split()[0]}")
    print(f"  {'':<14}{sys.executable}")

    required = [
        module
        for module in ("pyomo", "numpy", "matplotlib")
        if not _report_module(module, "required to solve")
    ]

    ipopt = shutil.which("ipopt")
    print(f"  {'ipopt':<14}{ipopt or 'absent    the solver binary'}")
    if not ipopt:
        required.append("ipopt")

    optional = [
        module
        for module in ("opendssdirect", "powerio")
        if not _report_module(module, "only to rebuild the networks")
    ]

    if ipopt and "pyomo" not in required:
        usable = [
            name
            for name in ScenarioSolver.LINEAR_SOLVERS
            if linear_solver_report(name)[0]
        ]
        print(f"  {'--solver':<14}{', '.join(usable) or 'none'}")

    networks = sorted(root.glob("*/network/bmopf.json"))
    scenarios = sorted(root.glob("*/scenarios/*.json"))
    print(f"\n  {'networks':<14}{len(networks)} feeder(s) under {root}")
    print(f"  {'scenarios':<14}{len(scenarios)} (built on first run if absent)")
    print()

    if required:
        print("Cannot solve: " + ", ".join(required) + " missing.")
        print("    conda env create -f environment.yml")
        print("    conda activate tsbi-opf")
        return 1
    if not networks:
        print(f"Cannot solve: no converted feeder networks under {root}.")
        print("They are tracked, so a clone has them; check you are in the")
        print("repository root, or run converters/dss_to_bmopf.py to rebuild.")
        return 1
    print("Ready. Try: python run_curtailment_study.py --case 4bus_120v")
    if optional:
        print(
            "Rebuilding networks from the OpenDSS feeders is unavailable here, "
            "which matters only if you change a feeder."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run selected scenarios, publish the result matrix, and render it."""
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.check:
        return check_environment(root)
    if args.plot_only:
        input_path = args.input or args.out
        output_path = args.plot_output or DEFAULT_PLOT_OUT
        try:
            NetworkPlotter.from_file(input_path).render_heatmap(
                output_path, args.plot_norm
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Could not generate curtailment plot: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {output_path}")
        return 0
    inputs = StudyInputs(root, per_case=CurtailmentStudy.scenarios_per_case())
    library = FeederLibrary()
    cases = library.ordered(args.case or library.cases())
    inputs.ensure(cases)
    try:
        paths = inputs.scenario_files(
            cases, args.P_mode, args.Q_mode, args.curtailment_strategy
        )
    except ScenarioSelectionError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.linear_solver is not None:
        usable, reason = linear_solver_report(args.linear_solver)
        if not usable:
            print(
                f"Ipopt here cannot use linear solver '{args.linear_solver}': {reason}",
                file=sys.stderr,
            )
            print(
                "MUMPS is built into every Ipopt; the others are loaded at run "
                "time. Run --check to list the ones this machine has.",
                file=sys.stderr,
            )
            return 1

    workers = max(1, int(args.workers))
    if args.tee and workers > 1:
        # Several solvers writing to one terminal produces unreadable output,
        # so reading the log takes precedence over speed here.
        print(
            f"--tee streams one solver log at a time; using 1 worker "
            f"instead of {workers}.",
            file=sys.stderr,
        )
        workers = 1
    print(f"Running {len(paths)} standalone JSON scenarios with {workers} worker(s)")

    study = CurtailmentStudy(
        ScenarioSolver(
            timeout=args.timeout,
            tee=args.tee,
            warm_start=args.warm_start,
            voltage_bands=library.bands(cases),
            linear_solver=args.linear_solver,
        ),
        workers=workers,
    )
    started = time.monotonic()
    results = study.run(paths)
    document = study.publish(results, cases, len(paths), labels=library.labels())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    plotter = NetworkPlotter(document)

    table_out = (
        DEFAULT_TABLE_OUT
        if args.out == DEFAULT_OUT
        else args.out.with_name(args.out.stem + "_runtime.csv")
    )
    plotter.write_runtime_table(table_out)
    print(f"Wrote {table_out}")

    failures = study.failures(results)
    print(
        f"Wrote {args.out}. {len(paths) - len(failures)}/{len(paths)} "
        f"scenarios passed in {time.monotonic() - started:.1f}s."
    )

    plot_failure = None
    if not args.no_plot:
        # A partial run names its figure after its own matrix, so it cannot
        # overwrite the published one the way a shared name would.
        plot_output = args.plot_output or (
            DEFAULT_PLOT_OUT
            if args.out == DEFAULT_OUT
            else args.out.with_suffix(".png")
        )
        try:
            plotter.render_heatmap(plot_output, "ALL")
            print(f"Wrote {plot_output}")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            plot_failure = exc
            print(
                f"Could not generate curtailment plot {plot_output}: {exc}",
                file=sys.stderr,
            )

    if failures:
        return 1
    return 1 if plot_failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
