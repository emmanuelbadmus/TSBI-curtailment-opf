#!/usr/bin/env python3
"""Convert the maintained OpenDSS feeders into BMOPF study inputs.

This script deliberately performs only the network-format conversion.  The
separate ``generate_curtailment_scenarios.py`` script reads the resulting
``<case>/network/bmopf.json`` and creates the standalone inverter scenarios.
Running without a network argument converts all five maintained feeders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This module lives in converters/, so the repository root is one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from converters._requires import require  # noqa: E402

powerio = require("powerio")

from converters.generate_curtailment_scenarios import (  # noqa: E402  # noqa: E402
    FeederLibrary,
    clean_network_document,
)
from converters.opendss_electrics import (  # noqa: E402
    OpenDSSElectricsError,
    apply_opendss_line_electrics,
    apply_opendss_load_models,
    apply_opendss_shunt_states,
    apply_opendss_source_impedance,
    canonicalise_bus_names,
)

DEFAULT_DSS_ROOT = REPO_ROOT / "data" / "dss"
DEFAULT_ROOT = REPO_ROOT / "data" / "bmopf_json"


def convert(network_path: Path) -> dict:
    """Convert an OpenDSS master file and return the BMOPF document."""
    network = powerio.dist.parse_file(network_path)
    conversion = network.to_format("bmopf")
    for warning in network.warnings:
        print(f"parse warning: {warning}", file=sys.stderr)
    for warning in getattr(conversion, "warnings", []) or []:
        print(f"conversion warning: {warning}", file=sys.stderr)
    document = json.loads(conversion.text)
    # powerio carries the topology; the line electrics are read back from
    # OpenDSS so the study solves the same impedances it is validated against.
    notes = canonicalise_bus_names(document)
    notes += apply_opendss_load_models(document, network_path)
    notes += apply_opendss_shunt_states(document, network_path)
    # Applied after the load and shunt fixes: it prunes bus terminals once
    # nothing references them.
    notes += apply_opendss_line_electrics(document, network_path)
    # Applied last: the branch it adds has no OpenDSS line to match against.
    notes += apply_opendss_source_impedance(document, network_path)
    for note in notes:
        print(f"opendss electrics: {note}", file=sys.stderr)
    clean_network_document(document)
    return document


def case_name_for(network_path: Path, explicit_name: str | None = None) -> str:
    """Infer ``<case>`` from the standard ``<case>/dss_files`` layout."""
    if explicit_name:
        return explicit_name
    if network_path.parent.name in {"dss_files", "network"}:
        return network_path.parent.parent.name
    return network_path.parent.name


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the feeder converter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "network",
        nargs="?",
        help="one OpenDSS Master.dss file; omit to convert all five feeders",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="study root for <case>/network/bmopf.json output",
    )
    parser.add_argument(
        "--case",
        dest="case_name",
        help="select one feeder under the feeder root, or override an "
        "explicit path's case name",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Convert one feeder or all maintained feeders and report failures."""
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if args.network is None:
        library = FeederLibrary(dss_root=DEFAULT_DSS_ROOT, root=root)
        available = library.cases()
        if args.case_name and args.case_name not in available:
            print(
                f"No feeder named {args.case_name} under {DEFAULT_DSS_ROOT}; "
                f"found {', '.join(available) or 'none'}",
                file=sys.stderr,
            )
            return 2
        cases = (args.case_name,) if args.case_name else available
        requests = [
            (
                DEFAULT_DSS_ROOT / case / "network" / "Master.dss",
                case,
            )
            for case in cases
        ]
    else:
        network_path = Path(args.network).expanduser().resolve()
        requests = [(network_path, args.case_name)]

    failures = []
    for network_path, explicit_case_name in requests:
        if not network_path.is_file():
            message = f"OpenDSS master file not found: {network_path}"
            print(f"Error: {message}", file=sys.stderr)
            failures.append(message)
            continue
        case_name = case_name_for(network_path, explicit_case_name)
        output = root / case_name / "network" / "bmopf.json"
        try:
            document = convert(network_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(document, indent=2) + "\n")
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            powerio.PowerIOError,
            OpenDSSElectricsError,
        ) as exc:
            message = f"Error converting {network_path}: {exc}"
            print(message, file=sys.stderr)
            failures.append(message)
            continue
        print(f"Wrote {output}")

    if failures:
        print(
            f"Conversion failed for {len(failures)}/{len(requests)} feeder(s).",
            file=sys.stderr,
        )
        return 1
    print(f"Converted {len(requests)} feeder(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
