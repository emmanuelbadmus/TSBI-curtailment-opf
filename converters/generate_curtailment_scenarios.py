#!/usr/bin/env python3
"""Generate the JSON inputs for the PV curtailment study.

The study root contains one feeder network and its 24 scenario files. This
generator reads ``<root>/<case>/network/bmopf.json``, connects the inverter
and PV definitions at the terminals its study design names, and writes one
file for each control/norm combination. Running it without arguments creates
all 120 scenarios for the five maintained feeders.

A scenario states what varies between scenarios: the network, the PV and
inverter models, the controls, the objective and the solver settings. What
belongs to the feeder as a whole -- where its inverters go, and the voltage
band it is held to -- stays in ``data/dss/<case>/study.json``, which
``FeederLibrary`` reads both here and at solve time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# This module lives in converters/, so the repository root is one level up.
# Put it on the path before importing src, so the script works when run
# directly as `python converters/generate_curtailment_scenarios.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.components.inverter import inverter_connection_id  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "data" / "bmopf_json"
DEFAULT_DSS_ROOT = REPO_ROOT / "data" / "dss"
P_MODES = ("CONSTANT_P", "MPPT")
Q_MODES = ("UPF", "CONSTANT_Q", "CPF", "VOLT_VAR")
NORMS = ("L1", "L2", "LINF")
INVERTER_ID = "inverter_1"
PV_ID = "pv_1"
INVERTER_S_VA = 5800.0
REQUESTED_AC_POWER_PU = 0.9
CONSTANT_Q_SETPOINT_PU = -0.05
CPF_POWER_FACTOR = 0.95
# Ipopt's default MUMPS pivot tolerance (1e-6) is too permissive for these KKT
# systems and the M1 LINF cases stall on it; 1e-2 is too aggressive and breaks
# cases on P and p5rhs0 that the default solves. A sweep over the union of both
# failure sets found 1e-3 to be the only value that solves all of them, and it
# is also the fastest. Conservative pivoting is what the HSL solvers used to
# provide here, which is why this study no longer needs one.
MUMPS_PIVOT_TOLERANCE = 1e-3

RATED_AC_TERMINAL_VOLTAGE_V = 120.0
DC_LINK_VOLTAGE_V = 200.0

# Build-time case metadata. The generated scenario JSON files are the runtime
# source of truth; this embedded table is only used when regenerating them.
# The study design of a feeder: where inverters go and how many. Kept beside
# the feeder in data/dss/<case>/study.json so adding one is a matter of adding
# files, not of editing this module. Every field has a default, so a feeder
# that ships no study.json is still swept like any other.
DEFAULT_STUDY_DESIGN = {
    "terminal_selection": "all_positive_load_terminals",
    "exclude_aggregate": True,
    "repeats": 1,
    "voltage_magnitude_limits_relative_to_base_case": {
        "minimum": 0.95,
        "maximum": 1.05,
    },
}
STUDY_DESIGN_FILENAME = "study.json"


class FeederLibrary:
    """The feeders on disk and the study design that belongs to each.

    A feeder is a directory holding an OpenDSS master file, so the set of
    feeders is whatever has been placed under the feeder root rather than a
    list this module has to be told about.
    """

    def __init__(self, dss_root: Path | None = None, root: Path | None = None):
        self.dss_root = Path(dss_root or DEFAULT_DSS_ROOT)
        self.root = Path(root or DEFAULT_ROOT)

    def cases(self) -> list[str]:
        """Return every feeder name found under the feeder root."""
        return sorted(
            path.parent.parent.name
            for path in self.dss_root.glob("*/network/Master.dss")
        )

    def design(self, case_name: str) -> dict:
        """Return the study design for one feeder, defaults filled in."""
        design = dict(DEFAULT_STUDY_DESIGN)
        stated = self.dss_root / case_name / STUDY_DESIGN_FILENAME
        if stated.is_file():
            given = json.loads(stated.read_text())
            if not isinstance(given, dict):
                raise ValueError(f"{stated} must hold a JSON object")
            # A stated placement replaces the default rather than adding to it.
            if given.keys() & {"terminals", "groups"}:
                design.pop("terminal_selection", None)
            design.update(given)
        design.setdefault("source", f"{case_name}/network/bmopf.json")
        return design

    def voltage_band(self, case_name: str) -> tuple[float, float]:
        """Return one feeder's voltage limits, relative to its base case.

        The limits scale the no-PV base-case magnitude of every bus, so they
        belong to the feeder rather than to any one of its scenarios.
        """
        stated = self.design(case_name).get(
            "voltage_magnitude_limits_relative_to_base_case"
        )
        if not isinstance(stated, dict) or stated.keys() != {"minimum", "maximum"}:
            raise ValueError(
                f"{case_name}: voltage_magnitude_limits_relative_to_base_case "
                "must state exactly a minimum and a maximum"
            )
        minimum, maximum = float(stated["minimum"]), float(stated["maximum"])
        if not 0.0 < minimum < maximum:
            raise ValueError(
                f"{case_name}: voltage limits require 0 < minimum < maximum"
            )
        return (minimum, maximum)

    def bands(self, cases: list[str]) -> dict:
        """Return the voltage band of each named feeder."""
        return {case: self.voltage_band(case) for case in cases}

    def ordered(self, cases: list[str]) -> list[str]:
        """Return the feeders smallest first, which is how a study reads.

        Size is the bus count of the converted network, so the order follows
        the feeders themselves rather than a list stated anywhere. A feeder
        that has not been converted yet sorts last, by name.
        """

        def size(case: str) -> tuple:
            network = self.root / case / "network" / "bmopf.json"
            if not network.is_file():
                return (1, 0, case)
            buses = len(json.loads(network.read_text()).get("bus") or {})
            return (0, buses, case)

        return sorted(cases, key=size)

    def converted(self) -> list[str]:
        """Return every feeder that has already been converted to BMOPF."""
        return [
            case
            for case in self.cases()
            if (self.root / case / "network" / "bmopf.json").is_file()
        ]

    def labels(self) -> dict:
        """Return each feeder's display label, defaulting to its name."""
        return {case: self.design(case).get("label", case) for case in self.cases()}


def inverter_type(
    *,
    dc_link_voltage_v: float,
    ac_terminal_voltage_v: float = 120.0,
) -> dict:
    """Return the electrical inverter type serialized into every case.

    Control settings intentionally do not live in the shared type.  They are
    scenario inputs because a CONSTANT_Q case has no reason to carry MPPT or
    Volt-VAR parameters (and vice versa).
    """
    return {
        "rated_ac_terminal_voltage_v": float(ac_terminal_voltage_v),
        "dc_link_voltage_v": float(dc_link_voltage_v),
        "constant_p_pv_voltage_v": float(dc_link_voltage_v),
        "frequency_hz": 60.0,
        "loss_smoothing_epsilon": 1e-8,
        "duty_cycle_limits": {
            "minimum": 1e-3,
            "maximum": 1.0 - 1e-3,
        },
        "semiconductor": {
            "mosfet": {
                "threshold_v": 0.30,
                "on_resistance_ohm": 25e-3,
                "turn_on_delay_s": 14e-9,
                "rise_time_s": 15e-9,
                "turn_off_delay_s": 58e-9,
                "fall_time_s": 11e-9,
            },
            "diode": {
                "forward_v": 1.10,
                "on_resistance_ohm": 50e-3,
                "recovery_time_s": 75e-9,
            },
        },
        "first_stage_converter": {
            "switching_frequency_hz": 50e3,
            "inductor_resistance_ohm": 1.8e-3,
        },
        "second_stage_converter": {"switching_frequency_hz": 16e3},
        "lcl_filter": {
            "inverter_side_inductance_h": 2.23e-3,
            "grid_side_inductance_h": 0.045e-3,
            "filter_capacitance_f": 15e-6,
            "damping_resistance_ohm": 0.55,
            "inverter_side_resistance_ohm": 5e-3,
            "grid_side_resistance_ohm": 5e-3,
        },
    }


def single_diode_sdm() -> dict:
    """Return the single-diode PV source data used by generated scenarios.

    The module parameters are evaluated at the original reference operating
    point.  The runtime constructs the equivalent array parameters and solves
    the implicit current equation; MPPT scenarios additionally calculate the
    maximum of V_pv * I_pv.
    """
    return {
        "array": {
            "series_modules": 3,
            "parallel_strings": 4,
        },
        "module": {
            "v_oc_v": 75.6,
            "i_sc_a": 6.58,
            "v_mp_v": 65.8,
            "i_mp_a": 6.08,
            "photocurrent_a": 6.58,
            "saturation_current_a": 2.7216651360183327e-16,
            "series_resistance_ohm": 0.7623804042495805,
            "shunt_resistance_ohm": 2.1147484573086397e17,
            "ideality_thermal_voltage_v": 2.0040211714446965,
        },
    }


def network_path(root: Path, case_name: str, entry: dict) -> Path:
    """Return one required feeder network or raise an actionable error."""
    path = root / entry["source"]
    if not path.is_file():
        raise FileNotFoundError(
            f"{case_name}: BMOPF network not found at {path}. "
            "Run dss_to_bmopf.py for this feeder first, or pass "
            "the study root containing <case>/network/bmopf.json with "
            "--root."
        )
    return path


def _top_positive_load_terminals(case_name: str, entry: dict, root: Path) -> list[str]:
    """Select a bounded, deterministic fleet from BMOPF load phases."""
    source = network_path(root, case_name, entry)
    document = json.loads(source.read_text())
    candidates = []
    for load_name, load in (document.get("load") or {}).items():
        if (
            entry.get("exclude_aggregate", False)
            and "aggregate" in str(load_name).lower()
        ):
            continue
        bus_name = load.get("bus")
        if bus_name not in document.get("bus", {}):
            continue
        grounded = {
            str(value)
            for value in document["bus"][bus_name].get(
                "perfectly_grounded_terminals", []
            )
        }
        terminals = [str(value) for value in load.get("terminal_map", [])]
        for position, p_w in enumerate(load.get("p_nom", []) or []):
            if position >= len(terminals) or float(p_w) <= 0.0:
                continue
            phase = terminals[position]
            if phase in grounded:
                continue
            candidates.append(
                (
                    float(p_w),
                    f"{bus_name}.{phase}",
                    str(load_name),
                )
            )
    # Keep one inverter per terminal and use stable lexical tie-breaking.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = []
    offset = int(entry.get("selection_offset", 0))
    count = int(entry.get("selection_count", 48))
    if offset < 0 or count < 0:
        raise ValueError(f"{case_name}: selection offset/count must be non-negative")
    target = offset + count
    seen = set()
    for _p_w, terminal, _load_name in candidates:
        if terminal in seen:
            continue
        seen.add(terminal)
        selected.append(terminal)
        if len(selected) >= target:
            break
    selected = selected[offset:target]
    if not selected:
        raise ValueError(
            f"{case_name}: top_positive_load_terminals selected no terminals"
        )
    return selected


def _all_positive_load_terminals(case_name: str, entry: dict, root: Path) -> list[str]:
    """Return every distinct positive-load phase using the same stable order."""
    selection = dict(entry)
    selection["selection_count"] = sum(
        len(load.get("p_nom", []) or [])
        for load in json.loads(network_path(root, case_name, entry).read_text())
        .get("load", {})
        .values()
    )
    return _top_positive_load_terminals(case_name, selection, root)


def terminals_for(case_name: str, entry: dict, root: Path) -> list[str]:
    """Resolve the terminal-selection rule from the build manifest."""
    if entry.get("terminals") is not None:
        return [str(value) for value in entry["terminals"]]
    if entry.get("terminal_selection") == "all_positive_load_terminals":
        return _all_positive_load_terminals(case_name, entry, root)
    if entry.get("terminal_selection") == "top_positive_load_terminals":
        return _top_positive_load_terminals(case_name, entry, root)
    raise ValueError(f"{case_name}: case manifest has no inverter terminals")


def connection_groups(
    case_name: str, entry: dict, root: Path
) -> list[tuple[list[str], int]]:
    """Resolve manifest groups without duplicating device data in JSON."""
    groups = entry.get("groups") or [entry]
    resolved = []
    for group in groups:
        selection = dict(entry)
        selection.update(group)
        repeats = int(selection.get("repeats", 1))
        if repeats <= 0:
            raise ValueError(f"{case_name}: inverter repeats must be positive")
        resolved.append((terminals_for(case_name, selection, root), repeats))
    return resolved


def solver_settings(case_name: str, p_mode: str, q_mode: str, norm: str) -> dict:
    """Return the explicit, convergence-tested settings for one scenario.

    The nonlinear inverter equations have singular or poorly conditioned
    regions near zero current.  These initial points and linear solvers were
    established by the repository's 120-case regression; they change only the
    numerical path to the same hard-constrained optimum.
    """
    # One initial point and one feasibility tolerance for every case. Earlier
    # revisions varied both per case; a sweep over uniform values showed the
    # variation was not only unnecessary but harmful, since M1 CONSTANT_P_CPF
    # LINF fails at the 1e-3 initial point it had been given and solves at the
    # plain 1e-4. All 120 cases solve on these values, so a reader never has to
    # ask why a particular case was solved differently from the rest.
    initial_delivery_fraction = 1e-4
    constraint_violation_tolerance = 1e-5

    settings = {
        "name": "ipopt",
        # MUMPS is the only linear solver this study needs. The HSL solvers are
        # licence restricted and absent from the conda-forge, Homebrew and pip
        # builds of Ipopt, so requiring one would make the study unrunnable for
        # most people. See MUMPS_PIVOT_TOLERANCE for what replaced it.
        "linear_solver": "mumps",
        "mumps_pivot_tolerance": MUMPS_PIVOT_TOLERANCE,
        "tol": 1e-9,
        "acceptable_tol": 1e-6,
        "constraint_violation_tolerance": constraint_violation_tolerance,
        "max_iter": 5000,
        "timeout_s": 60.0,
        "initial_delivery_fraction": initial_delivery_fraction,
        "acceptable_iter": 20,
        # Ipopt's fallback termination threshold, not this study's acceptance
        # criterion: a case is judged afterwards by the post-solve check in
        # run_curtailment_study.py, which is independent of the solver's own
        # view. Uniform for every case so no feeder is held to a different
        # standard than another.
        "acceptable_constr_viol_tol": 1e-4,
        "honor_original_bounds": "yes",
        "mu_init": 0.1,
        "mu_strategy": "adaptive",
        "retry_mu_strategy": "monotone",
    }
    if norm == "LINF":
        settings.update(
            {
                "linf_epigraph_margin": 5e-5,
                "linf_objective_tolerance": 1e-4,
            }
        )
    return settings


def build_scenario(
    case_name: str,
    entry: dict,
    p_mode: str,
    q_mode: str,
    norm: str,
    root: Path,
) -> dict:
    """Build one standalone JSON scenario for a control and norm combination."""
    source = network_path(root, case_name, entry)
    document = json.loads(source.read_text())
    # A standalone scenario contains the study inputs as first-class BMOPF
    # document sections.  Discard feeder-side annotations and any previous
    # study publication so no hidden or duplicated input survives generation.
    for key in ("extras", "inverter", "inverter_connections", "pv", "study", "solver"):
        document.pop(key, None)
    shared_type = inverter_type(
        dc_link_voltage_v=DC_LINK_VOLTAGE_V,
        ac_terminal_voltage_v=RATED_AC_TERMINAL_VOLTAGE_V,
    )
    if p_mode == "MPPT":
        # MPPT obtains the PV voltage from the active single-diode equation;
        # the fixed CONSTANT_P voltage would be unused in this mode.
        shared_type.pop("constant_p_pv_voltage_v")
    connections = {}
    used_connection_ids = set()
    for terminals, repeats in connection_groups(case_name, entry, root):
        for terminal in terminals:
            bus_name, phase = terminal.rsplit(".", 1)
            actual_bus_name = next(
                (
                    name
                    for name in document["bus"]
                    if str(name).casefold() == bus_name.casefold()
                ),
                None,
            )
            if actual_bus_name is None:
                raise ValueError(
                    f"Bus '{bus_name}' is not present in the BMOPF document"
                )
            canonical_terminal = f"{actual_bus_name}.{phase}"
            for _ in range(repeats):
                name = inverter_connection_id(canonical_terminal, used_connection_ids)
                connections[name] = {
                    "id": INVERTER_ID,
                    "dc_src_id": PV_ID,
                    "terminal": canonical_terminal,
                }

    reactive_control = {"mode": q_mode}
    if q_mode == "CONSTANT_Q":
        reactive_control["reactive_power_setpoint_pu"] = float(CONSTANT_Q_SETPOINT_PU)
    elif q_mode == "CPF":
        reactive_control["power_factor"] = float(CPF_POWER_FACTOR)
    elif q_mode == "VOLT_VAR":
        reactive_control.update(
            {
                "voltage_knees_pu": [0.92, 0.98, 1.02, 1.08],
                "reactive_power_minimum_pu": -0.1,
                "reactive_power_maximum_pu": 0.1,
                "reference_voltage_pu": 1.0,
                "curve_smoothing_epsilon_pu": 1e-3,
            }
        )

    document["inverter"] = {
        "id": INVERTER_ID,
        "model": "pv_grid_inverter",
        "s_va": INVERTER_S_VA,
        "electrical": shared_type,
        "controls": {
            "active_power": {
                "mode": p_mode,
                "requested_ac_power_pu": REQUESTED_AC_POWER_PU,
            },
            "reactive_power": reactive_control,
        },
        "weights": {"curtailment": 1.0},
    }
    # Every scenario carries the full single-diode source, including the
    # CONSTANT_P cases that hold the DC side at a fixed voltage and so never
    # evaluate the I-V curve. The PV array is a property of the case, not of
    # the control mode, and stating it everywhere keeps the scenarios
    # comparable and the file readable on its own.
    document["pv"] = {"id": PV_ID, **single_diode_sdm()}
    document["inverter_connections"] = connections
    # The voltage band is a property of the feeder, not of a control mode, so
    # it stays in data/dss/<case>/study.json and is read from there at solve
    # time rather than copied into all 24 of a feeder's scenarios.
    document["study"] = {
        "type": "curtailment",
        "objective": {
            "norm": norm,
        },
    }
    document["solver"] = solver_settings(case_name, p_mode, q_mode, norm)
    _set_generated_network_reference(document, case_name)
    document.setdefault("meta", {}).pop("reproducibility", None)
    # The filename and control sections identify a scenario. Duplicating that
    # identity in BMOPF metadata would make it easy for the two values to drift.
    document["meta"].pop("scenario", None)
    document["meta"].pop("case", None)
    document["meta"].pop("frequency", None)
    # A scenario embeds a copy of the feeder network. Stamping which network
    # that copy came from lets the runner refuse a scenario left behind by an
    # earlier conversion, instead of silently solving a stale feeder.
    document["meta"]["network_sha256"] = network_fingerprint(
        network_path(root, case_name, entry)
    )
    return document


# The sections a scenario copies from the feeder network. Hashing these rather
# than a whole file lets the same fingerprint be taken of a scenario's embedded
# copy and of the feeder it came from, so the two can be compared directly.
NETWORK_SECTIONS = (
    "bus",
    "line",
    "switch",
    "transformer",
    "shunt",
    "voltage_source",
    "load",
)


def network_fingerprint(source) -> str:
    """Return a SHA-256 over the network a document carries or describes."""
    document = (
        source if isinstance(source, dict) else json.loads(Path(source).read_text())
    )
    subset = {key: document.get(key) for key in NETWORK_SECTIONS}
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def clean_network_document(document: dict) -> None:
    """Drop the sections a curtailment study does not read.

    powerio publishes transformer no-load admittance under 'extras', which
    this model does not represent. Carrying it would put 71 KB of unread
    numbers in every network file and imply the study uses them.
    """
    document.pop("extras", None)


def _set_generated_network_reference(document: dict, case_name: str) -> None:
    """Keep the BMOPF provenance link valid inside generated study files."""
    sources = (document.get("meta") or {}).get("data_sources")
    if not isinstance(sources, list):
        return
    target = f"data/bmopf_json/{case_name}/network/bmopf.json"
    for source in sources:
        if not isinstance(source, dict):
            continue
        if source.get("format") == "BMOPF JSON" or source.get("name") == case_name:
            source["url"] = target


def generate_all(
    root: Path = DEFAULT_ROOT,
    cases: list[str] | None = None,
    dss_root: Path | None = None,
) -> list[Path]:
    """Generate the requested standalone scenario files and return their paths."""
    root = Path(root).expanduser().resolve()
    library = FeederLibrary(dss_root=dss_root, root=root)
    available = library.cases()
    selected = list(available) if cases is None else list(cases)
    unknown = [case for case in selected if case not in available]
    if unknown:
        raise ValueError(
            f"no feeder named {', '.join(unknown)} under {library.dss_root}; "
            f"found {', '.join(available) or 'none'}"
        )
    written = []
    for case_name in selected:
        entry = library.design(case_name)
        source = network_path(root, case_name, entry)
        document = json.loads(source.read_text())
        clean_network_document(document)
        source.write_text(json.dumps(document, indent=2) + "\n")
        case_dir = root / case_name / "scenarios"
        case_dir.mkdir(parents=True, exist_ok=True)
        for p_mode in P_MODES:
            for q_mode in Q_MODES:
                for norm in NORMS:
                    scenario = build_scenario(
                        case_name,
                        entry,
                        p_mode,
                        q_mode,
                        norm,
                        root,
                    )
                    path = case_dir / f"{p_mode}_{q_mode}_{norm}.json"
                    path.write_text(json.dumps(scenario, indent=2) + "\n")
                    written.append(path)
    expected = 24 * len(selected)
    if len(written) != expected:
        raise RuntimeError(
            f"expected {expected} scenario files, generated {len(written)}"
        )
    return written


def main(argv: list[str] | None = None) -> int:
    """Generate scenarios from the maintained BMOPF feeder networks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="study root containing <case>/network/bmopf.json",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="a feeder under the feeder root; repeat for several (default: all)",
    )
    args = parser.parse_args(argv)
    try:
        paths = generate_all(args.root, args.case)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(paths)} standalone scenario JSON files under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
