#!/usr/bin/env python3
"""Take line impedances and phasing from OpenDSS rather than the converter.

``powerio`` reproduces the feeder topology faithfully but does not carry the
OpenDSS line electrics across unchanged:

* per-length matrices are emitted in whichever unit the source file used, so
  the accompanying length is not always in the same system;
* matrices built from a ``linegeometry`` are re-derived instead of copied, and
  the reduction differs from the one OpenDSS performs;
* a line whose phase count is smaller than its linecode is written with the
  full matrix, where OpenDSS Kron-reduces onto the phases actually present;
* shunt susceptance is evaluated at 50 Hz on 60 Hz feeders.

OpenDSS is the reference these studies are validated against, so the line
electrics are read back from it directly.  Each line is rewritten with the
absolute series impedance in ohms, the pi shunt in siemens, and the phasing
OpenDSS actually solves, which removes every unit and reduction assumption
from the pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from converters._requires import require

dss = require("opendssdirect")

# Series and shunt blocks written onto each line; any per-length remnant of
# these prefixes is cleared first so no stale entry survives the rewrite.
MATRIX_PREFIXES = ("R_series", "X_series", "G_from", "B_from", "G_to", "B_to")

# A switch carries one impedance per phase, so its matrix must be diagonal to
# within the rounding of the values OpenDSS reports.
_SWITCH_MATRIX_TOL = 1e-12

SQRT3 = math.sqrt(3.0)

# Document sections that attach a component to (bus, terminal) pairs.
_SIMPLE_TERMINAL_SECTIONS = {
    "line": ("bus_from", "terminal_map_from", "bus_to", "terminal_map_to"),
    "switch": ("bus_from", "terminal_map_from", "bus_to", "terminal_map_to"),
    "transformer": ("bus_from", "terminal_map_from", "bus_to", "terminal_map_to"),
    "load": ("bus", "terminal_map", None, None),
    "shunt": ("bus", "terminal_map", None, None),
    "voltage_source": ("bus", "terminal_map", None, None),
    "generator": ("bus", "terminal_map", None, None),
}


# OpenDSS load models this pipeline reproduces, mapped onto BMOPF names.
LOAD_MODEL_NAMES = {
    1: "CONSTANT_POWER",
    2: "CONSTANT_IMPEDANCE",
    4: "CVR",
    5: "CONSTANT_CURRENT",
}


class OpenDSSElectricsError(RuntimeError):
    """Raised when a BMOPF line cannot be reconciled with the OpenDSS feeder."""


def _bus_and_nodes(spec: str, phases: int) -> tuple[str, list[str]]:
    """Split an OpenDSS ``bus.n1.n2`` terminal specification."""
    parts = str(spec).split(".")
    nodes = [part for part in parts[1:] if part != ""]
    if not nodes:
        # OpenDSS defaults an unqualified bus to nodes 1..nphases.
        nodes = [str(index + 1) for index in range(phases)]
    return parts[0], nodes


def read_opendss_lines(master_path: Path) -> dict[str, dict]:
    """Return the solved OpenDSS line electrics keyed by lower-case name."""
    dss.Command("Clear")
    dss.Command(f'Redirect "{Path(master_path).expanduser().resolve()}"')
    frequency = dss.Solution.Frequency() or 60.0

    lines: dict[str, dict] = {}
    index = dss.Lines.First()
    while index:
        size = dss.Lines.Phases()
        length = dss.Lines.Length()
        # RMatrix/XMatrix/CMatrix are per unit length in the units the line
        # declares, and Length() is in those same units, so the product is
        # absolute regardless of which unit the source file chose.
        series_r = np.array(dss.Lines.RMatrix(), dtype=float).reshape(size, size)
        series_x = np.array(dss.Lines.XMatrix(), dtype=float).reshape(size, size)
        shunt_c = np.array(dss.Lines.CMatrix(), dtype=float).reshape(size, size)
        bus_from, nodes_from = _bus_and_nodes(dss.Lines.Bus1(), size)
        bus_to, nodes_to = _bus_and_nodes(dss.Lines.Bus2(), size)
        lines[dss.Lines.Name().lower()] = {
            "size": size,
            "nodes_from": nodes_from,
            "nodes_to": nodes_to,
            "bus_from": bus_from,
            "bus_to": bus_to,
            "r_series": series_r * length,
            "x_series": series_x * length,
            # CMatrix is nF per unit length; convert the total to siemens.
            "b_shunt": 2.0 * math.pi * frequency * shunt_c * length * 1e-9,
            "i_max": float(dss.Lines.NormAmps()),
        }
        index = dss.Lines.Next()
    return lines


def _element_enabled(name: str) -> bool:
    """Return whether OpenDSS holds ``Line.<name>`` in service."""
    dss.Circuit.SetActiveElement(f"Line.{name}")
    if dss.CktElement.Name().lower() != f"line.{name}".lower():
        return False
    return bool(dss.CktElement.Enabled())


def _write_matrix(block: dict, prefix: str, values: np.ndarray) -> None:
    """Write a dense matrix into the flattened BMOPF ``<prefix>_i_j`` form."""
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            block[f"{prefix}_{row + 1}_{col + 1}"] = float(values[row, col])


def _referenced_terminals(document: dict) -> dict[str, set[str]]:
    """Collect every (bus, terminal) pair a component actually connects to."""
    referenced: dict[str, set[str]] = {}

    def note(bus, terminals) -> None:
        if bus is None or terminals is None:
            return
        referenced.setdefault(str(bus), set()).update(str(t) for t in terminals)

    def visit(section: dict, fields) -> None:
        bus_a, map_a, bus_b, map_b = fields
        for entry in section.values():
            if not isinstance(entry, dict):
                continue
            if bus_a in entry or bus_b in entry:
                note(entry.get(bus_a), entry.get(map_a))
                note(entry.get(bus_b), entry.get(map_b))
            else:
                # Transformers are nested one level deeper, by winding type.
                visit(entry, fields)

    for section_name, fields in _SIMPLE_TERMINAL_SECTIONS.items():
        section = document.get(section_name)
        if isinstance(section, dict):
            visit(section, fields)

    # A terminal held at ground is part of the bus even with nothing attached.
    for bus_name, bus in (document.get("bus") or {}).items():
        grounded = bus.get("perfectly_grounded_terminals") or []
        if grounded:
            note(bus_name, grounded)
    return referenced


def prune_unreferenced_terminals(document: dict) -> tuple[int, int]:
    """Drop bus terminals no component connects to, and any emptied bus.

    Rewriting the phasing from OpenDSS can leave a bus carrying terminals that
    only the superseded terminal maps mentioned.  Those would enter the model
    as floating nodes, so they are removed once nothing references them.
    """
    referenced = _referenced_terminals(document)
    buses = document.get("bus") or {}
    dropped_terminals = 0
    dropped_buses = []
    for bus_name, bus in buses.items():
        keep = referenced.get(str(bus_name), set())
        terminals = [str(t) for t in bus.get("terminal_names", [])]
        kept = [t for t in terminals if t in keep]
        dropped_terminals += len(terminals) - len(kept)
        if not kept:
            dropped_buses.append(bus_name)
            continue
        bus["terminal_names"] = kept
        grounded = bus.get("perfectly_grounded_terminals")
        if grounded is not None:
            bus["perfectly_grounded_terminals"] = [
                str(t) for t in grounded if str(t) in kept
            ]
    for bus_name in dropped_buses:
        del buses[bus_name]
    return dropped_terminals, len(dropped_buses)


def apply_opendss_line_electrics(document: dict, master_path: Path) -> list[str]:
    """Rewrite every BMOPF line from the OpenDSS feeder it was converted from.

    Returns the notes worth reporting to the caller.  Lines OpenDSS holds out
    of service are removed, because the reference solution does not energise
    them either.
    """
    solved = read_opendss_lines(master_path)
    lines = document.get("line")
    if not isinstance(lines, dict):
        return []

    notes: list[str] = []
    disabled: list[str] = []
    disabled_switches: list[str] = []
    rephased = 0
    for name, line in list(lines.items()):
        entry = solved.get(str(name).lower())
        if entry is None:
            if _element_enabled(str(name)):
                raise OpenDSSElectricsError(
                    f"Line '{name}' has no counterpart in the OpenDSS feeder."
                )
            disabled.append(name)
            continue

        size = entry["size"]
        if len(entry["nodes_from"]) != size or len(entry["nodes_to"]) != size:
            raise OpenDSSElectricsError(
                f"Line '{name}' declares {size} phases but OpenDSS lists "
                f"{len(entry['nodes_from'])}/{len(entry['nodes_to'])} nodes."
            )
        if [str(t) for t in line.get("terminal_map_from", [])] != entry["nodes_from"]:
            rephased += 1

        for prefix in MATRIX_PREFIXES:
            for key in [k for k in line if str(k).startswith(f"{prefix}_")]:
                del line[key]
        line.pop("linecode", None)
        # The matrices below are absolute, so no length may scale them again.
        line.pop("length", None)

        line["terminal_map_from"] = list(entry["nodes_from"])
        line["terminal_map_to"] = list(entry["nodes_to"])
        _write_matrix(line, "R_series", entry["r_series"])
        _write_matrix(line, "X_series", entry["x_series"])
        # The parser hands UnbalancedLine the full pi shunt and lets it halve,
        # so each end carries half of the total charging susceptance.
        half_shunt = 0.5 * entry["b_shunt"]
        _write_matrix(line, "B_from", half_shunt)
        _write_matrix(line, "B_to", half_shunt)
        _write_matrix(line, "G_from", np.zeros_like(half_shunt))
        _write_matrix(line, "G_to", np.zeros_like(half_shunt))
        line["i_max"] = [entry["i_max"]] * size

    # Switches are OpenDSS lines too, so their phasing and their series
    # impedance both come from the same place.  OpenDSS gives a switch a small
    # but real impedance; carrying it means no default has to be invented for
    # a closed switch, which is otherwise a short circuit.
    switches = document.get("switch") or {}
    for name, switch in list(switches.items()):
        entry = solved.get(str(name).lower())
        if entry is None:
            # powerio maps some OpenDSS lines onto switches, so a switch can
            # equally be one that OpenDSS holds out of service.
            if not _element_enabled(str(name)):
                del switches[name]
                disabled_switches.append(name)
            continue
        if [str(t) for t in switch.get("terminal_map_from", [])] != entry["nodes_from"]:
            rephased += 1
        switch["terminal_map_from"] = list(entry["nodes_from"])
        switch["terminal_map_to"] = list(entry["nodes_to"])
        if isinstance(switch.get("i_max"), list):
            switch["i_max"] = [switch["i_max"][0]] * entry["size"]

        # A switch is one impedance per phase, so the matrix has to be
        # diagonal and identical on every phase for a scalar to describe it.
        for label, matrix in (("R", entry["r_series"]), ("X", entry["x_series"])):
            diagonal = np.diag(matrix)
            off_diagonal = np.abs(matrix - np.diag(diagonal)).max()
            spread = float(diagonal.max() - diagonal.min())
            if off_diagonal > _SWITCH_MATRIX_TOL or spread > _SWITCH_MATRIX_TOL:
                raise OpenDSSElectricsError(
                    f"Switch '{name}' has a coupled or unequal {label} matrix, "
                    "which one impedance per phase cannot represent."
                )
        switch["r_series"] = float(np.diag(entry["r_series"])[0])
        switch["x_series"] = float(np.diag(entry["x_series"])[0])

    for name in disabled:
        del lines[name]
    if disabled or disabled_switches:
        out_of_service = sorted(disabled + disabled_switches)
        notes.append(
            f"removed {len(out_of_service)} branch(es) OpenDSS holds out of "
            "service: "
            + ", ".join(out_of_service[:5])
            + ("..." if len(out_of_service) > 5 else "")
        )
    if rephased:
        notes.append(f"rephased {rephased} line(s) onto the conductors OpenDSS solves")

    # Linecodes only carried per-length data, which no line references now.
    document.pop("linecode", None)
    terminals, buses = prune_unreferenced_terminals(document)
    if terminals or buses:
        notes.append(
            f"pruned {terminals} unreferenced bus terminal(s) and {buses} empty bus(es)"
        )
    return notes


def apply_opendss_load_models(document: dict, master_path: Path) -> list[str]:
    """Carry the OpenDSS load model onto each BMOPF load.

    ``powerio`` writes every load as constant power.  Feeders that model
    conservation-voltage-reduction demand (OpenDSS model 4) therefore lose the
    voltage dependence of P and Q, which shifts the secondary voltages by a
    percent or more.  The model number, its two CVR exponents, and the
    per-unit bounds outside which OpenDSS reverts to a constant impedance are
    read back from the feeder.
    """
    dss.Command("Clear")
    dss.Command(f'Redirect "{Path(master_path).expanduser().resolve()}"')

    solved: dict[str, dict] = {}
    index = dss.Loads.First()
    while index:
        dss.Circuit.SetActiveElement(f"Load.{dss.Loads.Name()}")
        solved[dss.Loads.Name().lower()] = {
            "kv": float(dss.Loads.kV()) * 1000.0,
            "phases": int(dss.CktElement.NumPhases()),
            "delta": bool(dss.Loads.IsDelta()),
            "model": int(dss.Loads.Model()),
            "cvr_watts": float(dss.Loads.CVRwatts()),
            "cvr_vars": float(dss.Loads.CVRvars()),
            "v_min_pu": float(dss.Loads.Vminpu()),
            "v_max_pu": float(dss.Loads.Vmaxpu()),
        }
        index = dss.Loads.Next()

    loads = document.get("load")
    if not isinstance(loads, dict):
        return []

    changed = 0
    missing: list[str] = []
    for name, load in loads.items():
        entry = solved.get(str(name).lower())
        if entry is None:
            missing.append(name)
            continue
        model_name = LOAD_MODEL_NAMES.get(entry["model"])
        if model_name is None:
            raise OpenDSSElectricsError(
                f"Load '{name}' uses OpenDSS model {entry['model']}, which this "
                "pipeline does not reproduce."
            )
        if load.get("model") != model_name:
            changed += 1
        load["model"] = model_name
        # OpenDSS states kV line-to-neutral for a single-phase load and
        # line-to-line otherwise.  BMOPF states v_nom line-to-line for WYE and
        # DELTA, and per phase for SINGLE_PHASE, so convert once here instead
        # of leaving the reader to infer which was meant.
        configuration = str(load.get("configuration", "WYE")).upper()
        line_to_neutral = entry["phases"] < 2 and not entry["delta"]
        if configuration == "SINGLE_PHASE":
            v_nom = entry["kv"] if line_to_neutral else entry["kv"] / SQRT3
        else:
            v_nom = entry["kv"] * SQRT3 if line_to_neutral else entry["kv"]
        load["v_nom"] = [v_nom] * max(len(load.get("p_nom") or [1]), 1)
        load["v_min_pu"] = entry["v_min_pu"]
        load["v_max_pu"] = entry["v_max_pu"]
        if model_name == "CVR":
            load["cvr_watts"] = entry["cvr_watts"]
            load["cvr_vars"] = entry["cvr_vars"]

    if missing:
        raise OpenDSSElectricsError(
            f"{len(missing)} BMOPF load(s) have no OpenDSS counterpart, "
            f"e.g. {sorted(missing)[:3]}."
        )
    notes = []
    if changed:
        notes.append(f"restored the OpenDSS load model on {changed} load(s)")
    return notes


def apply_opendss_shunt_states(document: dict, master_path: Path) -> list[str]:
    """Scale each shunt by the capacitor steps OpenDSS actually has in service.

    ``powerio`` writes a capacitor at its full nameplate susceptance whether or
    not the bank is switched in.  A feeder that ships with banks open therefore
    gains reactive support it does not have, which lifts the whole feeder.
    """
    dss.Command("Clear")
    dss.Command(f'Redirect "{Path(master_path).expanduser().resolve()}"')
    dss.Command("Solve")

    in_service: dict[str, float] = {}
    index = dss.Capacitors.First()
    while index:
        steps = dss.Capacitors.NumSteps() or 1
        states = list(dss.Capacitors.States())[:steps]
        closed = sum(1 for state in states if int(state) > 0)
        in_service[dss.Capacitors.Name().lower()] = closed / steps
        index = dss.Capacitors.Next()

    shunts = document.get("shunt")
    if not isinstance(shunts, dict):
        return []

    scaled = 0
    removed: list[str] = []
    for name, shunt in list(shunts.items()):
        fraction = in_service.get(str(name).lower())
        if fraction is None or fraction == 1.0:
            continue
        if fraction == 0.0:
            removed.append(name)
            del shunts[name]
            continue
        for key in [k for k in shunt if str(k).startswith(("B_", "G_"))]:
            shunt[key] = float(shunt[key]) * fraction
        scaled += 1

    notes = []
    if removed:
        notes.append(
            f"removed {len(removed)} switched-out capacitor bank(s): "
            + ", ".join(sorted(removed))
        )
    if scaled:
        notes.append(f"scaled {scaled} partly energised capacitor bank(s)")
    return notes


def _source_impedance_matrix(
    r1: float, x1: float, r0: float, x0: float, phases: int = 3
) -> np.ndarray:
    """Build the phase impedance of a source from its sequence values.

    The self and mutual terms come from the sequence pair the same way at any
    phase count, so a single- or two-phase source is described as readily as a
    three-phase one.
    """
    z1 = complex(r1, x1)
    z0 = complex(r0, x0)
    self_z = (z0 + 2.0 * z1) / 3.0
    mutual_z = (z0 - z1) / 3.0
    matrix = np.full((phases, phases), mutual_z, dtype=complex)
    np.fill_diagonal(matrix, self_z)
    return matrix


def apply_opendss_source_impedance(document: dict, master_path: Path) -> list[str]:
    """Place the voltage source behind its OpenDSS Thevenin impedance.

    ``powerio`` writes the source as an ideal terminal voltage, so the feeder
    is stiffer than OpenDSS models it and every bus sits high under load.  The
    source is moved onto a new internal bus and joined to its original
    terminals by a branch carrying the ``Vsource`` sequence impedance, which is
    how OpenDSS represents it internally.

    Must run after :func:`apply_opendss_line_electrics`, which drops any line
    that has no OpenDSS counterpart.
    """
    dss.Command("Clear")
    dss.Command(f'Redirect "{Path(master_path).expanduser().resolve()}"')

    impedances: dict[str, np.ndarray] = {}
    for source_name in dss.Vsources.AllNames():
        dss.Circuit.SetActiveElement(f"Vsource.{source_name}")

        def _value(field: str) -> float:
            return float(dss.Properties.Value(field) or 0.0)

        impedances[source_name.lower()] = (
            _value("R1"),
            _value("X1"),
            _value("R0"),
            _value("X0"),
        )

    buses = document.setdefault("bus", {})
    lines = document.setdefault("line", {})
    added = 0
    for name, source in (document.get("voltage_source") or {}).items():
        sequence = impedances.get(str(name).lower())
        if sequence is None:
            continue

        bus_name = source["bus"]
        bus = buses.get(bus_name)
        if bus is None:
            raise OpenDSSElectricsError(
                f"Voltage source '{name}' names unknown bus '{bus_name}'."
            )
        grounded = {str(t) for t in bus.get("perfectly_grounded_terminals", [])}
        phases = [str(t) for t in source.get("terminal_map", []) if t not in grounded]
        if not phases:
            raise OpenDSSElectricsError(
                f"Voltage source '{name}' has no phase terminal to sit behind."
            )
        matrix = _source_impedance_matrix(*sequence, phases=len(phases))

        internal_name = f"{bus_name}_source_internal"
        if internal_name in buses:
            raise OpenDSSElectricsError(f"Bus '{internal_name}' already exists.")
        buses[internal_name] = {
            "terminal_names": [str(t) for t in bus.get("terminal_names", [])],
            "perfectly_grounded_terminals": sorted(grounded),
        }
        source["bus"] = internal_name

        branch = {
            "bus_from": internal_name,
            "bus_to": bus_name,
            "terminal_map_from": list(phases),
            "terminal_map_to": list(phases),
        }
        _write_matrix(branch, "R_series", matrix.real)
        _write_matrix(branch, "X_series", matrix.imag)
        lines[f"{name}_source_impedance"] = branch
        added += 1

    return (
        [f"placed {added} source(s) behind the OpenDSS Thevenin impedance"]
        if added
        else []
    )


def canonicalise_bus_names(document: dict) -> list[str]:
    """Point every bus reference at the spelling the bus section declares.

    ``powerio`` does not always spell a bus the same way in the bus section and
    in the components that connect to it.  OpenDSS bus names are
    case-insensitive, so the two are the same bus, but leaving the mismatch in
    the file makes the reader resolve it.
    """
    buses = document.get("bus") or {}
    canonical = {str(name).lower(): str(name) for name in buses}
    fixed = 0

    def repair(entry: dict, field: str) -> None:
        nonlocal fixed
        value = entry.get(field)
        if not isinstance(value, str):
            return
        proper = canonical.get(value.lower())
        if proper is not None and proper != value:
            entry[field] = proper
            fixed += 1

    def visit(section: dict, fields) -> None:
        for entry in section.values():
            if not isinstance(entry, dict):
                continue
            if any(field in entry for field in fields):
                for field in fields:
                    repair(entry, field)
            else:
                visit(entry, fields)

    for section_name, spec in _SIMPLE_TERMINAL_SECTIONS.items():
        section = document.get(section_name)
        if isinstance(section, dict):
            visit(section, [f for f in (spec[0], spec[2]) if f])
    return (
        [f"spelled {fixed} bus reference(s) as the bus section does"] if fixed else []
    )
