"""Load BMOPF JSON networks into the internal component objects.

BMOPF is the draft multiconductor distribution data model of the IEEE PES
Task Force on Benchmarking Multiconductor OPF. This parser reads a BMOPF
document into ``DxNetworkModel``: buses, slacks, lines, transformers, loads,
generators, and the standalone inverter-study sections. A case can then run
without OpenDSS in the solver loop.

Terminal names are the BMOPF conductor labels, which for a
powerio-converted OpenDSS feeder are the DSS node numbers ("1", "2", "3"
for the phases, "4" for the neutral).  Perfectly grounded terminals hold
no state, so they become ``DxNetworkModel.GROUND_NODE`` instead of bus
objects, matching what the DSS parser builds for a grounded-wye winding.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from src.components.inverter import (
    InverterSpecError,
    dispatch_from_instance,
    validate_inverter_instance,
    validate_inverter_type,
)
from src.components.pv import PVSpecError, validate_sdm
from src.components.switch import SwitchStatus
from src.network_loader import DxNetworkModel, reset_component_state

SQRT3 = math.sqrt(3.0)

# Default initial angle per DSS-style phase label; used when the case has no
# voltage source on the terminal to take an angle from.
PHASE_ANGLE_RAD = {
    "1": 0.0,
    "2": -2.0 * math.pi / 3.0,
    "3": 2.0 * math.pi / 3.0,
    "A": 0.0,
    "B": -2.0 * math.pi / 3.0,
    "C": 2.0 * math.pi / 3.0,
}

# BMOPF voltage-dependence names to the internal LoadModel codes.
LOAD_MODEL_CODES = {
    "CONSTANT_POWER": 1,
    "CONSTANT_IMPEDANCE": 2,
    # Voltage-dependent demand: P and Q follow the terminal voltage raised to
    # 'cvr_watts' and 'cvr_vars'.  This is OpenDSS load model 4.
    "CVR": 4,
    "CONSTANT_CURRENT": 5,
}

# Sections of the draft schema this parser does not build yet.  They are
# rejected loudly rather than dropped, so a case is never silently altered.
UNSUPPORTED_SECTIONS = ("capacitor",)

# What this parser understands, as allow lists.  A deny list only catches the
# kinds someone thought to name: powerio 0.10 began emitting J1's voltage
# regulators as "single_phase_autotransformer", which was on no deny list, so
# the regulators were dropped without a word and J1's base power flow turned
# infeasible.  Anything not named here is rejected instead.
SUPPORTED_TRANSFORMERS = ("single_phase", "delta_wye")
SUPPORTED_SECTIONS = (
    # electrical content
    "bus",
    "voltage_source",
    "line",
    "linecode",
    "load",
    "shunt",
    "switch",
    "transformer",
    "generator",
    # standalone inverter scenarios
    "inverter",
    "inverter_connections",
    "pv",
    "study",
    "solver",
    # descriptive, carried through untouched
    "meta",
    "name",
    "terminal_conventions",
    "extras",
)


class ScenarioSelectionError(ValueError):
    """Raised when a study root holds a different set of scenarios than asked."""


class BMOPFParseError(ValueError):
    """Raised when a BMOPF document cannot be mapped onto the components."""


def _matrix(block: dict, prefix: str, size: int) -> np.ndarray:
    """Read a flattened BMOPF ``<prefix>_i_j`` matrix into a dense array."""
    out = np.zeros((size, size), dtype=float)
    for row in range(size):
        for col in range(size):
            out[row, col] = float(block.get(f"{prefix}_{row + 1}_{col + 1}", 0.0))
    return out


def _has_matrix(block: dict, prefix: str) -> bool:
    """Say whether a block carries any entry of a flattened matrix."""
    return any(str(key).startswith(f"{prefix}_") for key in block)


class BMOPFParser:
    """Build a :class:`DxNetworkModel` from a BMOPF JSON document."""

    def __init__(self, source):
        self.source_path = None
        if isinstance(source, dict):
            self.doc = source
        else:
            self.source_path = str(Path(source).expanduser().resolve())
            with open(self.source_path) as handle:
                self.doc = json.load(handle)

        self.warnings: list[str] = []
        self.bases: dict[tuple[str, str], float] = {}
        self.angles: dict[tuple[str, str], float] = {}
        self.active_terminals = self._find_active_terminals()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def parse(self) -> DxNetworkModel:
        """Build the network this document describes."""
        self._reject_unsupported()
        self._validate_standalone_study()
        self._resolve_terminal_voltages()

        reset_component_state()
        model = DxNetworkModel()
        model.source_format = "bmopf"
        model.source_path = self.source_path
        model.input_file_path = self.source_path
        model.bmopf_name = self.doc.get("name")
        model.bmopf_meta = self.doc.get("meta", {})
        model.study_settings = self.doc.get("study") or {}
        model.solver_settings = self.doc.get("solver") or {}

        self._create_buses(model)
        self._create_slacks(model)
        self._create_lines(model)
        self._create_shunts(model)
        self._create_switches(model)
        self._create_transformers(model)
        self._create_loads(model)
        self._create_generators(model)

        model.warnings = list(self.warnings)
        return model

    # ------------------------------------------------------------------
    # Document helpers
    # ------------------------------------------------------------------

    def _warn(self, message: str) -> None:
        """Record a finding once, however many phases raise it."""
        # Per-phase construction repeats the same finding; report it once.
        if message not in self.warnings:
            self.warnings.append(message)

    def _validate_standalone_study(self) -> None:
        """Validate the compact SDM/inverter contract before building a model."""

        def require_exact_keys(block: dict, expected: set[str], context: str) -> None:
            present = set(block)
            missing = sorted(expected - present)
            extra = sorted(present - expected)
            if missing or extra:
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if extra:
                    details.append("unused/unsupported " + ", ".join(extra))
                raise BMOPFParseError(f"{context}: {'; '.join(details)}")

        def finite_number(block: dict, key: str, context: str) -> float:
            try:
                result = float(block[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise BMOPFParseError(f"{context}.{key} must be numeric") from exc
            if not math.isfinite(result):
                raise BMOPFParseError(f"{context}.{key} must be finite")
            return result

        stray_extras = self.doc.get("extras")
        if isinstance(stray_extras, dict) and any(
            key in stray_extras
            for key in (
                "inverter",
                "inverter_connections",
                "inverter_types",
                "inverters",
                "pv",
                "study",
                "solver",
            )
        ):
            raise BMOPFParseError(
                "legacy inverter-study data found in extras; move inverter, "
                "inverter_connections, pv, study, and solver to the top level"
            )
        has_new_inverter = "inverter" in self.doc or "inverter_connections" in self.doc
        if not has_new_inverter:
            return

        if "extras" in self.doc:
            raise BMOPFParseError(
                "standalone inverter cases must publish study inputs at the top level; "
                "remove the legacy 'extras' wrapper"
            )

        definition = self.doc.get("inverter")
        connections = self.doc.get("inverter_connections")
        if not isinstance(definition, dict) or not definition:
            raise BMOPFParseError(
                "standalone inverter cases require one non-empty inverter definition"
            )
        if not isinstance(connections, dict) or not connections:
            raise BMOPFParseError(
                "standalone inverter cases require a non-empty inverter_connections map"
            )
        require_exact_keys(
            definition,
            {"id", "model", "s_va", "electrical", "controls", "weights"},
            "inverter",
        )
        duplicate_sections = ("inverter_types", "inverters", "ibr", "control_profile")
        duplicates = [name for name in duplicate_sections if name in self.doc]
        if duplicates:
            raise BMOPFParseError(
                "standalone cases must publish one inverter definition; remove duplicate "
                f"sections: {', '.join(duplicates)}"
            )

        inverter_id = definition.get("id")
        if not isinstance(inverter_id, str) or not inverter_id.strip():
            raise BMOPFParseError("standalone inverter cases require inverter.id")
        try:
            inverter_s_va = float(definition["s_va"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BMOPFParseError(
                "standalone inverter cases require numeric inverter.s_va"
            ) from exc
        if not math.isfinite(inverter_s_va) or inverter_s_va <= 0.0:
            raise BMOPFParseError("inverter.s_va must be positive and finite")
        inverter_model = definition.get("model")
        if inverter_model != "pv_grid_inverter":
            raise BMOPFParseError(
                "standalone inverter cases require inverter.model='pv_grid_inverter'"
            )
        electrical = definition.get("electrical")
        if not isinstance(electrical, dict) or not electrical:
            raise BMOPFParseError(
                "standalone inverter cases require inverter.electrical"
            )
        if "type" in definition:
            raise BMOPFParseError(
                "inverter must not contain the ambiguous type wrapper; use electrical"
            )
        if "rating_va" in electrical:
            raise BMOPFParseError(
                "inverter.electrical must not duplicate inverter.s_va as rating_va"
            )
        controls = definition.get("controls")
        if not isinstance(controls, dict):
            raise BMOPFParseError("inverter.controls must be an object")
        require_exact_keys(
            controls,
            {"active_power", "reactive_power"},
            "inverter.controls",
        )
        active_power = (
            controls.get("active_power") if isinstance(controls, dict) else None
        )
        active_mode = (
            str(active_power.get("mode", "")).upper()
            if isinstance(active_power, dict)
            else ""
        )
        if active_mode not in ("CONSTANT_P", "MPPT"):
            raise BMOPFParseError(
                "inverter.controls.active_power.mode must be CONSTANT_P or MPPT"
            )
        try:
            validated_type = validate_inverter_type(
                electrical, active_power_mode=active_mode
            )
        except InverterSpecError as exc:
            raise BMOPFParseError(str(exc)) from exc
        pv = self.doc.get("pv")
        if not isinstance(pv, dict):
            raise BMOPFParseError(
                "standalone inverter cases require a pv source definition"
            )
        pv_id = pv.get("id")
        if not isinstance(pv_id, str) or not pv_id.strip():
            raise BMOPFParseError("standalone inverter cases require pv.id")
        # The single-diode source is required in every case, including
        # CONSTANT_P, which holds the DC voltage fixed and never evaluates the
        # I-V curve. It is still validated there, so a case cannot carry a
        # malformed array that a later change to MPPT would silently rely on.
        require_exact_keys(pv, {"id", "array", "module"}, "pv")
        try:
            validate_sdm(pv)
        except (InverterSpecError, PVSpecError) as exc:
            raise BMOPFParseError(f"pv: {exc}") from exc

        for name, connection in connections.items():
            if not isinstance(connection, dict):
                raise BMOPFParseError(
                    f"inverter_connections[{name!r}] must be an object"
                )
            require_exact_keys(
                connection,
                {"id", "dc_src_id", "terminal"},
                f"inverter_connections[{name!r}]",
            )
            connection_id = connection.get("id")
            if connection_id != inverter_id:
                raise BMOPFParseError(
                    f"inverter connection {name!r} references id {connection_id!r}; "
                    f"expected shared inverter id {inverter_id!r}"
                )
            dc_src_id = connection.get("dc_src_id")
            if dc_src_id != pv_id:
                raise BMOPFParseError(
                    f"inverter connection {name!r} references dc_src_id {dc_src_id!r}; "
                    f"expected shared PV id {pv_id!r}"
                )
            if "s_va" in connection:
                raise BMOPFParseError(
                    f"inverter connection {name!r} duplicates inverter.s_va; "
                    "remove connection.s_va"
                )
            try:
                validate_inverter_instance(
                    {
                        "type": inverter_id,
                        "terminal": connection.get("terminal"),
                        "s_va": inverter_s_va,
                        "active_power": copy.deepcopy(active_power),
                        "reactive_power": copy.deepcopy(
                            controls.get("reactive_power") or {}
                        ),
                        "weights": copy.deepcopy(definition.get("weights") or {}),
                    },
                    {inverter_id: validated_type},
                )
            except (InverterSpecError, PVSpecError) as exc:
                raise BMOPFParseError(f"inverter {name}: {exc}") from exc

        study = self.doc.get("study")
        if (
            not isinstance(study, dict)
            or study.get("type") != "curtailment"
            or not isinstance(study.get("objective"), dict)
        ):
            raise BMOPFParseError(
                "standalone inverter cases require study.type='curtailment' "
                "and study.objective"
            )
        # The voltage band is not stated here: it belongs to the feeder's
        # study design, which FeederLibrary.voltage_band reads and checks.
        require_exact_keys(study, {"type", "objective"}, "study")
        require_exact_keys(study["objective"], {"norm"}, "study.objective")
        norm = str(study["objective"].get("norm", "")).upper()
        if norm not in ("L1", "L2", "LINF"):
            raise BMOPFParseError("study.objective.norm must be one of L1, L2, or LINF")
        solver = self.doc.get("solver")
        if not isinstance(solver, dict) or not solver.get("name"):
            raise BMOPFParseError(
                "standalone inverter cases require solver with a solver name"
            )
        common_solver_fields = {
            "name",
            "linear_solver",
            "mumps_pivot_tolerance",
            "tol",
            "acceptable_tol",
            "constraint_violation_tolerance",
            "max_iter",
            "timeout_s",
            "initial_delivery_fraction",
            "acceptable_iter",
            "acceptable_constr_viol_tol",
            "honor_original_bounds",
            "mu_init",
            "mu_strategy",
            "retry_mu_strategy",
        }
        expected_solver_fields = set(common_solver_fields)
        if norm == "LINF":
            expected_solver_fields.update(
                {"linf_epigraph_margin", "linf_objective_tolerance"}
            )
        require_exact_keys(solver, expected_solver_fields, "solver")
        if solver["name"] != "ipopt":
            raise BMOPFParseError(
                "standalone inverter cases support solver.name='ipopt'"
            )
        if not isinstance(solver["linear_solver"], str) or not solver["linear_solver"]:
            raise BMOPFParseError("solver.linear_solver must be a non-empty string")
        for key in (
            "tol",
            "acceptable_tol",
            "constraint_violation_tolerance",
            "timeout_s",
            "acceptable_constr_viol_tol",
            "mu_init",
            "mumps_pivot_tolerance",
        ):
            if finite_number(solver, key, "solver") <= 0.0:
                raise BMOPFParseError(f"solver.{key} must be positive")
        for key in ("max_iter", "acceptable_iter"):
            value = finite_number(solver, key, "solver")
            if value <= 0.0 or value != int(value):
                raise BMOPFParseError(f"solver.{key} must be a positive integer")
        initial_fraction = finite_number(solver, "initial_delivery_fraction", "solver")
        if not 0.0 < initial_fraction <= 1.0:
            raise BMOPFParseError("solver.initial_delivery_fraction must be in (0, 1]")
        if solver["honor_original_bounds"] not in ("yes", "no"):
            raise BMOPFParseError("solver.honor_original_bounds must be 'yes' or 'no'")
        for key in ("mu_strategy", "retry_mu_strategy"):
            if solver[key] not in ("adaptive", "monotone"):
                raise BMOPFParseError(f"solver.{key} must be 'adaptive' or 'monotone'")
        if norm == "LINF":
            for key in ("linf_epigraph_margin", "linf_objective_tolerance"):
                if finite_number(solver, key, "solver") <= 0.0:
                    raise BMOPFParseError(f"solver.{key} must be positive")

    def _section(self, name: str) -> dict:
        """Return one top-level section of the document as a mapping."""
        section = self.doc.get(name) or {}
        if not isinstance(section, dict):
            raise BMOPFParseError(f"Section '{name}' must be an object.")
        return section

    def _grounded(self, bus_name: str) -> set[str]:
        """Return the terminals of a bus that are held at ground."""
        bus_name = self._canonical_bus_name(bus_name)
        bus = self._section("bus").get(bus_name)
        if bus is None:
            raise BMOPFParseError(f"Unknown bus '{bus_name}'.")
        grounded = {str(t) for t in bus.get("perfectly_grounded_terminals", [])}
        conventions = self.doc.get("terminal_conventions") or {}
        grounded.update(
            str(t)
            for t in conventions.get("ground", [])
            if str(t) in {str(x) for x in bus.get("terminal_names", [])}
        )
        return grounded

    def _find_active_terminals(self) -> dict[str, set[str]]:
        """Return bus terminals referenced by at least one network element.

        Some BMOPF conversions retain all three phase labels on a sparse
        single-phase bus even when two of those labels never occur in any
        element.  Creating optimization variables for those labels changes
        neither the circuit nor the solution, but it makes large cases much
        harder to solve.  This index prunes only unreferenced labels; the JSON
        remains unchanged and every referenced conductor is preserved.
        """
        active: dict[str, set[str]] = {}

        def add(bus_name, terminals):
            if bus_name is None:
                return
            canonical = self._canonical_bus_name(bus_name)
            active.setdefault(canonical, set()).update(str(t) for t in terminals)

        for source in self._section("voltage_source").values():
            add(source.get("bus"), source.get("terminal_map", []))
        for line in self._section("line").values():
            add(line.get("bus_from"), line.get("terminal_map_from", []))
            add(line.get("bus_to"), line.get("terminal_map_to", []))
        for load in self._section("load").values():
            add(load.get("bus"), load.get("terminal_map", []))
        for generator in self._section("generator").values():
            add(generator.get("bus"), generator.get("terminal_map", []))
        for shunt in self._section("shunt").values():
            add(shunt.get("bus"), shunt.get("terminal_map", []))
        for switch in self._section("switch").values():
            add(switch.get("bus_from"), switch.get("terminal_map_from", []))
            add(switch.get("bus_to"), switch.get("terminal_map_to", []))
        transformers = self._section("transformer")
        for records in transformers.values():
            for transformer in records.values():
                add(
                    transformer.get("bus_from"),
                    transformer.get("terminal_map_from", []),
                )
                add(transformer.get("bus_to"), transformer.get("terminal_map_to", []))
        for instance in (self.doc.get("inverter_connections") or {}).values():
            terminal = str(instance.get("terminal", ""))
            if "." in terminal:
                bus_name, phase = terminal.rsplit(".", 1)
                add(bus_name, [phase])
        return active

    def _phase_terminals(self, bus_name: str) -> list[str]:
        """Return the terminals of a bus that carry a phase."""
        bus_name = self._canonical_bus_name(bus_name)
        bus = self._section("bus")[bus_name]
        grounded = self._grounded(bus_name)
        declared = [str(t) for t in bus.get("terminal_names", [])]
        referenced = self.active_terminals.get(bus_name)
        if referenced:
            declared = [terminal for terminal in declared if terminal in referenced]
        return [terminal for terminal in declared if terminal not in grounded]

    def _canonical_bus_name(self, bus_name: str) -> str:
        """Resolve harmless case-only bus-name differences in BMOPF records."""
        bus_name = str(bus_name)
        buses = self._section("bus")
        if bus_name in buses:
            return bus_name
        folded = bus_name.casefold()
        matches = [name for name in buses if str(name).casefold() == folded]
        if len(matches) == 1:
            self._warn(
                f"BMOPF bus reference '{bus_name}' uses different casing; "
                f"using '{matches[0]}'."
            )
            return matches[0]
        return bus_name

    def _reject_unsupported(self) -> None:
        """Refuse a document using a section this parser cannot build."""
        for name in UNSUPPORTED_SECTIONS:
            if self._section(name):
                raise BMOPFParseError(
                    f"BMOPF section '{name}' is present but not supported by this "
                    "parser yet."
                )
        unknown_sections = sorted(set(self.doc) - set(SUPPORTED_SECTIONS))
        if unknown_sections:
            raise BMOPFParseError(
                "BMOPF document has section(s) this parser does not read: "
                + ", ".join(repr(name) for name in unknown_sections)
                + ". Refusing rather than dropping them, because an ignored "
                "section silently changes the network."
            )
        unknown_kinds = sorted(
            set(self._section("transformer")) - set(SUPPORTED_TRANSFORMERS)
        )
        if unknown_kinds:
            raise BMOPFParseError(
                "BMOPF transformer kind(s) not supported by this parser: "
                + ", ".join(repr(kind) for kind in unknown_kinds)
                + ". Supported kinds are "
                + ", ".join(repr(kind) for kind in SUPPORTED_TRANSFORMERS)
                + ". Dropping one would silently change the network."
            )
        if not self._section("bus"):
            raise BMOPFParseError("BMOPF document has no 'bus' section.")
        if not self._section("voltage_source"):
            raise BMOPFParseError("BMOPF document has no 'voltage_source' section.")

    # ------------------------------------------------------------------
    # Voltage bases
    # ------------------------------------------------------------------

    def _resolve_terminal_voltages(self) -> None:
        """Seed per-terminal voltage bases and propagate them along lines.

        BMOPF buses carry no voltage base of their own, so the bases come
        from the elements that do publish one, the source magnitudes and
        the transformer winding voltages, and then travel across lines,
        which join terminals at the same voltage.
        """
        for source in self._section("voltage_source").values():
            bus_name = self._canonical_bus_name(source["bus"])
            grounded = self._grounded(bus_name)
            magnitudes = source.get("v_magnitude", [])
            angles = source.get("v_angle", [])
            for position, terminal in enumerate(source.get("terminal_map", [])):
                terminal = str(terminal)
                if terminal in grounded:
                    continue
                if position < len(magnitudes) and magnitudes[position] > 0.0:
                    self.bases[(bus_name, terminal)] = float(magnitudes[position])
                if position < len(angles):
                    self.angles[(bus_name, terminal)] = float(angles[position])

        for xfmr in self._section("transformer").get("single_phase", {}).values():
            for side in ("from", "to"):
                bus_name = self._canonical_bus_name(xfmr[f"bus_{side}"])
                v_nom = float(xfmr[f"v_nom_{side}"])
                grounded = self._grounded(bus_name)
                for terminal in xfmr[f"terminal_map_{side}"]:
                    terminal = str(terminal)
                    if terminal in grounded or v_nom <= 0.0:
                        continue
                    self.bases.setdefault((bus_name, terminal), v_nom)

        # Delta-wye nameplate voltages are line-to-line.  The internal
        # component model uses phase-to-neutral voltage bases on every bus,
        # including the phase branches of a delta winding.
        for xfmr in self._section("transformer").get("delta_wye", {}).values():
            for side in ("from", "to"):
                v_nom = float(xfmr.get(f"v_nom_{side}", 0.0)) / SQRT3
                if v_nom <= 0.0:
                    continue
                bus_name = self._canonical_bus_name(xfmr[f"bus_{side}"])
                grounded = self._grounded(bus_name)
                for terminal in xfmr.get(f"terminal_map_{side}", []):
                    terminal = str(terminal)
                    if terminal not in grounded:
                        self.bases.setdefault((bus_name, terminal), v_nom)

        self._propagate_bases_along_lines()

        missing = [
            (bus_name, terminal)
            for bus_name in self._section("bus")
            for terminal in self._phase_terminals(bus_name)
            if (bus_name, terminal) not in self.bases
        ]
        if missing:
            listed = ", ".join(f"{bus}.{terminal}" for bus, terminal in missing)
            raise BMOPFParseError(
                "No voltage base could be resolved for these terminals: "
                f"{listed}. Every terminal must reach a voltage source or a "
                "transformer winding through lines."
            )

    def _propagate_bases_along_lines(self) -> None:
        """Carry voltage bases across lines and switches, which do not change them."""
        edges = []
        for line in self._section("line").values():
            from_map = [str(t) for t in line["terminal_map_from"]]
            to_map = [str(t) for t in line["terminal_map_to"]]
            for from_terminal, to_terminal in zip(from_map, to_map, strict=True):
                edges.append(
                    (
                        (self._canonical_bus_name(line["bus_from"]), from_terminal),
                        (self._canonical_bus_name(line["bus_to"]), to_terminal),
                    )
                )

        # Closed switches also connect voltage-equivalent terminals.  Open
        # switches intentionally do not provide a propagation edge.
        for switch in self._section("switch").values():
            if bool(switch.get("open_switch", False)):
                continue
            for from_terminal, to_terminal in zip(
                switch.get("terminal_map_from", []),
                switch.get("terminal_map_to", []),
                strict=True,
            ):
                edges.append(
                    (
                        (
                            self._canonical_bus_name(switch["bus_from"]),
                            str(from_terminal),
                        ),
                        (
                            self._canonical_bus_name(switch["bus_to"]),
                            str(to_terminal),
                        ),
                    )
                )

        # A BMOPF bus is a common voltage level.  Some converted feeders list
        # an unenergized phase terminal even when no line record carries it;
        # use another terminal on that same bus to resolve its base without
        # inventing a separate voltage level.
        for bus_name in self._section("bus"):
            terminals = self._phase_terminals(bus_name)
            for left in terminals:
                for right in terminals:
                    if left != right:
                        edges.extend([((bus_name, left), (bus_name, right))])

        changed = True
        while changed:
            changed = False
            for left, right in edges:
                for src, dst in ((left, right), (right, left)):
                    if src in self.bases and dst not in self.bases:
                        self.bases[dst] = self.bases[src]
                        changed = True

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _create_buses(self, model: DxNetworkModel) -> None:
        """Build a bus object for every phase terminal in the document."""
        source_buses = {src["bus"] for src in self._section("voltage_source").values()}
        # Keep the source bus first so its internal indices match the DSS path.
        ordered = [name for name in self._section("bus") if name in source_buses]
        ordered += [name for name in self._section("bus") if name not in source_buses]

        for bus_name in ordered:
            for terminal in self._phase_terminals(bus_name):
                angle = self.angles.get(
                    (bus_name, terminal), PHASE_ANGLE_RAD.get(terminal, 0.0)
                )
                model.create_bus(
                    self.bases[(bus_name, terminal)],
                    angle,
                    bus_name,
                    None,
                    terminal,
                    False,
                )

    def _bus_obj(self, model: DxNetworkModel, bus_name: str, terminal):
        """Return the bus for a terminal, or GROUND_NODE when it is grounded."""
        bus_name = self._canonical_bus_name(bus_name)
        terminal = str(terminal)
        if terminal in self._grounded(bus_name):
            return DxNetworkModel.GROUND_NODE
        bus = model.bus_name_map.get(f"{bus_name}_{terminal}")
        if bus is None:
            raise BMOPFParseError(f"Terminal '{bus_name}.{terminal}' has no bus.")
        return bus

    def _create_slacks(self, model: DxNetworkModel) -> None:
        """Build the voltage sources that hold the network's reference."""
        from src.components.slack import Slack

        for name, source in self._section("voltage_source").items():
            bus_name = source["bus"]
            grounded = self._grounded(bus_name)
            magnitudes = source.get("v_magnitude", [])
            angles = source.get("v_angle", [])
            for position, terminal in enumerate(source.get("terminal_map", [])):
                terminal = str(terminal)
                if terminal in grounded:
                    continue
                bus = self._bus_obj(model, bus_name, terminal)
                magnitude = (
                    float(magnitudes[position]) if position < len(magnitudes) else 0.0
                )
                angle = float(angles[position]) if position < len(angles) else 0.0
                slack = Slack(bus, magnitude, angle)
                slack.name = name
                model.slack.append(slack)

        if not model.slack:
            raise BMOPFParseError(
                "No slack terminals were built from 'voltage_source'."
            )

    def _line_impedance(self, name: str, line: dict, size: int):
        """Return (Z per unit length, Y_shunt per unit length, length)."""
        linecode_name = line.get("linecode")
        if linecode_name is not None:
            block = self._section("linecode").get(linecode_name)
            if block is None:
                raise BMOPFParseError(
                    f"Line '{name}' references unknown linecode '{linecode_name}'."
                )
            length = float(line.get("length", 0.0))
            if length <= 0.0:
                raise BMOPFParseError(
                    f"Line '{name}' uses linecode '{linecode_name}' but has no length."
                )
            # The schema states linecode matrices per metre.
            scale = 1.0
        elif _has_matrix(line, "R_series"):
            # Inline matrices are absolute ohms; carry them on a unit length.
            block = line
            length = 1.0
            scale = 1.0
        else:
            raise BMOPFParseError(f"Line '{name}' carries no impedance data.")

        impedance = (
            _matrix(block, "R_series", size) + 1j * _matrix(block, "X_series", size)
        ) * scale

        shunt = None
        if _has_matrix(block, "B_from") or _has_matrix(block, "G_from"):
            y_from = _matrix(block, "G_from", size) + 1j * _matrix(
                block, "B_from", size
            )
            y_to = _matrix(block, "G_to", size) + 1j * _matrix(block, "B_to", size)
            if _has_matrix(block, "B_to") and not np.allclose(y_from, y_to):
                self._warn(
                    f"Line '{name}' has different 'from' and 'to' shunt matrices; the "
                    "pi model splits one matrix evenly, so the 'from' side is used."
                )
            # BMOPF states the shunt already halved per line end; UnbalancedLine
            # halves whatever it is handed, so hand it the full pi shunt.
            shunt = 2.0 * y_from * scale

        return impedance, shunt, length

    def _create_lines(self, model: DxNetworkModel) -> None:
        """Build the lines, with their series impedance and charging."""
        for name, line in self._section("line").items():
            from_bus_name = line["bus_from"]
            to_bus_name = line["bus_to"]
            from_map = [str(t) for t in line["terminal_map_from"]]
            to_map = [str(t) for t in line["terminal_map_to"]]
            if len(from_map) != len(to_map):
                raise BMOPFParseError(
                    f"Line '{name}' maps {len(from_map)} 'from' terminals onto "
                    f"{len(to_map)} 'to' terminals."
                )
            for bus_name, terminals in (
                (from_bus_name, from_map),
                (to_bus_name, to_map),
            ):
                grounded = self._grounded(bus_name) & set(terminals)
                if grounded:
                    raise BMOPFParseError(
                        f"Line '{name}' connects grounded terminal(s) "
                        f"{sorted(grounded)} on bus '{bus_name}'; the pi model needs "
                        "a bus on every conductor."
                    )

            impedance, shunt, length = self._line_impedance(name, line, len(from_map))
            ampacity = line.get("i_max")
            if ampacity is None and line.get("linecode") is not None:
                ampacity = self._section("linecode")[line["linecode"]].get("i_max")

            model.create_unbalanced_line(
                impedance,
                shunt,
                from_bus_name,
                to_bus_name,
                length,
                list(zip(from_map, to_map, strict=True)),
                list(ampacity) if ampacity else None,
                name=name,
            )

    def _create_shunts(self, model: DxNetworkModel) -> None:
        """Build the shunt admittances, such as capacitor banks."""
        for name, shunt in self._section("shunt").items():
            terminals = [str(value) for value in shunt.get("terminal_map", [])]
            if not terminals:
                raise BMOPFParseError(f"Shunt '{name}' has no terminal map.")
            bus_name = shunt.get("bus")
            buses = [self._bus_obj(model, bus_name, terminal) for terminal in terminals]
            if any(bus is DxNetworkModel.GROUND_NODE for bus in buses):
                raise BMOPFParseError(f"Shunt '{name}' connects a grounded terminal.")
            size = len(terminals)
            y_matrix = np.zeros((size, size), dtype=complex)
            for row in range(size):
                for col in range(size):
                    g = float(shunt.get(f"G_{row + 1}_{col + 1}", 0.0))
                    b = float(shunt.get(f"B_{row + 1}_{col + 1}", 0.0))
                    y_matrix[row, col] = complex(g, b)
            model.create_shunt(buses, y_matrix, name=name)

    def _create_switches(self, model: DxNetworkModel) -> None:
        """Build the switches, each with the impedance it declares."""
        for name, switch in self._section("switch").items():
            from_bus_name = switch["bus_from"]
            to_bus_name = switch["bus_to"]
            from_map = [str(value) for value in switch.get("terminal_map_from", [])]
            to_map = [str(value) for value in switch.get("terminal_map_to", [])]
            if len(from_map) != len(to_map) or not from_map:
                raise BMOPFParseError(
                    f"Switch '{name}' has incompatible terminal maps."
                )
            status = (
                SwitchStatus.OPEN
                if bool(switch.get("open_switch", False))
                else SwitchStatus.CLOSED
            )
            if "r_series" not in switch or "x_series" not in switch:
                raise BMOPFParseError(
                    f"Switch '{name}' needs 'r_series' and 'x_series'; a closed "
                    "switch is an ideal short without them."
                )
            resistance_ohm = switch["r_series"]
            reactance_ohm = switch["x_series"]
            for index, (from_terminal, to_terminal) in enumerate(
                zip(from_map, to_map, strict=True)
            ):
                from_bus = self._bus_obj(model, from_bus_name, from_terminal)
                to_bus = self._bus_obj(model, to_bus_name, to_terminal)
                if (
                    from_bus is DxNetworkModel.GROUND_NODE
                    or to_bus is DxNetworkModel.GROUND_NODE
                ):
                    raise BMOPFParseError(
                        f"Switch '{name}' connects a grounded terminal."
                    )
                model.create_switch(
                    from_bus,
                    to_bus,
                    status,
                    from_terminal,
                    name=f"{name}_{index + 1}",
                    resistance_ohm=resistance_ohm,
                    reactance_ohm=reactance_ohm,
                )

    def _create_transformers(self, model: DxNetworkModel) -> None:
        """Build the transformers, one object per phase."""
        from src.components.transformer import Transformer

        for name, xfmr in self._section("transformer").get("single_phase", {}).items():
            from_map = [str(t) for t in xfmr["terminal_map_from"]]
            to_map = [str(t) for t in xfmr["terminal_map_to"]]
            for side, terminals in (("from", from_map), ("to", to_map)):
                if len(terminals) != 2:
                    raise BMOPFParseError(
                        f"Transformer '{name}' needs exactly two '{side}' terminals, "
                        f"got {terminals}."
                    )

            v_nom_from = float(xfmr["v_nom_from"])
            v_nom_to = float(xfmr["v_nom_to"])
            if v_nom_to <= 0.0:
                raise BMOPFParseError(f"Transformer '{name}' has a zero 'to' voltage.")
            turns_ratio = v_nom_from / v_nom_to

            # The internal model carries one series impedance referred to the
            # secondary, so the primary-side half is moved across the ratio.
            r_ohm = (
                float(xfmr.get("r_series_to", 0.0))
                + float(xfmr.get("r_series_from", 0.0)) / turns_ratio**2
            )
            x_ohm = (
                float(xfmr.get("x_series_to", 0.0))
                + float(xfmr.get("x_series_from", 0.0)) / turns_ratio**2
            )

            transformer = Transformer(
                name,
                self._bus_obj(model, xfmr["bus_from"], from_map[0]),
                self._bus_obj(model, xfmr["bus_from"], from_map[1]),
                self._bus_obj(model, xfmr["bus_to"], to_map[0]),
                self._bus_obj(model, xfmr["bus_to"], to_map[1]),
                r_ohm,
                x_ohm,
                True,
                turns_ratio,
                0.0,
                0.0,
                0.0,
                float(xfmr.get("s_rating", 0.0)),
                pri_conn="Y",
                sec_conn="Y",
            )
            model.__dict__["3p_transformers"].append(transformer)

        # BMOPF stores a three-phase delta-wye transformer as one record,
        # while DxNetworkModel's Transformer represents one branch.  Expand
        # the record into three delta-to-grounded-wye branches.  The partner
        # convention (1-3, 2-1, 3-2) matches the DSS loader's ANSI/lag
        # convention and preserves the winding connection in the equations.
        for name, xfmr in self._section("transformer").get("delta_wye", {}).items():
            from_map = [str(t) for t in xfmr.get("terminal_map_from", [])]
            to_map = [str(t) for t in xfmr.get("terminal_map_to", [])]
            if len(from_map) != 3 or len(to_map) < 3:
                raise BMOPFParseError(
                    f"Delta-wye transformer '{name}' needs three primary and "
                    f"at least three secondary terminals; got {from_map} and {to_map}."
                )
            v_nom_from = float(xfmr.get("v_nom_from", 0.0))
            v_nom_to = float(xfmr.get("v_nom_to", 0.0))
            if v_nom_from <= 0.0 or v_nom_to <= 0.0:
                raise BMOPFParseError(
                    f"Delta-wye transformer '{name}' has invalid winding voltages."
                )
            # Transformer.tr is expressed against phase-to-neutral bus bases,
            # while this BMOPF record publishes delta and wye line-to-line
            # nameplate voltages.  A delta branch is line-to-line, so the
            # phase-domain ratio needs the delta-to-wye sqrt(3) factor.
            turns_ratio = SQRT3 * v_nom_from / v_nom_to
            r_ohm = float(xfmr.get("r_series", 0.0))
            x_ohm = float(xfmr.get("x_series", 0.0))
            if r_ohm < 0.0 or x_ohm < 0.0:
                raise BMOPFParseError(
                    f"Delta-wye transformer '{name}' has negative series impedance."
                )
            rating = float(xfmr.get("s_rating", 0.0))
            for index, (from_terminal, to_terminal) in enumerate(
                zip(from_map, to_map[:3], strict=True)
            ):
                partner_terminal = from_map[(index - 1) % 3]
                transformer = Transformer(
                    f"{name}_{from_terminal}",
                    self._bus_obj(model, xfmr["bus_from"], from_terminal),
                    self._bus_obj(model, xfmr["bus_from"], partner_terminal),
                    self._bus_obj(model, xfmr["bus_to"], to_terminal),
                    DxNetworkModel.GROUND_NODE,
                    r_ohm,
                    x_ohm,
                    True,
                    turns_ratio,
                    0.0,
                    0.0,
                    0.0,
                    rating,
                    pri_conn="D",
                    sec_conn="Y",
                )
                model.__dict__["3p_transformers"].append(transformer)

    def _load_nominal_v(self, name, configuration, v_nom, bus) -> float:
        """Return the per-terminal nominal voltage for one load phase.

        The schema states ``v_nom`` line-to-line for a WYE load, which a
        phase-domain model needs per phase, and per phase already for the
        other configurations.  A load rated for a different voltage than the
        bus it sits on is ordinary, so the value is taken as stated rather
        than reconciled against the bus base.
        """
        if v_nom is None:
            return float(bus.v_base_v)
        v_nom = float(v_nom)
        return v_nom / SQRT3 if configuration == "WYE" else v_nom

    def _create_loads(self, model: DxNetworkModel) -> None:
        """Build the loads, one object per phase, with their voltage model."""
        for name, load in self._section("load").items():
            bus_name = load["bus"]
            configuration = str(load.get("configuration", "WYE")).upper()
            model_name = str(load.get("model", "CONSTANT_POWER")).upper()
            if model_name not in LOAD_MODEL_CODES:
                raise BMOPFParseError(
                    f"Load '{name}' uses model '{model_name}', which this parser "
                    f"does not build (supported: {sorted(LOAD_MODEL_CODES)})."
                )
            load_model = LOAD_MODEL_CODES[model_name]
            cvr_watts = float(load.get("cvr_watts", 1.0))
            cvr_vars = float(load.get("cvr_vars", 2.0))
            # Outside these bounds OpenDSS holds the load on a constant
            # impedance, so the limits belong to the load definition.
            v_min_pu = load.get("v_min_pu")
            v_max_pu = load.get("v_max_pu")

            terminals = [str(t) for t in load["terminal_map"]]
            grounded = self._grounded(bus_name)
            p_nom = load.get("p_nom", [])
            q_nom = load.get("q_nom", [])
            v_nom = load.get("v_nom", [])

            phase_terminals = [t for t in terminals if t not in grounded]
            if configuration == "SINGLE_PHASE":
                phase_terminals = phase_terminals[:1]
            if len(phase_terminals) > len(p_nom):
                raise BMOPFParseError(
                    f"Load '{name}' has {len(phase_terminals)} phase terminals but "
                    f"{len(p_nom)} power entries."
                )

            for position, terminal in enumerate(phase_terminals):
                from_bus = self._bus_obj(model, bus_name, terminal)
                if configuration == "DELTA":
                    partner = phase_terminals[(position + 1) % len(phase_terminals)]
                    to_bus = self._bus_obj(model, bus_name, partner)
                else:
                    # The return terminal is the last mapped one; grounded
                    # returns collapse onto GROUND_NODE.
                    to_bus = self._bus_obj(model, bus_name, terminals[-1])

                model.create_load(
                    name,
                    from_bus,
                    to_bus,
                    float(p_nom[position]),
                    float(q_nom[position]) if position < len(q_nom) else 0.0,
                    str(position),
                    terminal,
                    None,
                    nominal_v=self._load_nominal_v(
                        name,
                        configuration,
                        v_nom[position] if position < len(v_nom) else None,
                        from_bus,
                    ),
                    load_model=load_model,
                    cvr_watts=cvr_watts,
                    cvr_vars=cvr_vars,
                    vmin_pu=v_min_pu,
                    vmax_pu=v_max_pu,
                    # Outside the stated band OpenDSS holds the load on a
                    # constant impedance; honour that only when the document
                    # states the band.
                    use_opendss_voltage_limits=(
                        v_min_pu is not None or v_max_pu is not None
                    ),
                )

    def _inverter_records(self) -> dict:
        """Return each inverter connection paired with its type and PV array."""
        connections = self.doc.get("inverter_connections") or {}
        definition = self.doc.get("inverter") or {}
        if connections:
            if not isinstance(definition, dict) or not definition:
                raise BMOPFParseError("inverter_connections requires inverter")
            type_id = str(definition.get("id", ""))
            try:
                shared_s_va = float(definition["s_va"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BMOPFParseError("inverter requires numeric s_va") from exc
            controls = definition.get("controls") or {}
            pv = self.doc.get("pv") or {}
            pv_id = str(pv.get("id", ""))
            if not type_id:
                raise BMOPFParseError("inverter_connections requires inverter.id")
            if not pv_id:
                raise BMOPFParseError("inverter_connections requires pv.id")
            weights = definition.get("weights") or {}
            records = {}
            for name, connection in connections.items():
                if not isinstance(connection, dict):
                    raise BMOPFParseError(
                        f"inverter_connections[{name!r}] must be an object"
                    )
                if connection.get("id") != type_id:
                    raise BMOPFParseError(
                        f"inverter connection {name!r} references id "
                        f"{connection.get('id')!r}; expected {type_id!r}"
                    )
                if connection.get("dc_src_id") != pv_id:
                    raise BMOPFParseError(
                        f"inverter connection {name!r} references dc_src_id "
                        f"{connection.get('dc_src_id')!r}; expected {pv_id!r}"
                    )
                if "s_va" in connection:
                    raise BMOPFParseError(
                        f"inverter connection {name!r} duplicates inverter.s_va; "
                        "remove connection.s_va"
                    )
                record = copy.deepcopy(connection)
                record["type"] = type_id
                record["active_power"] = copy.deepcopy(
                    controls.get("active_power") or {}
                )
                record["reactive_power"] = copy.deepcopy(
                    controls.get("reactive_power") or {}
                )
                record["weights"] = copy.deepcopy(weights)
                record["s_va"] = shared_s_va
                records[str(name)] = record
            return records
        return {}

    def _inverter_types(self) -> dict:
        """Return the inverter type definitions the connections refer to."""
        definition = self.doc.get("inverter") or {}
        if isinstance(definition, dict) and definition:
            type_id = str(definition.get("id", "inverter_1"))
            raw_type = definition.get("electrical") or {}
            raw_types = {type_id: raw_type}
            controls = definition.get("controls") or {}
            active_mode = str(
                (controls.get("active_power") or {}).get("mode", "")
            ).upper()
        else:
            raw_types = {}
            active_mode = ""
        if not isinstance(raw_types, dict) or not raw_types:
            raise BMOPFParseError(
                "inverter cases require inverter with complete "
                "JSON-defined electrical data"
            )
        try:
            return {
                str(name): validate_inverter_type(spec, active_power_mode=active_mode)
                for name, spec in raw_types.items()
            }
        except InverterSpecError as exc:
            raise BMOPFParseError(str(exc)) from exc

    def _create_generators(self, model: DxNetworkModel) -> None:
        """Build the inverters this document connects to the network."""
        records = self._inverter_records()
        types = self._inverter_types() if records else {}
        pv = self.doc.get("pv") or {}
        for name, generator in self._section("generator").items():
            bus_name = generator["bus"]
            grounded = self._grounded(bus_name)
            terminals = [
                str(t) for t in generator["terminal_map"] if str(t) not in grounded
            ]
            if not terminals:
                raise BMOPFParseError(f"Generator '{name}' has no phase terminal.")

            record = records.get(name)
            if record is None:
                # A BMOPF generator publishes an envelope, not a set point, so
                # there is nothing to inject in a power flow.
                self._warn(
                    f"Generator '{name}' has no inverter record in inverter_connections and no "
                    "dispatch; created at zero injection."
                )
                for terminal in terminals:
                    model.create_generator(
                        f"{name}_{terminal}" if len(terminals) > 1 else name,
                        self._bus_obj(model, bus_name, terminal),
                        0.0,
                        0.0,
                        terminal,
                        gen_type="PQ",
                    )
                continue

            if len(terminals) > 1:
                raise BMOPFParseError(
                    f"Inverter '{name}' maps {len(terminals)} phase terminals; the "
                    "inverter model attaches to exactly one."
                )
            terminal = terminals[0]
            expected = f"{bus_name}.{terminal}"
            if str(record.get("terminal", expected)) != expected:
                self._warn(
                    f"Inverter '{name}' records terminal "
                    f"{record.get('terminal')!r} but its generator sits on "
                    f"{expected!r}; the generator wins."
                )

            try:
                instance = validate_inverter_instance(record, types)
                type_spec = types[str(instance["type"])]
                dispatch = dispatch_from_instance(instance, pv)
            except (InverterSpecError, PVSpecError) as exc:
                raise BMOPFParseError(f"inverter {name}: {exc}") from exc

            model.create_inverter(
                name,
                self._bus_obj(model, bus_name, terminal),
                float(dispatch["p_w"]),
                float(dispatch["q_var"]),
                terminal,
                inverter_type=type_spec,
                rating_va=float(instance["s_va"]),
                p_mode=str(dispatch["p_mode"]),
                pv_sdm=dispatch["pv_sdm"],
                control_mode=str(dispatch["control_mode"]),
                power_factor=dispatch["power_factor"],
                volt_var_knees=dispatch["volt_var_knees"],
                volt_var_q_min_pu=dispatch["volt_var_q_min_pu"],
                volt_var_q_max_pu=dispatch["volt_var_q_max_pu"],
                volt_var_v_ref_pu=dispatch["volt_var_v_ref_pu"],
                volt_var_smoothing_epsilon=dispatch["volt_var_smoothing_epsilon"],
                optimization_weight=float(dispatch["weight"]),
            )

        # Standalone scenarios keep the BMOPF generator section free of
        # duplicated inverter envelopes.  Their compact connection map is
        # resolved here into the same internal inverter objects.
        generator_names = set(self._section("generator"))
        for name, record in records.items():
            if name in generator_names:
                continue
            terminal_ref = str(record.get("terminal", ""))
            if "." not in terminal_ref:
                raise BMOPFParseError(
                    f"inverter {name}: connection requires terminal BUS.PHASE"
                )
            bus_name, terminal = terminal_ref.rsplit(".", 1)
            bus_name = self._canonical_bus_name(bus_name)
            if terminal in self._grounded(bus_name):
                raise BMOPFParseError(f"inverter {name}: terminal cannot be grounded")
            try:
                instance = validate_inverter_instance(record, types)
                type_spec = types[str(instance["type"])]
                dispatch = dispatch_from_instance(instance, pv)
            except (InverterSpecError, PVSpecError) as exc:
                raise BMOPFParseError(f"inverter {name}: {exc}") from exc
            model.create_inverter(
                name,
                self._bus_obj(model, bus_name, terminal),
                float(dispatch["p_w"]),
                float(dispatch["q_var"]),
                terminal,
                inverter_type=type_spec,
                rating_va=float(instance["s_va"]),
                p_mode=str(dispatch["p_mode"]),
                pv_sdm=dispatch["pv_sdm"],
                control_mode=str(dispatch["control_mode"]),
                power_factor=dispatch["power_factor"],
                volt_var_knees=dispatch["volt_var_knees"],
                volt_var_q_min_pu=dispatch["volt_var_q_min_pu"],
                volt_var_q_max_pu=dispatch["volt_var_q_max_pu"],
                volt_var_v_ref_pu=dispatch["volt_var_v_ref_pu"],
                volt_var_smoothing_epsilon=dispatch["volt_var_smoothing_epsilon"],
                optimization_weight=float(dispatch["weight"]),
            )


def load_bmopf_network(source) -> DxNetworkModel:
    """Parse a BMOPF JSON file (or already-loaded document) into components."""
    return BMOPFParser(source).parse()


# ---------------------------------------------------------------------------
# Locating the published scenario documents
# ---------------------------------------------------------------------------


def check_scenario_is_current(scenario_path: Path, document: dict) -> None:
    """Fail unless the network a scenario carries is the one it claims.

    A scenario embeds its own copy of the feeder network and records that
    network's fingerprint. Two things can then go wrong silently: the copy can
    be edited, and the feeder can be converted again without the scenarios
    being regenerated. The fingerprint is taken of the same sections either
    way, so one check catches both.
    """
    from converters.generate_curtailment_scenarios import network_fingerprint

    stamped = (document.get("meta") or {}).get("network_sha256")
    if not stamped:
        raise SystemExit(
            f"{scenario_path} carries no meta.network_sha256, so the network "
            "it embeds cannot be checked. Rerun 'python converters/"
            "generate_curtailment_scenarios.py'."
        )
    if network_fingerprint(document) != stamped:
        raise SystemExit(
            f"{scenario_path} does not carry the network it claims: its "
            "embedded network does not match meta.network_sha256."
        )
    network = scenario_path.parent.parent / "network" / "bmopf.json"
    if network.is_file() and network_fingerprint(network) != stamped:
        raise SystemExit(
            f"{scenario_path} was built from a different version of "
            f"{network}. Rerun 'python converters/generate_curtailment_"
            "scenarios.py' so the scenarios match the converted feeders."
        )


class StudyInputs:
    """The published scenario documents under one study root."""

    def __init__(
        self, root, per_case: int = 24, sweep: tuple[int, int, int] = (2, 4, 3)
    ):
        self.root = Path(root).expanduser().resolve()
        # How many scenarios a full sweep gives one case, and the size of each
        # axis, so a selection along any axis has a known expected count.
        self.per_case = int(per_case)
        self.p_modes, self.q_modes, self.norms = sweep

    def ensure(self, cases: list[str]) -> None:
        """Build whatever part of the study inputs is missing.

        The scenarios are generated artifacts, so a clone has none. Building
        one needs only the standard library and the feeder's converted
        network, which is tracked. Producing a network needs OpenDSS and
        powerio, so that step runs only when a network is genuinely absent,
        and an environment without those tools can still solve.
        """
        needs_scenarios = [
            case
            for case in cases
            if len(list((self.root / case / "scenarios").glob("*.json")))
            < self.per_case
        ]
        if not needs_scenarios:
            return
        needs_network = [
            case
            for case in needs_scenarios
            if not (self.root / case / "network" / "bmopf.json").is_file()
        ]

        # Imported here so reading a document never pulls in the converters.
        from converters.generate_curtailment_scenarios import generate_all

        if needs_network:
            print(
                f"Converting {', '.join(needs_network)} from OpenDSS; this "
                "needs powerio and OpenDSS...",
                flush=True,
            )
            from converters.dss_to_bmopf import main as convert

            if convert(["--root", str(self.root)]) != 0:
                raise SystemExit("OpenDSS conversion failed; cannot build study inputs")

        print(
            f"Building scenarios for {', '.join(needs_scenarios)} (a second)...",
            flush=True,
        )
        generate_all(self.root, needs_scenarios)

    def scenario_files(
        self,
        cases: list[str],
        p_mode: str | None = None,
        q_mode: str | None = None,
        norm: str | None = None,
    ) -> list[Path]:
        """Return the scenario documents for a selection, in a stable order.

        A selection has a known size, so a count that does not match it means
        the study root is incomplete and is reported rather than solved.
        """
        paths: list[Path] = []
        for case in cases:
            paths.extend(sorted((self.root / case / "scenarios").glob("*.json")))
        if p_mode:
            paths = [path for path in paths if path.name.startswith(p_mode + "_")]
        if q_mode:
            paths = [path for path in paths if f"_{q_mode}_" in path.name]
        if norm:
            paths = [path for path in paths if path.stem.endswith("_" + norm)]
        expected = (
            len(cases)
            * (1 if p_mode else self.p_modes)
            * (1 if q_mode else self.q_modes)
            * (1 if norm else self.norms)
        )
        if len(paths) != expected:
            raise ScenarioSelectionError(
                f"Expected {expected} scenario files, found {len(paths)} "
                f"under {self.root}"
            )
        return paths
