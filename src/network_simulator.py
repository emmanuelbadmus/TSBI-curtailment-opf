"""Pyomo-based power-flow and curtailment models."""

import contextlib
import copy
import io
import json
import logging
import math
import multiprocessing as mp
import re
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pyomo.environ as pe
from pyomo.common.errors import PyomoException
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    Objective,
    SolverFactory,
    Var,
    minimize,
    value,
)

from src.provenance import RunProvenance

_NUMERIC_EPS = 1e-12
_VOLTAGE_BASE_EPS = 1e-9

# This module lives in src/, so the repository root is one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]

# A constraint is judged against the precision its own terms can carry: with
# terms of order 1e11, cancelling them to zero leaves roughly 1e11 * 2.2e-16.
_RELATIVE_CONSTRAINT_TOLERANCE = 1e-14

# ---------------------------------------------------------------------------
# Per-unit conversion helpers
# ---------------------------------------------------------------------------
# Kept beside the solver because this repository has one active BMOPF
# electrical runtime.  The phase-domain base is a model convention; study
# inputs such as inverter physics, controls, objectives, and solver settings
# remain in each scenario JSON.
_MIN_BASE_V = 1e-6
_S_BASE_3PH_VA = 1_000_000.0
_LOGGER = logging.getLogger(__name__)


def s_base_3ph_va() -> float:
    """Return the three-phase power base of the per-unit system."""
    return _S_BASE_3PH_VA


def s_base_1ph_va() -> float:
    """Return the per-phase power base of the per-unit system."""
    return s_base_3ph_va() / 3.0


def resolve_voltage_base(v_base_v: float, fallback_v: float = 1.0) -> float:
    """Return a usable voltage base, falling back when one is missing."""
    value = abs(float(v_base_v))
    if value > _MIN_BASE_V:
        return value
    fallback = abs(float(fallback_v))
    if fallback > _MIN_BASE_V:
        return fallback
    return 1.0


# Load voltage-band choices settle in two or three passes; the cap only stops
# a pathological oscillation between two bands.
_MAX_LOAD_BAND_PASSES = 5


@dataclass(frozen=True)
class ElectricalBase:
    """The power and voltage base a quantity is per-unitised against."""

    s_base_va: float
    v_base_v: float

    @property
    def i_base_a(self) -> float:
        """Return the current base in amperes."""
        return self.s_base_va / self.v_base_v

    @property
    def z_base_ohm(self) -> float:
        """Return the impedance base in ohms."""
        return (self.v_base_v**2) / self.s_base_va

    @property
    def y_base_s(self) -> float:
        """Return the admittance base in siemens."""
        return 1.0 / self.z_base_ohm


def base_for_voltage(v_base_v: float, s_base_va: float | None = None) -> ElectricalBase:
    """Return the per-unit base belonging to one voltage level."""
    s_value = float(s_base_1ph_va() if s_base_va is None else s_base_va)
    return ElectricalBase(s_value, resolve_voltage_base(v_base_v))


def voltage_to_pu(v_raw: float, v_base_v: float) -> float:
    """Convert a voltage in volts to per unit."""
    return float(v_raw) / resolve_voltage_base(v_base_v)


def voltage_from_pu(v_pu: float, v_base_v: float) -> float:
    """Convert a per-unit voltage back to volts."""
    return float(v_pu) * resolve_voltage_base(v_base_v)


def current_to_pu(
    i_raw: float, v_base_v: float, s_base_va: float | None = None
) -> float:
    """Convert a current in amperes to per unit."""
    base = base_for_voltage(v_base_v, s_base_va=s_base_va)
    return float(i_raw) / base.i_base_a


def power_to_pu(s_raw_va: float, s_base_va: float | None = None) -> float:
    """Convert a power in volt-amperes to per unit."""
    base = float(s_base_1ph_va() if s_base_va is None else s_base_va)
    return float(s_raw_va) / base


def impedance_to_pu(
    z_raw_ohm: complex, v_base_v: float, s_base_va: float | None = None
) -> complex:
    """Convert an impedance in ohms to per unit."""
    return (
        complex(z_raw_ohm) / base_for_voltage(v_base_v, s_base_va=s_base_va).z_base_ohm
    )


def admittance_to_pu(
    y_raw_s: complex, v_base_v: float, s_base_va: float | None = None
) -> complex:
    """Convert an admittance in siemens to per unit."""
    return complex(y_raw_s) / base_for_voltage(v_base_v, s_base_va=s_base_va).y_base_s


def complex_current_to_pu(
    i_raw: complex, v_base_v: float, s_base_va: float | None = None
) -> complex:
    """Convert a complex current in amperes to per unit."""
    return complex(i_raw) / base_for_voltage(v_base_v, s_base_va=s_base_va).i_base_a


def voltage_scale(from_v_base: float, to_v_base: float) -> float:
    """Return the factor carrying a per-unit voltage between two bases."""
    return resolve_voltage_base(from_v_base) / resolve_voltage_base(to_v_base)


def current_scale(from_v_base: float, to_v_base: float) -> float:
    """Return the factor carrying a per-unit current between two bases."""
    return resolve_voltage_base(to_v_base) / resolve_voltage_base(from_v_base)


# Internal PV-to-grid TSBI states (SI units), in the order the simulator
# declares their Pyomo variables.  The curtailment engine stores/restores
# these as the inverter's ``_warm_state`` for continuation solves.
class _EchoingBuffer(io.StringIO):
    """Capture solver output and echo it to the real stdout.

    Both solve paths redirect stdout so the log can be parsed afterwards for
    iteration counts and failure messages. With ``tee`` the caller also wants
    to watch it, so write through to the process's own stdout rather than
    choosing between capturing and showing.
    """

    def write(self, text: str) -> int:
        """Write to the terminal and to the captured log at once."""
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        return super().write(text)


def _capture_stream(tee: bool | None):
    """Return the buffer solver output is captured into."""
    return _EchoingBuffer() if tee else io.StringIO()


def _iteration_count(log_out: str, results=None) -> str:
    """Return the iteration count Ipopt reported, or '?' if it is absent.

    Ipopt prints it as "Number of Iterations....: N". Benchmarking needs it
    for the optimization, not only for the reference power flow, so both solve
    paths read it the same way.
    """
    matches = re.findall(r"[Ii]terations[.\s]*[:\s]*(\d+)", log_out or "")
    if matches:
        return str(int(matches[-1]))
    solver = getattr(results, "solver", None)
    return str(getattr(solver, "iterations", "?")) if solver is not None else "?"


INVERTER_STATE_NAMES = (
    "v_pv",
    "i_pv",
    "i_dc",
    "duty_cycle",
    "magnitude_modulation",
    "modulation_r",
    "modulation_i",
    "v_ac_r",
    "v_ac_i",
    "i_ac_r",
    "i_ac_i",
    "i_t2_r",
    "i_t2_i",
)


class NetworkSimulator:
    """The power flow of one network, as a Pyomo model.

    Builds a variable for every bus voltage and component current, states
    each component's device law and one KCL row per terminal, and solves."""

    def __init__(self, network=None):
        self.network = network
        self.model = None
        self.solved_model = None
        self.solved = False

    def create_and_initialize_model(self):
        """Build the variables of the model and seed them with a starting point."""
        try:
            if self.network is None:
                return "Error: No network found."

            net = self.network
            m = ConcreteModel()
            m.name = "power_systems_model"

            all_transformers = list(getattr(net, "3p_transformers", []))
            transformers_with_aux_vars = [
                transformer
                for transformer in all_transformers
                if getattr(transformer, "requires_aux_vars", True)
            ]

            n = {
                k: len(getattr(net, k, []) or [])
                for k in [
                    "buses",
                    "slack",
                    "switches",
                    "loads",
                    "generators",
                    "inverters",
                ]
            }
            n["3p_transformers"] = len(transformers_with_aux_vars)
            n["3p_transformers_total"] = len(all_transformers)

            for key, val in n.items():
                setattr(m, f"num_{key}", val)

            def V(name, count, init=0.0):
                setattr(m, name, Var(range(count), initialize=init))

            def V_dict(name, count, init_dict):
                setattr(
                    m,
                    name,
                    Var(range(count), initialize=lambda _, i: init_dict.get(i, 0.0)),
                )

            def V_dict_bounded(name, count, init_dict, bounds):
                setattr(
                    m,
                    name,
                    Var(
                        range(count),
                        initialize=lambda _, i: init_dict.get(i, 0.0),
                        bounds=bounds,
                    ),
                )

            # ------------------------------------------------------------------
            # Bus voltages
            # ------------------------------------------------------------------
            bus_vr_init, bus_vi_init = {}, {}
            for b in net.buses:
                if hasattr(b, "int_bus_id"):
                    bus_vr_init[b.int_bus_id] = getattr(b, "Vr_pu", math.cos(b.Va_init))
                    bus_vi_init[b.int_bus_id] = getattr(b, "Vi_pu", math.sin(b.Va_init))

            V_dict("ipopt_vr_list", n["buses"], bus_vr_init)
            V_dict("ipopt_vi_list", n["buses"], bus_vi_init)

            # ------------------------------------------------------------------
            # Slack currents
            # ------------------------------------------------------------------
            V("slack_vr_list", n["slack"], 0.0)
            V("slack_vi_list", n["slack"], 0.0)

            # ------------------------------------------------------------------
            # Transformer aux voltages & currents
            # ------------------------------------------------------------------
            xfmr_vr_init, xfmr_vi_init = {}, {}
            xfmr_i_init = {}
            for idx, t in enumerate(transformers_with_aux_vars):
                xfmr_vr_init[idx] = getattr(t.to_bus_pos, "Vr_pu", 1.0)
                xfmr_vi_init[idx] = getattr(t.to_bus_pos, "Vi_pu", 0.0)

                rating = getattr(t, "power_rating", 0.0)
                v_base = getattr(t.to_bus_pos, "v_base_v", 1.0)
                i_guess_raw = (rating / v_base * 0.1) if v_base > 0 else 0.0
                xfmr_i_init[idx] = current_to_pu(i_guess_raw, v_base)

            V_dict("xfmr_vr_aux_list", n["3p_transformers"], xfmr_vr_init)
            V_dict("xfmr_vi_aux_list", n["3p_transformers"], xfmr_vi_init)

            for pfx in ["ir_pri", "ii_pri", "ir_sec", "ii_sec"]:
                V_dict(f"xfmr_{pfx}_list", n["3p_transformers"], xfmr_i_init)

            # ------------------------------------------------------------------
            # Switch currents
            # ------------------------------------------------------------------
            V("switch_ir_list", n["switches"], 0.0)
            V("switch_ii_list", n["switches"], 0.0)

            # ------------------------------------------------------------------
            # Explicit Current Initialization (I ≈ S*/V*)
            # ------------------------------------------------------------------
            for coll_name, prefix, idx_attr, bus_attr in [
                ("generators", "gen", "gen_idx", "bus"),
                ("inverters", "inv", "inv_idx", "bus"),
                ("loads", "load", "load_idx", "from_bus"),
            ]:
                ir_init, ii_init = {}, {}
                coll = getattr(net, coll_name, []) or []
                for idx, comp in enumerate(coll):
                    setattr(comp, idx_attr, idx)
                    bus = getattr(comp, bus_attr)
                    p_value = float(getattr(comp, "raw_p_w", 0.0))
                    q_value = float(getattr(comp, "raw_q_var", 0.0))
                    if coll_name == "inverters":
                        # NetworkSimulatorInvOpt intentionally starts at a
                        # small feasible dispatch.  KCL must start from that
                        # same PV/current state, not from the full requested
                        # nameplate power.
                        p_value = float(getattr(comp, "p_init", p_value))
                        grid_v = (
                            complex(
                                getattr(bus, "Vr_pu", 1.0),
                                getattr(bus, "Vi_pu", 0.0),
                            )
                            * bus.v_base_v
                        )
                        v_t2 = grid_v / float(comp.interface_turns_ratio)
                        if abs(v_t2) <= _VOLTAGE_BASE_EPS:
                            v_t2 = complex(comp.ac_terminal_voltage_v, 0.0)
                        initial_q = getattr(comp, "initial_reactive_power", None)
                        if callable(initial_q):
                            q_value = float(initial_q(v_t2))
                    p_pu = power_to_pu(p_value, s_base_va=bus.s_base_va)
                    q_pu = power_to_pu(q_value, s_base_va=bus.s_base_va)
                    vr0, vi0 = getattr(bus, "Vr_pu", 1.0), getattr(bus, "Vi_pu", 0.0)
                    vmag2 = vr0**2 + vi0**2

                    if vmag2 > _NUMERIC_EPS:
                        ir_init[idx] = (p_pu * vr0 + q_pu * vi0) / vmag2
                        ii_init[idx] = (p_pu * vi0 - q_pu * vr0) / vmag2
                    else:
                        ir_init[idx], ii_init[idx] = 0.0, 0.0

                V_dict(f"{prefix}_ir_list", n[coll_name], ir_init)
                V_dict(f"{prefix}_ii_list", n[coll_name], ii_init)

            # ------------------------------------------------------------------
            # Internal PV-to-grid TSBI states (SI units)
            # ------------------------------------------------------------------
            inverter_initials = {name: {} for name in INVERTER_STATE_NAMES}
            for idx, inverter in enumerate(getattr(net, "inverters", []) or []):
                for name, initial_value in inverter.initial_state.items():
                    inverter_initials[name][idx] = initial_value

            V_dict_bounded(
                "inv_v_pv_list",
                n["inverters"],
                inverter_initials["v_pv"],
                lambda _model, i: (
                    0.0,
                    getattr(net.inverters[i], "pv_voltage_max_v", None),
                ),
            )
            V_dict_bounded(
                "inv_i_pv_list",
                n["inverters"],
                inverter_initials["i_pv"],
                lambda _model, i: (
                    0.0
                    if getattr(net.inverters[i], "pv_sdm", None) is not None
                    else None,
                    getattr(net.inverters[i], "pv_current_max_a", None),
                ),
            )
            V_dict("inv_i_dc_list", n["inverters"], inverter_initials["i_dc"])
            V_dict_bounded(
                "inv_duty_cycle_list",
                n["inverters"],
                inverter_initials["duty_cycle"],
                (
                    min(
                        (float(inv.duty_cycle_min) for inv in net.inverters),
                        default=0.001,
                    ),
                    max(
                        (float(inv.duty_cycle_max) for inv in net.inverters),
                        default=0.999,
                    ),
                ),
            )
            V_dict_bounded(
                "inv_modulation_list",
                n["inverters"],
                inverter_initials["magnitude_modulation"],
                (0.0, 1.0),
            )
            V_dict_bounded(
                "inv_modulation_r_list",
                n["inverters"],
                inverter_initials["modulation_r"],
                (-1.0, 1.0),
            )
            V_dict_bounded(
                "inv_modulation_i_list",
                n["inverters"],
                inverter_initials["modulation_i"],
                (-1.0, 1.0),
            )
            for state_name in (
                "v_ac_r",
                "v_ac_i",
                "i_ac_r",
                "i_ac_i",
                "i_t2_r",
                "i_t2_i",
            ):
                V_dict(
                    f"inv_{state_name}_list",
                    n["inverters"],
                    inverter_initials[state_name],
                )

            # ------------------------------------------------------------------
            # Shared containers
            # ------------------------------------------------------------------
            m.cons = ConstraintList()
            m.pending_limits = {}

            # Component-owned constraint containers for explicit-current devices
            m._dx_load_constraints = ConstraintList()
            m._dx_generator_constraints = ConstraintList()
            m._dx_inverter_constraints = ConstraintList()

            self.model = m
            return (
                f"Model created: {n['buses']} buses, {n['loads']} loads, "
                f"{n['3p_transformers_total']} xfmrs, "
                f"{n['generators']} generators, {n['inverters']} inverters."
            )

        # This public builder returns a message because the batch runner must
        # record a failed scenario instead of losing the whole worker process.
        except Exception as exc:  # noqa: BLE001
            return f"Error creating model: {exc}"

    def build_constraints(self):
        """State every component's device law and the KCL row of every terminal."""
        try:
            if self.model is None:
                return "Error: No model initialized."
            if self.network is None:
                return "Error: No network loaded."

            m = self.model
            net = self.network

            cnames = [
                "KCL_real",
                "KCL_imag",
                "vr_slack",
                "vi_slack",
                "xfmr_ir_aux",
                "xfmr_ii_aux",
                "xfmr_vr_pri",
                "xfmr_vi_pri",
                "xfmr_ir_sec",
                "xfmr_ii_sec",
                "xfmr_vr_kvl",
                "xfmr_vi_kvl",
                "vr_switch",
                "vi_switch",
            ]

            # Remove old top-level constraint containers
            for nm in [*cnames, "voltage"]:
                attr = f"{nm}_constraints"
                for sfx in [attr, f"{attr}_index"]:
                    if hasattr(m, sfx):
                        m.del_component(getattr(m, sfx))

            # Remove old component-owned constraint lists if present
            for nm in [
                "_dx_load_constraints",
                "_dx_generator_constraints",
                "_dx_inverter_constraints",
            ]:
                if hasattr(m, nm):
                    m.del_component(getattr(m, nm))

            # Recreate component-owned lists
            m._dx_load_constraints = ConstraintList()
            m._dx_generator_constraints = ConstraintList()
            m._dx_inverter_constraints = ConstraintList()

            # Reset component-side build flags so constraints can be recreated
            for ld in getattr(net, "loads", []) or []:
                if hasattr(ld, "_constraints_built"):
                    ld._constraints_built = False
                if hasattr(ld, "_model"):
                    ld._model = None
                if hasattr(ld, "_constraint_list"):
                    ld._constraint_list = None

            for gen in getattr(net, "generators", []) or []:
                if hasattr(gen, "_constraints_built"):
                    gen._constraints_built = False
                if hasattr(gen, "_model"):
                    gen._model = None
                if hasattr(gen, "_constraint_list"):
                    gen._constraint_list = None

            for inverter in getattr(net, "inverters", []) or []:
                if hasattr(inverter, "_constraints_built"):
                    inverter._constraints_built = False
                if hasattr(inverter, "_model"):
                    inverter._model = None
                if hasattr(inverter, "_constraint_list"):
                    inverter._constraint_list = None
                if hasattr(inverter, "_constraint_expressions"):
                    inverter._constraint_expressions = {}

            eq = {nm: {} for nm in cnames}
            for nm in cnames:
                setattr(m, f"{nm}_constraints", ConstraintList())

            # ------------------------------------------------------------------
            # Buses
            # ------------------------------------------------------------------
            for bus in net.buses:
                bus.create_ipopt_bus_vars(m)
                bus.initialize_voltages()
                bus.add_to_eqn_list(eq["KCL_real"], eq["KCL_imag"])

            for shunt in getattr(net, "matrix_shunts", []) or []:
                shunt.assign_ipopt_vars(m)
                shunt.add_to_eqn_list(eq["KCL_real"], eq["KCL_imag"])

            # ------------------------------------------------------------------
            # Lines
            # ------------------------------------------------------------------
            for ele in net.lines:
                for ln in ele.lines:
                    ln.create_ipopt_vars(m, net.buses)
                    ln.add_to_eqn_list(eq["KCL_real"], eq["KCL_imag"])

            # ------------------------------------------------------------------
            # Slack
            # ------------------------------------------------------------------
            for s in net.slack:
                s.assign_ipopt_vars(m)
                s.add_to_eqn_list(
                    eq["KCL_real"],
                    eq["KCL_imag"],
                    eq["vr_slack"],
                    eq["vi_slack"],
                )

            # ------------------------------------------------------------------
            # 3-phase transformers
            # ------------------------------------------------------------------
            for t in getattr(net, "3p_transformers", []):
                t.assign_xfmr_ipopt_vars(m)
                init = getattr(t, "initialize_ipopt_vars_from_voltage", None)
                if callable(init):
                    init()
                if getattr(t, "requires_aux_vars", True):
                    t.add_to_eqn_list(
                        eq["KCL_real"],
                        eq["KCL_imag"],
                        eq["xfmr_ir_aux"],
                        eq["xfmr_ii_aux"],
                        eq["xfmr_vr_pri"],
                        eq["xfmr_vi_pri"],
                        eq["xfmr_ir_sec"],
                        eq["xfmr_ii_sec"],
                        eq["xfmr_vr_kvl"],
                        eq["xfmr_vi_kvl"],
                    )
                else:
                    t.add_to_eqn_list(eq["KCL_real"], eq["KCL_imag"])

            # ------------------------------------------------------------------
            # Loads
            # ------------------------------------------------------------------
            for ld in net.loads:
                ld.assign_ipopt_vars(m)
                ld.add_to_eqn_list(eq["KCL_real"], eq["KCL_imag"])

            # ------------------------------------------------------------------
            # Generators
            # ------------------------------------------------------------------
            for gen in getattr(net, "generators", []):
                gen.assign_ipopt_vars(m)
                gen.add_to_eqn_list(eq["KCL_real"], eq["KCL_imag"])

            # ------------------------------------------------------------------
            # Inverters
            # ------------------------------------------------------------------
            for inverter in getattr(net, "inverters", []):
                inverter.assign_ipopt_vars(m)
                inverter.add_to_eqn_list(eq["KCL_real"], eq["KCL_imag"])

            # ------------------------------------------------------------------
            # Switches
            # ------------------------------------------------------------------
            for sw in net.switches:
                sw.assign_ipopt_vars(m)
                init = getattr(sw, "initialize_ipopt_vars_from_voltage", None)
                if callable(init):
                    init()
                sw.add_to_eqn_list(
                    eq["KCL_real"],
                    eq["KCL_imag"],
                    eq["vr_switch"],
                    eq["vi_switch"],
                )

            for s in net.slack:
                idx = getattr(s.bus, "int_bus_id", None)
                if idx is None:
                    continue
                if idx in eq["KCL_real"]:
                    try:
                        s.ipopt_ir_slack.value += float(value(eq["KCL_real"][idx]))
                    except (PyomoException, TypeError, ValueError) as exc:
                        _LOGGER.debug(
                            "Could not seed the real slack current", exc_info=exc
                        )
                if idx in eq["KCL_imag"]:
                    try:
                        s.ipopt_ii_slack.value += float(value(eq["KCL_imag"][idx]))
                    except (PyomoException, TypeError, ValueError) as exc:
                        _LOGGER.debug(
                            "Could not seed the imaginary slack current", exc_info=exc
                        )

            total = 0
            for nm in cnames:
                cl = getattr(m, f"{nm}_constraints")
                for eqn in eq[nm].values():
                    cl.add(expr=eqn == 0)
                    total += 1

            if hasattr(m, "_dx_load_constraints"):
                total += len(m._dx_load_constraints)
            if hasattr(m, "_dx_generator_constraints"):
                total += len(m._dx_generator_constraints)
            if hasattr(m, "_dx_inverter_constraints"):
                total += len(m._dx_inverter_constraints)

            if total == 0:
                return "Error: No constraints generated."

            return f"Constraints built: {total} total."

        # Keep the error-return contract used by the standalone study runner.
        except Exception as exc:  # noqa: BLE001
            return f"Error building constraints: {exc}"

    def build_objective(self):
        """State the objective; a plain power flow only has to be feasible."""
        if self.model is None:
            return "Error: Model not initialized."
        if hasattr(self.model, "objective"):
            self.model.del_component(self.model.objective)

        self.model.objective = Objective(expr=0.0, sense=minimize)
        return "Objective built."

    def solve(self, network=None, timeout=None, tee=None):
        """Standard solve execution with log capture."""
        network_changed = network is not None
        if network is not None:
            self.network = network
        if self.network is None:
            return (False, "Error: No network found.", "?", 0.0)

        solve_time, start_time = 0.0, time.time()
        f_out, f_err = _capture_stream(tee), io.StringIO()
        try:

            def _configure_solver(tol=None):
                settings = getattr(self.network, "solver_settings", {}) or {}
                required = {
                    "name",
                    "linear_solver",
                    "tol",
                    "constraint_violation_tolerance",
                    "max_iter",
                    "honor_original_bounds",
                    "mu_init",
                    "mu_strategy",
                }
                missing = sorted(required - set(settings))
                if missing:
                    raise ValueError(
                        "solver settings are missing from the JSON document: "
                        + ", ".join(missing)
                    )
                solver_name = str(settings["name"])
                opt = SolverFactory(solver_name)
                opt.options.update(
                    {
                        "tol": float(settings["tol"]) if tol is None else tol,
                        "constr_viol_tol": float(
                            settings["constraint_violation_tolerance"]
                        ),
                        "max_iter": int(settings["max_iter"]),
                        "honor_original_bounds": settings["honor_original_bounds"],
                        "mu_init": float(settings["mu_init"]),
                        "mu_strategy": settings["mu_strategy"],
                        **_linear_solver_options(settings),
                    }
                )
                if timeout:
                    opt.options["max_cpu_time"] = float(timeout)
                return opt

            def _solve_current_model(tol=None):
                nonlocal f_out, f_err, solve_time
                if not hasattr(self.model, "objective"):
                    msg = self.build_objective()
                    if isinstance(msg, str) and msg.startswith("Error"):
                        raise RuntimeError(msg)
                f_out, f_err = _capture_stream(tee), io.StringIO()
                results = None
                with (
                    contextlib.redirect_stdout(f_out),
                    contextlib.redirect_stderr(f_err),
                ):
                    results = _configure_solver(tol=tol).solve(
                        self.model,
                        # Always ask Pyomo for the solver log: it is the only
                        # place Ipopt reports its iteration count, and the ASL
                        # results object does not carry it. stdout is
                        # redirected into a buffer either way; the caller's
                        # tee only decides whether that buffer also echoes.
                        tee=True,
                        load_solutions=False,
                    )
                    solve_time = time.time() - start_time
                    if results and str(results.solver.termination_condition) in {
                        "optimal",
                        "locallyOptimal",
                        "feasible",
                    }:
                        self.model.solutions.load_from(results)
                tc = str(results.solver.termination_condition) if results else "None"
                log_out = f_out.getvalue() + f_err.getvalue()
                iter_count = _iteration_count(log_out, results)
                return results, tc, log_out, iter_count

            if self.model is None or network_changed:
                msg = self.create_and_initialize_model()
                if isinstance(msg, str) and msg.startswith("Error"):
                    return (False, msg, "?", 0.0)
                msg = self.build_constraints()
                if isinstance(msg, str) and msg.startswith("Error"):
                    return (False, msg, "?", 0.0)
            results, tc, log_out, iter_count = _solve_current_model()

            if tc not in {"optimal", "locallyOptimal", "feasible"}:
                # ---------------------------------------------------------
                # Retry the reference power flow from its initial point with
                # a small voltage regularisation objective. This retry only
                # applies to the no-inverter reference solve. The curtailment
                # subclass keeps its requested objective unchanged.
                # ---------------------------------------------------------
                try:
                    self.model = None
                    msg = self.create_and_initialize_model()
                    if isinstance(msg, str) and msg.startswith("Error"):
                        return (False, f"Solver failed ({tc})", "?", solve_time)
                    msg = self.build_constraints()
                    if isinstance(msg, str) and msg.startswith("Error"):
                        return (False, f"Solver failed ({tc})", "?", solve_time)

                    m2 = self.model
                    # Build regularisation objective: minimise Σ (vr - vr0)² + (vi - vi0)²
                    if hasattr(m2, "objective"):
                        m2.del_component(m2.objective)
                    reg_expr = 0.0
                    for b in self.network.buses:
                        idx = getattr(b, "int_bus_id", None)
                        if idx is None:
                            continue
                        vr0 = float(getattr(b, "Vr_pu", 0.0))
                        vi0 = float(getattr(b, "Vi_pu", 0.0))
                        reg_expr += (m2.ipopt_vr_list[idx] - vr0) ** 2
                        reg_expr += (m2.ipopt_vi_list[idx] - vi0) ** 2
                    m2.objective = Objective(expr=1e-6 * reg_expr, sense=minimize)

                    results2, tc2, log_out2, iter_count2 = _solve_current_model()
                    if tc2 in {"optimal", "locallyOptimal", "feasible"}:
                        tc, log_out, iter_count = tc2, log_out2, iter_count2
                # Solver/plugin failures during the optional retry must leave
                # the original termination condition available to the caller.
                except Exception as exc:
                    _LOGGER.debug("Reference-solve retry failed", exc_info=exc)

            if tc not in {"optimal", "locallyOptimal", "feasible"}:
                self.model = None
                # Report the count even on failure: whether a solve stalled
                # after a few iterations or ground through hundreds is the
                # first thing a benchmark reader wants to know.
                return (False, f"Solver failed ({tc})", iter_count, solve_time)

            if self.model is None:
                return (False, "Solver failed (model unavailable)", "?", solve_time)

            # Re-take the load voltage-band choices at the solution and repeat
            # while any of them moves.  A rebuild that fails to solve leaves
            # the last good solution in place rather than losing it.
            for _ in range(_MAX_LOAD_BAND_PASSES):
                if not self._refresh_cvr_branches():
                    break
                previous_model = self.model
                self.model = None
                rebuild = self.create_and_initialize_model()
                if isinstance(rebuild, str) and rebuild.startswith("Error"):
                    self.model = previous_model
                    break
                rebuild = self.build_constraints()
                if isinstance(rebuild, str) and rebuild.startswith("Error"):
                    self.model = previous_model
                    break
                results, tc_next, log_out, iter_next = _solve_current_model()
                if tc_next not in {"optimal", "locallyOptimal", "feasible"}:
                    self.model = previous_model
                    break
                tc, iter_count = tc_next, iter_next

            self.solved_model, self.solved = self.model, True
            # As above: the status, not the captured log.
            return (True, tc, iter_count, solve_time)

        # A solver backend can raise backend-specific exceptions. Convert them
        # into the stable failure tuple consumed by the study runner.
        except Exception as exc:  # noqa: BLE001
            return self._format_solver_error(exc, locals())

    def _refresh_cvr_branches(self) -> int:
        """Move every CVR load's branch onto the solution just computed.

        OpenDSS holds a load on a constant impedance outside its stated voltage
        band.  The band a load falls in is a discrete choice, so it is fixed
        before the solve rather than switched on a variable, which would make
        the problem non-smooth.  Fixing it at a flat start misplaces every load
        that ends up outside the band, so the choice is re-taken at the
        solution and the solve repeated until it stops moving.

        Returns the number of loads whose band changed.
        """
        changed = 0
        for load in getattr(self.network, "loads", []) or []:
            operating = getattr(load, "operating_vpu", None)
            if operating is None:
                continue
            value_now = operating()
            if value_now is None:
                continue
            previous = load.branch_vpu
            if load.cvr_branch_label(previous if previous is not None else 1.0) != (
                load.cvr_branch_label(value_now)
            ):
                changed += 1
            load.branch_vpu = value_now
        return changed

    def _format_solver_error(self, error, local_vars):
        """Build a standard failure tuple after an unexpected solver error."""
        log_out = ""
        f_out, f_err = local_vars.get("f_out"), local_vars.get("f_err")
        if f_out and f_err:
            try:
                log_out = "\nLog: " + f_out.getvalue() + f_err.getvalue()
            except (AttributeError, OSError, TypeError, ValueError):
                log_out = ""
        return (
            False,
            f"CRITICAL ERROR: {error}{log_out}",
            "?",
            local_vars.get("solve_time", 0.0),
        )


# ---------------------------------------------------------------------------
# Curtailment optimizer
# ---------------------------------------------------------------------------


def _curtailment_bounds(inv) -> tuple[float, float]:
    """Bounds for P_curtailed = P_requested - P_delivered."""
    raw = float(inv.raw_p_w)
    if raw <= 0.0:
        return 0.0, 0.0
    if getattr(inv, "p_mode", "CONSTANT_P") == "MPPT":
        pv_mpp = getattr(inv, "pv_mpp", None)
        minimum = 0.0
        if pv_mpp is not None:
            minimum = max(raw - float(pv_mpp["p_mpp_w"]), 0.0)
        return minimum, raw
    return 0.0, raw


def _initial_delivery(inv, fraction: float) -> float:
    """Small nonsingular export used consistently by TSBI and curtailment."""
    raw = float(inv.raw_p_w)
    if raw <= 0.0:
        return raw
    return raw * float(fraction)


def _initial_curtailment(inv, fraction: float) -> float:
    """Curtailment value matching the inverter's initial delivered power."""
    raw = float(inv.raw_p_w)
    if raw <= 0.0:
        return 0.0
    lower, upper = _curtailment_bounds(inv)
    return min(max(raw - _initial_delivery(inv, fraction), lower), upper)


class NetworkSimulatorInvOpt(NetworkSimulator):
    """Feasibility AC model with a minimal-curtailment objective."""

    def __init__(
        self,
        network,
        norm: str = "L1",
        voltage_band: tuple | None = None,
        voltage_reference: dict | None = None,
        initial_curtailments: list[float] | None = None,
        initial_delivery_fraction: float | None = None,
        scale_linf_variables: bool = False,
    ):
        super().__init__(network)
        self.norm = str(norm).upper()
        if self.norm not in ("L1", "L2", "LINF"):
            raise ValueError(f"norm must be 'L1', 'L2' or 'Linf', got {norm!r}")
        self.voltage_band = voltage_band
        # Per-bus voltage magnitudes the band is measured against, keyed by
        # int_bus_id. Supplying them explicitly keeps the band independent of
        # bus.Vr_pu, which callers also use to seed the initial point, so the
        # two cannot be changed by accident together.
        self.voltage_reference = dict(voltage_reference or {})
        self.initial_curtailments = initial_curtailments
        self.scale_linf_variables = bool(scale_linf_variables)
        solver_settings = getattr(network, "solver_settings", {}) or {}
        if (
            initial_delivery_fraction is None
            and "initial_delivery_fraction" not in solver_settings
        ):
            raise ValueError(
                "the standalone JSON must provide solver.initial_delivery_fraction"
            )
        self.initial_delivery_fraction = (
            float(solver_settings["initial_delivery_fraction"])
            if initial_delivery_fraction is None
            else float(initial_delivery_fraction)
        )
        bounds = [_curtailment_bounds(inv) for inv in (network.inverters or [])]
        self.curtailment_var_scales = [
            max(float(upper), 1.0) for _lower, upper in bounds
        ]
        self.curtailment_scale = max(
            (
                float(getattr(inv, "curtailment_weight", 1.0)) * max(float(upper), 0.0)
                for inv, (_lower, upper) in zip(
                    network.inverters or [], bounds, strict=True
                )
            ),
            default=1.0,
        )
        self.linf_incumbent = None
        if not 0.0 < self.initial_delivery_fraction <= 1.0:
            raise ValueError("initial_delivery_fraction must be in (0, 1]")

    def create_and_initialize_model(self):
        """Build the model, adding the curtailment variable of every inverter."""
        n_inverters = len(self.network.inverters or [])
        if (
            self.initial_curtailments is not None
            and len(self.initial_curtailments) != n_inverters
        ):
            raise ValueError(
                "initial_curtailments length must match the inverter fleet"
            )
        for i, inverter in enumerate(self.network.inverters or []):
            if self.initial_curtailments is not None:
                lower, upper = _curtailment_bounds(inverter)
                proposed = float(self.initial_curtailments[i])
                initial_curtailment = min(max(proposed, lower), upper)
                inverter.p_init = float(inverter.raw_p_w) - initial_curtailment
            else:
                inverter.p_init = _initial_delivery(
                    inverter, self.initial_delivery_fraction
                )
        result = super().create_and_initialize_model()
        if self.model is None:
            return result
        bounds = {
            i: _curtailment_bounds(inverter)
            for i, inverter in enumerate(self.network.inverters or [])
        }
        if self.initial_curtailments is not None:
            initials = {
                i: min(
                    max(float(self.initial_curtailments[i]), bounds[i][0]), bounds[i][1]
                )
                for i in range(n_inverters)
            }
        else:
            initials = {
                i: _initial_curtailment(inverter, self.initial_delivery_fraction)
                for i, inverter in enumerate(self.network.inverters or [])
            }
        if self.norm == "LINF" and self.scale_linf_variables:
            variable_scales = self.curtailment_var_scales
            self.model.u_curtail_list = Var(
                range(n_inverters),
                bounds=lambda _model, i: (
                    bounds.get(i, (0.0, 0.0))[0] / variable_scales[i],
                    bounds.get(i, (0.0, 0.0))[1] / variable_scales[i],
                ),
                initialize=lambda _model, i: initials.get(i, 0.0) / variable_scales[i],
            )
            self.model.p_curtail_list = pe.Expression(
                range(n_inverters),
                rule=lambda _model, i: (
                    variable_scales[i] * self.model.u_curtail_list[i]
                ),
            )
        else:
            self.model.p_curtail_list = Var(
                range(n_inverters),
                bounds=lambda _model, i: bounds.get(i, (0.0, 0.0)),
                initialize=lambda _model, i: initials.get(i, 0.0),
            )
        if self.norm == "LINF":
            max_request = (
                max(
                    upper
                    * float(getattr(inverter, "curtailment_weight", 1.0))
                    / self.curtailment_scale
                    for inverter, (_lower, upper) in zip(
                        self.network.inverters or [], bounds.values(), strict=True
                    )
                )
                if bounds
                else 0.0
            )
            max_initial = (
                max(
                    float(getattr(inverter, "curtailment_weight", 1.0))
                    * initials[i]
                    / self.curtailment_scale
                    for i, inverter in enumerate(self.network.inverters or [])
                )
                if initials
                else 0.0
            )
            self.model.max_curtail = Var(
                bounds=(0.0, max_request), initialize=max_initial
            )
        return result

    def build_objective(self):
        """Minimise the selected norm of the per-inverter curtailment."""
        if self.model is None:
            return "Error: Model not initialized."
        if hasattr(self.model, "objective"):
            self.model.del_component(self.model.objective)
        n_inverters = len(self.network.inverters or [])
        p_scale = max(
            getattr(
                self,
                "curtailment_scale",
                max(
                    (
                        max(float(inv.raw_p_w), 0.0)
                        for inv in (self.network.inverters or [])
                    ),
                    default=1.0,
                ),
            ),
            1.0,
        )
        weights = [
            float(getattr(inverter, "curtailment_weight", 1.0))
            for inverter in (self.network.inverters or [])
        ]
        if self.norm == "L1":
            norm_expr = (
                sum(
                    weights[i] * self.model.p_curtail_list[i]
                    for i in range(n_inverters)
                )
                / p_scale
            )
        elif self.norm == "L2":
            norm_expr = (
                sum(
                    weights[i] * self.model.p_curtail_list[i] ** 2
                    for i in range(n_inverters)
                )
                / p_scale**2
            )
        else:
            norm_expr = self.model.max_curtail
        self.model.objective = Objective(expr=norm_expr, sense=minimize)
        return f"Objective built (one-shot minimize {self.norm} curtailment norm)."

    def solve(self, network=None, timeout=None, tee=None):
        """Solve without changing the requested curtailment objective."""
        if network is not None:
            self.network = network
        if self.model is None:
            msg = self.create_and_initialize_model()
            if isinstance(msg, str) and msg.startswith("Error"):
                return (False, msg, "?", 0.0)
            msg = self.build_constraints()
            if isinstance(msg, str) and msg.startswith("Error"):
                return (False, msg, "?", 0.0)
        if (
            next(self.model.component_data_objects(pe.Objective, active=True), None)
            is None
        ):
            msg = self.build_objective()
            if isinstance(msg, str) and msg.startswith("Error"):
                return (False, msg, "?", 0.0)

        solve_time, start_time = 0.0, time.monotonic()

        def attempt(mu_strategy: str, tol: float | None = None) -> tuple[str, str]:
            nonlocal solve_time
            settings = getattr(self.network, "solver_settings", {}) or {}
            required = {
                "name",
                "linear_solver",
                "tol",
                "constraint_violation_tolerance",
                "acceptable_tol",
                "acceptable_iter",
                "acceptable_constr_viol_tol",
                "max_iter",
                "honor_original_bounds",
                "mu_init",
            }
            missing = sorted(required - set(settings))
            if missing:
                raise ValueError(
                    "solver settings are missing from the JSON document: "
                    + ", ".join(missing)
                )
            solver_name = str(settings["name"])
            opt = SolverFactory(solver_name)
            opt.options.update(
                {
                    "tol": float(settings["tol"]) if tol is None else float(tol),
                    "constr_viol_tol": float(
                        settings["constraint_violation_tolerance"]
                    ),
                    "acceptable_tol": float(settings["acceptable_tol"]),
                    "acceptable_iter": int(settings["acceptable_iter"]),
                    "acceptable_constr_viol_tol": float(
                        settings["acceptable_constr_viol_tol"]
                    ),
                    "max_iter": int(settings["max_iter"]),
                    "honor_original_bounds": settings["honor_original_bounds"],
                    "mu_init": float(settings["mu_init"]),
                    "mu_strategy": mu_strategy,
                    **_linear_solver_options(settings),
                }
            )
            if timeout:
                opt.options["max_cpu_time"] = float(timeout)
            f_out, f_err = _capture_stream(tee), io.StringIO()
            with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
                results = opt.solve(
                    self.model,
                    # See the note above: the log is always requested so the
                    # iteration count can be read from it.
                    tee=True,
                    load_solutions=False,
                )
                solve_time = time.monotonic() - start_time
                termination = str(results.solver.termination_condition)
                if termination in {"optimal", "locallyOptimal"}:
                    self.model.solutions.load_from(results)
            return termination, f_out.getvalue() + f_err.getvalue()

        settings = getattr(self.network, "solver_settings", {}) or {}
        first_strategy = str(settings.get("mu_strategy", ""))
        retry_strategy = str(settings.get("retry_mu_strategy", ""))
        termination, log_out = attempt(first_strategy)
        if termination not in {"optimal", "locallyOptimal"}:
            termination, log_out = attempt(retry_strategy)
        if termination not in {"optimal", "locallyOptimal"}:
            # Ipopt builds differ in where their restoration phase gives up.
            # These cases stall at a point that is already feasible to well
            # inside constr_viol_tol while dual infeasibility sits just above
            # `tol`, so retry the first strategy at the acceptable tolerance.
            # The constraint-violation tolerance is untouched, so the hard
            # voltage and ampacity limits are enforced exactly as before.
            termination, log_out = attempt(
                first_strategy, tol=float(settings["acceptable_tol"])
            )
        if termination not in {"optimal", "locallyOptimal"}:
            tail = "\n".join(line for line in log_out.splitlines() if line.strip())[
                -1500:
            ]
            return (
                False,
                f"Solver failed ({termination}, hard voltage/ampacity constraints)\n{tail}",
                _iteration_count(log_out),
                solve_time,
            )
        self.solved_model, self.solved = self.model, True
        # Report the termination status, not the log. The log is always
        # captured now so the iteration count can be read from it, but pasting
        # several kilobytes of it into every result row would bury the answer.
        return (True, termination, _iteration_count(log_out), solve_time)

    def curtailment_report(self) -> list[dict]:
        """Return per-inverter active-power accounting in watts."""
        if self.model is None:
            return []
        report = []
        for i, inverter in enumerate(self.network.inverters or []):
            requested = max(float(inverter.raw_p_w), 0.0)
            curtailed = float(pe.value(self.model.p_curtail_list[i]))
            physical_mppt = getattr(inverter, "pv_sdm", None) is not None
            bus = inverter.bus
            terminal_voltage_scale = bus.v_base_v / inverter.interface_turns_ratio
            terminal_vr = float(pe.value(inverter.ipopt_vr)) * terminal_voltage_scale
            terminal_vi = float(pe.value(inverter.ipopt_vi)) * terminal_voltage_scale
            ac_export = terminal_vr * float(
                pe.value(inverter.i_t2_r)
            ) + terminal_vi * float(pe.value(inverter.i_t2_i))
            pv_source_power = (
                float(pe.value(inverter.v_pv)) * float(pe.value(inverter.i_pv))
                if physical_mppt
                else None
            )
            converter_loss = (
                max(pv_source_power - ac_export, 0.0)
                if pv_source_power is not None
                else None
            )
            if physical_mppt:
                network_curtailment_source = max(
                    float(inverter.pv_mpp["p_mpp_w"]) - pv_source_power, 0.0
                )
                source_tolerance = (
                    5.0
                    * float(
                        (getattr(self.network, "solver_settings", {}) or {})[
                            "constraint_violation_tolerance"
                        ]
                    )
                    * float(inverter.rating_va)
                )
                if network_curtailment_source <= source_tolerance:
                    network_curtailment_source = 0.0
                efficiency = (
                    ac_export / pv_source_power
                    if pv_source_power is not None and pv_source_power > 1e-9
                    else 0.0
                )
                network_curtailment = network_curtailment_source
                availability_shortfall = max(
                    requested - float(inverter.pv_mpp["p_mpp_w"]), 0.0
                )
            else:
                efficiency = None
                network_curtailment_source = curtailed
                network_curtailment = curtailed
                availability_shortfall = 0.0
            available = max(requested - availability_shortfall, 0.0)
            report.append(
                {
                    "inverter": inverter.name,
                    "terminal": f"{inverter.bus.NodeName}.{inverter.phase}",
                    "requested_w": requested,
                    "available_w": available,
                    "curtailed_w": curtailed,
                    "availability_shortfall_w": availability_shortfall,
                    "network_curtailment_w": network_curtailment,
                    "objective_curtailment_w": curtailed,
                    "network_curtailment_source_w": network_curtailment_source,
                    "network_curtailment_basis": "pv_source_w"
                    if physical_mppt
                    else "ac_terminal_w",
                    "delivered_w": requested - curtailed,
                    "ac_export_w": ac_export,
                    "pv_source_power_w": pv_source_power,
                    "converter_loss_w": converter_loss,
                    "conversion_efficiency": efficiency,
                    "pv_mpp_w": None
                    if not physical_mppt
                    else float(inverter.pv_mpp["p_mpp_w"]),
                    "pv_mpp_voltage_v": None
                    if not physical_mppt
                    else float(inverter.pv_mpp["v_mpp_v"]),
                    "pv_mpp_current_a": None
                    if not physical_mppt
                    else float(inverter.pv_mpp["i_mpp_a"]),
                }
            )
        return report

    def build_constraints(self):
        """State the network's laws, plus the voltage and ampacity limits curtailment must respect."""
        model = self.model
        if model is not None:
            for i, inverter in enumerate(self.network.inverters or []):
                raw = float(inverter.raw_p_w)
                inverter.p_target = raw - model.p_curtail_list[i] if raw > 0.0 else raw
        result = super().build_constraints()
        if self.model is None:
            return result
        model = self.model
        n_inverters = len(self.network.inverters or [])
        if self.norm == "LINF":
            if hasattr(model, "curtail_linf_constraints"):
                model.del_component(model.curtail_linf_constraints)
            model.curtail_linf_constraints = ConstraintList()
            for i in range(n_inverters):
                model.curtail_linf_constraints.add(
                    model.max_curtail
                    >= float(
                        getattr(self.network.inverters[i], "curtailment_weight", 1.0)
                    )
                    * model.p_curtail_list[i]
                    / self.curtailment_scale
                )
        if self.voltage_band is not None:
            v_lo, v_hi = map(float, self.voltage_band)
            if hasattr(model, "voltage_band_constraints"):
                model.del_component(model.voltage_band_constraints)
            model.voltage_band_constraints = ConstraintList()
            for bus in self.network.buses:
                idx = getattr(bus, "int_bus_id", None)
                if idx is None:
                    continue
                if idx in self.voltage_reference:
                    ref_mag = float(self.voltage_reference[idx])
                else:
                    ref_vr = float(getattr(bus, "Vr_pu", 0.0) or 0.0)
                    ref_vi = float(getattr(bus, "Vi_pu", 0.0) or 0.0)
                    ref_mag = math.hypot(ref_vr, ref_vi)
                if ref_mag <= 1e-6:
                    continue
                v_sq = model.ipopt_vr_list[idx] ** 2 + model.ipopt_vi_list[idx] ** 2
                model.voltage_band_constraints.add(v_sq <= (ref_mag * v_hi) ** 2)
                model.voltage_band_constraints.add(v_sq >= (ref_mag * v_lo) ** 2)
        if hasattr(model, "line_ampacity_constraints"):
            model.del_component(model.line_ampacity_constraints)
        model.line_ampacity_constraints = ConstraintList()
        for element in getattr(self.network, "lines", []) or []:
            for line in getattr(element, "lines", []) or []:
                ampacity = getattr(line, "ampacity_a", None)
                if ampacity is None or ampacity <= 0.0:
                    continue
                ir, ii = line.find_Ir_Ii_from()
                limit_pu = ampacity / max(float(line.i_base_from), 1e-12)
                model.line_ampacity_constraints.add(ir**2 + ii**2 <= limit_pu**2)
        return result


# ---------------------------------------------------------------------------
# Solving one published scenario, and judging the solution
# ---------------------------------------------------------------------------


def _linear_solver_options(settings: dict) -> dict:
    """Return the Ipopt options that select and tune the linear solver.

    ``mumps_pivtol`` is a MUMPS option, so it is sent only when MUMPS is the
    linear solver in use; Ipopt rejects an option that its active linear
    solver does not define.
    """
    linear_solver = str(settings["linear_solver"]).lower()
    options = {"linear_solver": linear_solver}
    if linear_solver == "mumps":
        options["mumps_pivtol"] = float(settings["mumps_pivot_tolerance"])
    return options


class ScenarioSolver:
    """Solve published scenario documents with one set of solver options.

    The options are fixed once and reused for every scenario, so no case is
    solved on different terms than another.
    """

    # Ipopt's linear solvers. MUMPS is built into every Ipopt distribution and
    # is what the published results use. The HSL solvers (ma27 through ma97)
    # and the rest are present only in builds linked against them, so asking
    # for one is checked before a run rather than discovered 120 failures in.
    LINEAR_SOLVERS = (
        "mumps",
        "ma27",
        "ma57",
        "ma77",
        "ma86",
        "ma97",
        "pardiso",
        "pardisomkl",
        "spral",
        "wsmp",
    )
    DEFAULT_LINEAR_SOLVER = "mumps"
    # MA77 is an out-of-core solver: it holds its factorisation in scratch
    # files named the same way for every solve. Two solves sharing a directory
    # therefore overwrite each other and both fail, and an interrupted run
    # leaves the files behind. Each such solve gets a directory of its own.
    SCRATCH_FILE_LINEAR_SOLVERS = ("ma77",)

    def __init__(
        self,
        timeout=None,
        tee: bool = False,
        warm_start: bool = True,
        voltage_bands: dict | None = None,
        linear_solver: str | None = None,
    ):
        self.timeout = timeout
        self.tee = bool(tee)
        self.warm_start = bool(warm_start)
        # None leaves each scenario on the linear solver its JSON states.
        self.linear_solver = linear_solver
        # A feeder's voltage band comes from its study design, so every
        # scenario of that feeder is held to the same limits. Bands given
        # here take precedence; anything missing is read from disk once.
        self.voltage_bands = dict(voltage_bands or {})

    def voltage_band(self, case_name: str) -> tuple[float, float]:
        """Return the relative voltage band one feeder is held to."""
        band = self.voltage_bands.get(case_name)
        if band is None:
            # Imported here so solving does not require the generator unless
            # the caller left a band unstated.
            from converters.generate_curtailment_scenarios import FeederLibrary

            band = FeederLibrary().voltage_band(case_name)
            self.voltage_bands[case_name] = band
        return band

    def _effective_solver_settings(self, document: dict) -> dict:
        """Return the solver settings a scenario is actually solved with.

        The scenario document itself is left untouched, since the parser
        validates it as published. A requested linear solver is applied to
        this copy, which both the reference power flow and the curtailment
        optimization are given, so the two never differ.
        """
        settings = copy.deepcopy(document["solver"])
        if self.linear_solver is not None:
            settings["linear_solver"] = self.linear_solver
            if self.linear_solver != "mumps":
                # A MUMPS pivot tolerance means nothing to another solver, and
                # a published row should record only what was actually sent.
                settings.pop("mumps_pivot_tolerance", None)
        return settings

    def _constraint_term_scale(self, constraint) -> float:
        """Return the magnitude of the largest term the constraint body sums."""
        terms = getattr(constraint.body, "args", None)
        if not terms:
            return 0.0
        scale = 0.0
        for term in terms:
            try:
                value = abs(float(pe.value(term)))
            except (PyomoException, TypeError, ValueError):
                continue
            if math.isfinite(value):
                scale = max(scale, value)
        return scale

    def _largest_constraint_violation(
        self, model, absolute_tolerance: float
    ) -> tuple[float, str | None, float, str | None]:
        """Report the largest constraint violation and the worst one relative to
        what its own arithmetic can resolve.

        Returns the largest absolute violation with its name, then the largest
        ratio of a violation to its allowance with its name. A ratio above 1.0 is
        a real violation; the absolute figure is reported for continuity.
        """
        maximum = 0.0
        name = None
        worst_ratio = 0.0
        worst_name = None
        for constraint in model.component_data_objects(pe.Constraint, active=True):
            body = float(pe.value(constraint.body))
            if not math.isfinite(body):
                return math.inf, constraint.name, math.inf, constraint.name
            violation = 0.0
            if constraint.lower is not None:
                violation = max(violation, float(pe.value(constraint.lower)) - body)
            if constraint.upper is not None:
                violation = max(violation, body - float(pe.value(constraint.upper)))
            if violation > maximum:
                maximum = violation
                name = constraint.name
            if violation <= absolute_tolerance:
                # Within the absolute tolerance, so the term scale is not needed.
                continue
            allowance = max(
                absolute_tolerance,
                _RELATIVE_CONSTRAINT_TOLERANCE
                * self._constraint_term_scale(constraint),
            )
            ratio = violation / allowance if allowance > 0.0 else math.inf
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_name = constraint.name
        return max(maximum, 0.0), name, worst_ratio, worst_name

    def _variable_bound_violation(self, model) -> float:
        """Return the largest active Pyomo variable-bound violation."""
        maximum = 0.0
        for variable in model.component_data_objects(pe.Var, active=True):
            current = float(pe.value(variable))
            if not math.isfinite(current):
                return math.inf
            if variable.lb is not None:
                maximum = max(maximum, float(pe.value(variable.lb)) - current)
            if variable.ub is not None:
                maximum = max(maximum, current - float(pe.value(variable.ub)))
        return max(maximum, 0.0)

    def _quality(self, simulator, band: tuple[float, float]) -> dict:
        """Run post-solve electrical checks and return a serializable report."""
        model = simulator.model
        network = simulator.network
        report = simulator.curtailment_report()
        residuals = [float(inv.max_constraint_residual()) for inv in network.inverters]
        finite = [value for value in residuals if math.isfinite(value)]
        max_residual = max(finite, default=0.0)
        nonfinite = len(residuals) - len(finite)
        v_lo, v_hi = map(float, band)
        max_voltage = 0.0
        for bus in network.buses:
            idx = getattr(bus, "int_bus_id", None)
            if idx is None:
                continue
            ref = math.hypot(float(bus.Vr_pu), float(bus.Vi_pu))
            if ref <= 1e-9:
                continue
            magnitude = math.hypot(
                float(pe.value(model.ipopt_vr_list[idx])),
                float(pe.value(model.ipopt_vi_list[idx])),
            )
            ratio = magnitude / ref
            max_voltage = max(max_voltage, v_lo - ratio, ratio - v_hi)

        max_ampacity = 0.0
        rated_lines = 0
        for element in network.lines:
            for line in element.lines:
                ampacity = getattr(line, "ampacity_a", None)
                if ampacity is None or ampacity <= 0.0:
                    continue
                rated_lines += 1
                ir, ii = line.find_Ir_Ii_from()
                current = (
                    math.hypot(float(pe.value(ir)), float(pe.value(ii)))
                    * line.i_base_from
                )
                max_ampacity = max(max_ampacity, current / float(ampacity))

        max_apparent = 0.0
        for inv in network.inverters:
            vr = (
                float(pe.value(inv.ipopt_vr))
                * inv.bus.v_base_v
                / inv.interface_turns_ratio
            )
            vi = (
                float(pe.value(inv.ipopt_vi))
                * inv.bus.v_base_v
                / inv.interface_turns_ratio
            )
            ir = float(pe.value(inv.i_t2_r))
            ii = float(pe.value(inv.i_t2_i))
            p = vr * ir + vi * ii
            q = vi * ir - vr * ii
            max_apparent = max(max_apparent, math.hypot(p, q) / inv.rating_va)

        curtailed = sorted(float(item["curtailed_w"]) for item in report)
        total = sum(curtailed)
        mean = statistics.fmean(curtailed) if curtailed else 0.0
        std = statistics.pstdev(curtailed) if len(curtailed) > 1 else 0.0
        if total > 0.0 and curtailed:
            n = len(curtailed)
            gini = (
                2.0
                * sum((i + 1) * value for i, value in enumerate(curtailed))
                / (n * total)
                - (n + 1.0) / n
            )
        else:
            gini = 0.0
        solver_settings = getattr(network, "solver_settings", {}) or {}
        try:
            constraint_tolerance = float(
                solver_settings["constraint_violation_tolerance"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "the parsed scenario JSON must provide solver.constraint_violation_tolerance"
            ) from exc
        tolerance = max(5.0 * constraint_tolerance, 5e-5)
        (
            max_constraint_violation,
            largest_constraint,
            constraint_allowance_ratio,
            worst_scaled_constraint,
        ) = self._largest_constraint_violation(model, tolerance)
        valid = (
            constraint_allowance_ratio <= 1.0
            and self._variable_bound_violation(model) <= tolerance
            and nonfinite == 0
            and max_residual <= tolerance
            and max_voltage <= tolerance
            and max_ampacity <= 1.0 + tolerance
            and max_apparent <= 1.0 + tolerance
        )
        return {
            "valid": bool(valid),
            "tolerance": tolerance,
            "max_model_constraint_violation": max_constraint_violation,
            "largest_model_constraint": largest_constraint,
            "constraint_relative_tolerance": _RELATIVE_CONSTRAINT_TOLERANCE,
            "worst_constraint_allowance_ratio": constraint_allowance_ratio,
            "worst_scaled_constraint": worst_scaled_constraint,
            "max_variable_bound_violation": self._variable_bound_violation(model),
            "max_tsbi_residual": max_residual,
            "nonfinite_tsbi_residuals": nonfinite,
            "max_voltage_band_violation_pu": max(max_voltage, 0.0),
            "rated_line_phases": rated_lines,
            "max_line_ampacity_ratio": max_ampacity,
            "max_inverter_apparent_power_ratio": max_apparent,
            "curtailment_distribution": {
                "min_w": curtailed[0] if curtailed else 0.0,
                "mean_w": mean,
                "std_w": std,
                "max_w": curtailed[-1] if curtailed else 0.0,
                "active_units": sum(value > 1e-3 for value in curtailed),
                "gini": max(gini, 0.0),
            },
        }

    def _curtailment_norm_metrics(self, report: list[dict]) -> dict[str, float]:
        """Calculate raw, availability, and network curtailment norm metrics."""
        values = [abs(float(item["curtailed_w"])) for item in report]
        availability = [
            abs(float(item.get("availability_shortfall_w", 0.0))) for item in report
        ]
        network = [
            abs(float(item.get("network_curtailment_w", item["curtailed_w"])))
            for item in report
        ]
        return {
            "l1_w": sum(values),
            "l2_w": math.sqrt(sum(value * value for value in values)),
            "linf_w": max(values, default=0.0),
            "availability_shortfall_l1_w": sum(availability),
            "network_curtailment_l1_w": sum(network),
            "network_curtailment_l2_w": math.sqrt(
                sum(value * value for value in network)
            ),
            "network_curtailment_linf_w": max(network, default=0.0),
        }

    def _base_document(self, document: dict) -> dict:
        """Remove standalone inverter publications before a reference solve."""
        base = copy.deepcopy(document)
        instance_names = set(base.get("inverter_connections") or {})
        for name in instance_names:
            base.get("generator", {}).pop(name, None)
        for key in ("inverter", "inverter_connections", "pv"):
            base.pop(key, None)
        return base

    def solve(self, path) -> tuple[str, str, dict]:
        """Solve one scenario JSON and return its case, mode, and result entry."""
        if self.linear_solver in self.SCRATCH_FILE_LINEAR_SOLVERS:
            # A directory per solve, so nothing is written into the repository
            # and parallel solves cannot overwrite one another's factorisation.
            with (
                tempfile.TemporaryDirectory(prefix="ipopt-scratch-") as scratch,
                contextlib.chdir(scratch),
            ):
                return self._solve(path)
        return self._solve(path)

    def _solve(self, path) -> tuple[str, str, dict]:
        """Solve one scenario in whatever working directory the caller set."""
        timeout, tee, warm_start = self.timeout, self.tee, self.warm_start

        # Imported here: the parser reaches this module through the component
        # classes, so a module-level import would close the cycle.
        from src.bmopf_parser import check_scenario_is_current, load_bmopf_network

        scenario_path = Path(path)
        case_name = scenario_path.parent.parent.name
        try:
            scenario_file = str(scenario_path.relative_to(REPO_ROOT))
        except ValueError:
            scenario_file = str(scenario_path)
        document = json.loads(scenario_path.read_text())
        check_scenario_is_current(scenario_path, document)
        study = document.get("study") or {}
        inverter_definition = document.get("inverter") or {}
        controls = inverter_definition.get("controls") or {}
        active_power = controls.get("active_power") or {}
        reactive_power = controls.get("reactive_power") or {}
        mode = "{}_{}".format(
            str(active_power.get("mode", "")).upper(),
            str(reactive_power.get("mode", "")).upper(),
        )
        norm = str((study.get("objective") or {}).get("norm", "")).upper()
        band = self.voltage_band(case_name)
        solver_settings = self._effective_solver_settings(document)
        solver_timeout = float(
            timeout if timeout is not None else solver_settings["timeout_s"]
        )
        initial_delivery_fraction = float(solver_settings["initial_delivery_fraction"])

        base = load_bmopf_network(self._base_document(document))
        # The reference feeder is a derived view of the same standalone JSON.
        # Give it the scenario's solver block before solving so the base solve has
        # no hidden defaults of its own.
        base.solver_settings = copy.deepcopy(solver_settings)
        base_sim = NetworkSimulator(base)
        base_ok, base_msg, base_iterations, _ = base_sim.solve(
            timeout=solver_timeout, tee=tee
        )
        if not base_ok or base_sim.model is None:
            return (
                case_name,
                mode,
                {
                    "norm": norm,
                    "mode": mode,
                    "scenario_file": scenario_file,
                    "solved": False,
                    "msg": f"base solve failed: {base_msg}",
                    "iterations": base_iterations,
                    "requested_w": 0.0,
                    "delivered_w": 0.0,
                    "curtailed_w": 0.0,
                },
            )
        network = load_bmopf_network(document)
        network.solver_settings = copy.deepcopy(solver_settings)
        # The reference magnitudes define the voltage band and are always used.
        # Seeding each bus with the reference solve's voltages is separate: it only
        # changes where the optimization starts, so --no-warm-start can be given
        # without altering the band and therefore the answer.
        voltage_reference = {}
        for bus in network.buses:
            idx = bus.int_bus_id
            ref_vr = float(pe.value(base_sim.model.ipopt_vr_list[idx]))
            ref_vi = float(pe.value(base_sim.model.ipopt_vi_list[idx]))
            voltage_reference[idx] = math.hypot(ref_vr, ref_vi)
            if warm_start:
                bus.Vr_pu, bus.Vi_pu = ref_vr, ref_vi

        started = time.monotonic()
        promote_linf = norm == "LINF"

        def prepare(simulator):
            build = simulator.create_and_initialize_model()
            if isinstance(build, str) and build.startswith("Error"):
                return build
            built_constraints = simulator.build_constraints()
            if isinstance(built_constraints, str) and built_constraints.startswith(
                "Error"
            ):
                return built_constraints
            built_objective = simulator.build_objective()
            if isinstance(built_objective, str) and built_objective.startswith("Error"):
                return built_objective
            return None

        # Solve the requested norm directly, min-max included. LINF then costs two
        # solves rather than three, and reaches the same optimum: across the 40
        # LINF cases the direct solve matched the seeded one to a median of 0% in
        # the objective it minimises, and never landed on a worse peak.
        simulator = NetworkSimulatorInvOpt(
            network,
            norm=norm,
            voltage_band=band,
            voltage_reference=voltage_reference,
            initial_delivery_fraction=initial_delivery_fraction,
            scale_linf_variables=promote_linf,
        )
        preparation_error = prepare(simulator)
        if preparation_error is not None:
            return (
                case_name,
                mode,
                {"norm": norm, "solved": False, "msg": preparation_error},
            )
        ok, message, iterations, solve_time = simulator.solve(
            timeout=solver_timeout, tee=tee
        )
        if not ok:
            return (
                case_name,
                mode,
                {
                    "norm": norm,
                    "mode": mode,
                    "scenario_file": scenario_file,
                    "solved": False,
                    "msg": message,
                    "iterations": iterations,
                    "time_s": round(time.monotonic() - started, 3),
                    "voltage_band": {"minimum": band[0], "maximum": band[1]},
                    "solve_settings": {
                        "warm_start": bool(warm_start),
                        "initial_delivery_fraction": initial_delivery_fraction,
                        "linear_solver": solver_settings.get("linear_solver"),
                        "mumps_pivot_tolerance": solver_settings.get(
                            "mumps_pivot_tolerance"
                        ),
                        "max_iter": solver_settings.get("max_iter"),
                        "max_cpu_time_s": solver_timeout,
                    },
                },
            )
        report = simulator.curtailment_report()
        metrics = self._curtailment_norm_metrics(report)
        raw_requested = sum(item["requested_w"] for item in report)
        curtailed = metrics["l1_w"]
        availability_shortfall = metrics["availability_shortfall_l1_w"]
        network_curtailment = metrics["network_curtailment_l1_w"]
        delivered = max(raw_requested - curtailed, 0.0)
        quality = self._quality(simulator, band)
        if promote_linf:
            # Record the min-max objective the solve settled on. There is no
            # incumbent to compare it against: LINF is solved directly, so the
            # post-solve checks in _quality are what establish the result.
            quality["linf_final_objective"] = float(
                pe.value(simulator.model.max_curtail)
            )
        entry = {
            "norm": norm,
            "mode": mode,
            "scenario_file": scenario_file,
            "solved": bool(quality["valid"]),
            "msg": message or "optimal",
            "iterations": iterations,
            "time_s": round(float(solve_time), 3),
            "voltage_band": {"minimum": band[0], "maximum": band[1]},
            # Echoed so a row explains itself: the same case at a different
            # initial point or pivot tolerance is a different measurement.
            "solve_settings": {
                "warm_start": bool(warm_start),
                "initial_delivery_fraction": initial_delivery_fraction,
                "linear_solver": solver_settings.get("linear_solver"),
                "mumps_pivot_tolerance": solver_settings.get("mumps_pivot_tolerance"),
                "tol": solver_settings.get("tol"),
                "constr_viol_tol": solver_settings.get(
                    "constraint_violation_tolerance"
                ),
                "max_iter": solver_settings.get("max_iter"),
                "max_cpu_time_s": solver_timeout,
                "mu_strategy": solver_settings.get("mu_strategy"),
                "stages": 2,
            },
            "requested_w": raw_requested,
            "delivered_w": delivered,
            "curtailed_w": curtailed,
            "availability_shortfall_w": availability_shortfall,
            "network_curtailment_w": network_curtailment,
            "network_curtailment_basis": (
                "pv_source_w" if mode.startswith("MPPT_") else "ac_terminal_w"
            ),
            "delivered_fraction": delivered / raw_requested if raw_requested else 1.0,
            "curtailment_fraction": curtailed / raw_requested if raw_requested else 0.0,
            "availability_shortfall_fraction": (
                availability_shortfall / raw_requested if raw_requested else 0.0
            ),
            "network_curtailment_fraction": (
                network_curtailment / raw_requested if raw_requested else 0.0
            ),
            "fraction": delivered / raw_requested if raw_requested else 1.0,
            "objective_diagnostics": {
                "optimized_norm": norm,
                "scaled_model_objective": float(
                    pe.value(simulator.model.max_curtail)
                    if norm == "LINF"
                    else pe.value(simulator.model.objective)
                ),
                **metrics,
            },
            "quality": quality,
            "inverters": len(network.inverters),
        }
        if not quality["valid"]:
            entry["msg"] = "post-solve electrical validation failed"
        return case_name, mode, entry


class CurtailmentStudy:
    """Solve a set of scenarios and assemble the published result matrix.

    The sweep is a property of the study, so the modes and norms are stated
    here once and everything downstream reads them from the published matrix.
    """

    P_MODES = ("CONSTANT_P", "MPPT")
    Q_MODES = ("UPF", "CONSTANT_Q", "CPF", "VOLT_VAR")
    NORMS = ("L1", "L2", "LINF")

    def __init__(self, solver: ScenarioSolver, workers: int = 1):
        self.solver = solver
        self.workers = max(1, int(workers))

    @staticmethod
    def entries(results: dict):
        """Yield every result entry in a completed run."""
        for by_norm in results.values():
            for by_case in by_norm.values():
                yield from by_case.values()

    @classmethod
    def control_modes(cls) -> list[str]:
        """Return every active/reactive control combination, in sweep order."""
        return [
            f"{p_mode}_{q_mode}" for p_mode in cls.P_MODES for q_mode in cls.Q_MODES
        ]

    @classmethod
    def scenarios_per_case(cls) -> int:
        """Return how many scenarios one feeder contributes to a full sweep."""
        return len(cls.P_MODES) * len(cls.Q_MODES) * len(cls.NORMS)

    def units_and_base(self) -> dict:
        """Return the per-unit convention every reported quantity is expressed in."""
        from src.network_loader import DxNetworkModel

        return {
            "s_base_three_phase_va": s_base_3ph_va(),
            "s_base_single_phase_va": s_base_1ph_va(),
            "voltage_base": "each bus uses its own nominal line-to-neutral voltage",
            "frequency_hz": DxNetworkModel.FREQUENCY,
            "power_units": "W in reported quantities, per unit inside the model",
        }

    def cross_objective_validation(self, results: dict) -> dict:
        """Check that each solved control mode has distinct norm diagnostics."""
        failures = []
        checks = 0
        for case_results in results.values():
            for mode_results in case_results.values():
                entries = {norm: mode_results.get(norm) for norm in self.NORMS}
                if any(
                    not entry or not entry.get("solved") for entry in entries.values()
                ):
                    continue
                checks += 1
                labels = {
                    norm: str(
                        (entries[norm].get("objective_diagnostics") or {}).get(
                            "optimized_norm", ""
                        )
                    ).upper()
                    for norm in self.NORMS
                }
                objectives = {
                    norm: float(
                        (entries[norm].get("objective_diagnostics") or {}).get(
                            "scaled_model_objective"
                        )
                    )
                    for norm in self.NORMS
                }
                if labels != {norm: norm for norm in self.NORMS} or not all(
                    math.isfinite(value) for value in objectives.values()
                ):
                    failures.append(
                        {
                            "mode": mode_results,
                            "objective_labels": labels,
                            "objectives": objectives,
                        }
                    )
        return {
            "comparable_case_modes": checks,
            "valid": not failures,
            "failures": failures,
        }

    def _ordered(self, results: dict, cases: list[str]) -> dict:
        """Return matrix results in stable norm, case and control-mode order."""
        mode_order = {mode: index for index, mode in enumerate(self.control_modes())}
        ordered = {}
        for norm in self.NORMS:
            by_case = results.get(norm, {})
            ordered[norm] = {}
            for case in cases:
                if case not in by_case:
                    continue
                by_mode = by_case[case]
                ordered[norm][case] = {
                    mode: by_mode[mode]
                    for mode in sorted(
                        by_mode,
                        key=lambda mode: (mode_order.get(mode, len(mode_order)), mode),
                    )
                }
        return ordered

    def run(self, paths: list) -> dict:
        """Solve every scenario, reporting progress, and return results by norm."""
        results = {norm: {} for norm in self.NORMS}
        start = time.monotonic()
        completed = 0

        def record(result):
            nonlocal completed
            case, mode, entry = result
            norm = entry.get("norm", "UNKNOWN")
            results.setdefault(norm, {}).setdefault(case, {}).setdefault(mode, entry)
            completed += 1
            status = "PASS" if entry.get("solved") else "FAIL"
            print(
                f"[{completed}/{len(paths)}] {case:<38} {mode:<28} {norm:<4} "
                f"{status} ({time.monotonic() - start:.0f}s)",
                flush=True,
            )

        if self.workers == 1:
            for path in paths:
                record(self.solver.solve(path))
        else:
            # 'spawn' is available on Windows and macOS and avoids inheriting
            # OpenDSS/Pyomo state from the parent process. Every worker entry
            # point stays at module scope so both platforms can spawn it.
            with ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
            ) as pool:
                futures = [pool.submit(self.solver.solve, path) for path in paths]
                for future in as_completed(futures):
                    record(future.result())
        return results

    def publish(
        self,
        results: dict,
        cases: list[str],
        scenario_count: int,
        labels: dict | None = None,
    ) -> dict:
        """Assemble the published result matrix from a completed run."""
        return {
            "meta": {
                "method": "standalone_json_joint_nonlinear_opt",
                "formulation": (
                    "phase-domain current-voltage AC optimization; each inverter "
                    "is a two-stage bidirectional equivalent circuit solved "
                    "jointly with the network, minimising the selected norm of "
                    "per-inverter curtailment subject to hard relative voltage "
                    "limits, BMOPF from-side series-current ratings and inverter "
                    "kVA limits"
                ),
                "stages_per_case": 2,
                "cases": cases,
                "case_order": cases,
                "scenario_count": scenario_count,
                "control_modes": self.control_modes(),
                "case_labels": dict(labels or {}),
                "environment": RunProvenance().describe(),
                "units_and_base": self.units_and_base(),
                "post_solve_validation": {
                    "solved": sum(
                        entry.get("solved", False) for entry in self.entries(results)
                    )
                },
                "cross_objective_validation": self.cross_objective_validation(
                    {
                        case: {
                            mode: {
                                norm: results.get(norm, {}).get(case, {}).get(mode)
                                for norm in self.NORMS
                            }
                            for mode in {
                                mode
                                for norm in self.NORMS
                                for mode in results.get(norm, {}).get(case, {})
                            }
                        }
                        for case in cases
                    }
                ),
            },
            "results": self._ordered(results, cases),
        }

    def failures(self, results: dict) -> list:
        """Return the entries that did not solve."""
        return [entry for entry in self.entries(results) if not entry.get("solved")]
