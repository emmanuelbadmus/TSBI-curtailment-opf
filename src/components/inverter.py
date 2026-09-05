"""Physics-based PV-to-grid inverter with a JSON-defined DC source."""

from __future__ import annotations

import copy
import math
import re
from itertools import count

from pyomo.environ import ConstraintList, exp, sqrt, value

from src import network_simulator as per_unit

from .bus import Bus
from .pv import (
    calculate_mpp,
    operating_point_for_power,
    validate_sdm,
)


class InverterSpecError(ValueError):
    """Raised when an inverter type, control, or connection is invalid."""


PHASE_ALIASES = {
    "1": "1",
    "2": "2",
    "3": "3",
    "a": "1",
    "b": "2",
    "c": "3",
}


def _number(block: dict, key: str, *, positive: bool = False) -> float:
    """Read one finite number from a block, or say which key was wrong."""
    try:
        result = float(block[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise InverterSpecError(f"inverter requires numeric {key!r}") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        requirement = "positive and finite" if positive else "finite"
        raise InverterSpecError(f"inverter {key!r} must be {requirement}")
    return result


def _require_exact_keys(block: dict, expected: set[str], context: str) -> None:
    """Reject missing or unused keys in one compact study-input object."""
    present = set(block)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unused/unsupported " + ", ".join(extra))
        raise InverterSpecError(f"{context}: {'; '.join(details)}")


def validate_inverter_type(raw: dict, *, active_power_mode: str | None = None) -> dict:
    """Validate one complete JSON inverter electrical definition."""
    if not isinstance(raw, dict):
        raise InverterSpecError("inverter electrical definition must be an object")
    spec = copy.deepcopy(raw)
    if "model" in spec:
        raise InverterSpecError(
            "inverter electrical data must not contain model; put it on inverter.model"
        )
    if "source" in spec:
        raise InverterSpecError(
            "inverter electrical data contains provenance metadata; remove 'source'"
        )
    if "controls" in spec:
        raise InverterSpecError(
            "inverter electrical data must contain electrical data only; put controls in the scenario inverter block"
        )
    if "rating_va" in spec:
        raise InverterSpecError(
            "inverter electrical data contains rating_va; put the nameplate as inverter.s_va"
        )

    mode = None if active_power_mode is None else str(active_power_mode).upper()
    if mode not in (None, "CONSTANT_P", "MPPT"):
        raise InverterSpecError(f"unsupported active-power mode {active_power_mode!r}")
    required = {
        "rated_ac_terminal_voltage_v",
        "dc_link_voltage_v",
        "frequency_hz",
        "loss_smoothing_epsilon",
        "duty_cycle_limits",
        "semiconductor",
        "first_stage_converter",
        "second_stage_converter",
        "lcl_filter",
    }
    if mode in (None, "CONSTANT_P"):
        required.add("constant_p_pv_voltage_v")
    _require_exact_keys(spec, required, "inverter.electrical")

    _number(spec, "rated_ac_terminal_voltage_v", positive=True)
    _number(spec, "dc_link_voltage_v", positive=True)
    if mode in (None, "CONSTANT_P"):
        _number(spec, "constant_p_pv_voltage_v", positive=True)
    _number(spec, "frequency_hz", positive=True)
    _number(spec, "loss_smoothing_epsilon", positive=True)

    duty = spec.get("duty_cycle_limits")
    if not isinstance(duty, dict):
        raise InverterSpecError("duty_cycle_limits must be an object")
    _require_exact_keys(duty, {"minimum", "maximum"}, "duty_cycle_limits")
    duty_minimum = _number(duty, "minimum")
    duty_maximum = _number(duty, "maximum")
    if not 0.0 < duty_minimum < duty_maximum < 1.0:
        raise InverterSpecError("duty_cycle_limits must lie strictly inside (0, 1)")

    semiconductor = spec.get("semiconductor")
    if not isinstance(semiconductor, dict):
        raise InverterSpecError("inverter electrical data requires semiconductor data")
    for section, fields in {
        "mosfet": (
            "threshold_v",
            "on_resistance_ohm",
            "turn_on_delay_s",
            "rise_time_s",
            "turn_off_delay_s",
            "fall_time_s",
        ),
        "diode": ("forward_v", "on_resistance_ohm", "recovery_time_s"),
    }.items():
        values = semiconductor.get(section)
        if not isinstance(values, dict):
            raise InverterSpecError(
                f"inverter electrical data requires semiconductor.{section}"
            )
        _require_exact_keys(values, set(fields), f"semiconductor.{section}")
        for key in fields:
            _number(values, key, positive=key.endswith(("_ohm", "_s")))
    _require_exact_keys(
        semiconductor, {"mosfet", "diode"}, "inverter.electrical.semiconductor"
    )

    first_stage = spec.get("first_stage_converter")
    second_stage = spec.get("second_stage_converter")
    for section, values in (
        ("first_stage_converter", first_stage),
        ("second_stage_converter", second_stage),
    ):
        if not isinstance(values, dict):
            raise InverterSpecError(f"inverter electrical data requires {section} data")
        _number(values, "switching_frequency_hz", positive=True)
    _require_exact_keys(
        first_stage,
        {"switching_frequency_hz", "inductor_resistance_ohm"},
        "inverter.electrical.first_stage_converter",
    )
    _require_exact_keys(
        second_stage,
        {"switching_frequency_hz"},
        "inverter.electrical.second_stage_converter",
    )
    _number(first_stage, "inductor_resistance_ohm", positive=True)

    lcl = spec.get("lcl_filter")
    if not isinstance(lcl, dict):
        raise InverterSpecError("inverter electrical data requires lcl_filter data")
    lcl_fields = (
        "inverter_side_inductance_h",
        "grid_side_inductance_h",
        "filter_capacitance_f",
        "damping_resistance_ohm",
        "inverter_side_resistance_ohm",
        "grid_side_resistance_ohm",
    )
    _require_exact_keys(lcl, set(lcl_fields), "inverter.electrical.lcl_filter")
    for key in lcl_fields:
        _number(lcl, key, positive=True)
    return spec


def inverter_connection_id(terminal: str, used: set[str] | None = None) -> str:
    """Return a stable human-readable ID for one bus-phase connection."""
    bus_name, phase = parse_terminal(terminal)
    bus_token = bus_name
    if bus_token[:1].lower() == "n" and bus_token[1:].isdigit():
        bus_token = bus_token[1:]
    bus_token = re.sub(r"[^A-Za-z0-9]+", "_", bus_token).strip("_").lower()
    phase_token = {"1": "a", "2": "b", "3": "c"}[phase]
    base = f"inverter_{bus_token}{phase_token}"
    if used is None:
        return base
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _validate_volt_var_control(control: dict) -> dict:
    """Validate the Volt-VAR settings of a scenario control block."""
    if not isinstance(control, dict):
        raise InverterSpecError("VOLT_VAR reactive_power must be an object")
    knees = control.get("voltage_knees_pu")
    if not isinstance(knees, list) or len(knees) != 4:
        raise InverterSpecError("VOLT_VAR voltage_knees_pu must contain four values")
    knees = [_number({"value": value}, "value") for value in knees]
    if not (knees[0] < knees[1] <= knees[2] < knees[3]):
        raise InverterSpecError("Volt-VAR knees must be ordered")
    q_min = _number(control, "reactive_power_minimum_pu")
    q_max = _number(control, "reactive_power_maximum_pu")
    if not -1.0 <= q_min < 0.0:
        raise InverterSpecError("VOLT_VAR reactive_power_minimum_pu must be in [-1, 0)")
    if not 0.0 < q_max <= 1.0:
        raise InverterSpecError("VOLT_VAR reactive_power_maximum_pu must be in (0, 1]")
    v_ref = _number(control, "reference_voltage_pu")
    if not 0.5 < v_ref <= 1.5:
        raise InverterSpecError("VOLT_VAR reference_voltage_pu must be in (0.5, 1.5]")
    _number(control, "curve_smoothing_epsilon_pu", positive=True)
    return {
        "knees_pu": knees,
        "q_min_pu": q_min,
        "q_max_pu": q_max,
        "v_ref_pu": v_ref,
        "smoothing_epsilon": float(control["curve_smoothing_epsilon_pu"]),
    }


def validate_inverter_instance(raw: dict, types: dict) -> dict:
    """Validate one inverter connection record and return a private copy."""
    if not isinstance(raw, dict):
        raise InverterSpecError("inverter instance must be an object")
    instance = copy.deepcopy(raw)
    type_name = str(instance.get("type", ""))
    if not type_name or type_name not in types:
        raise InverterSpecError(
            f"inverter instance references unknown type {type_name!r}"
        )
    terminal = str(instance.get("terminal", ""))
    if "." not in terminal:
        raise InverterSpecError("inverter instance requires terminal BUS.PHASE")
    _number(instance, "s_va", positive=True)
    active = instance.get("active_power")
    reactive = instance.get("reactive_power")
    weights = instance.get("weights")
    if not isinstance(active, dict) or not isinstance(reactive, dict):
        raise InverterSpecError(
            "inverter instance requires active_power and reactive_power"
        )
    if str(active.get("mode", "")).upper() not in ("CONSTANT_P", "MPPT"):
        raise InverterSpecError("active_power.mode must be CONSTANT_P or MPPT")
    if str(reactive.get("mode", "")).upper() not in (
        "UPF",
        "CONSTANT_Q",
        "CPF",
        "VOLT_VAR",
    ):
        raise InverterSpecError("reactive_power.mode is unsupported")
    if not isinstance(weights, dict):
        raise InverterSpecError("each inverter requires a weights object")
    _require_exact_keys(weights, {"curtailment"}, "inverter.weights")
    weight = _number(weights, "curtailment", positive=True)
    if weight <= 0.0:
        raise InverterSpecError("inverter weights.curtailment must be positive")

    _require_exact_keys(
        active,
        {"mode", "requested_ac_power_pu"},
        "inverter.controls.active_power",
    )
    requested_ac_power_pu = _number(active, "requested_ac_power_pu")
    if not 0.0 <= requested_ac_power_pu <= 1.0:
        raise InverterSpecError("active_power.requested_ac_power_pu must be in [0, 1]")
    mode = str(reactive["mode"]).upper()
    if mode == "CONSTANT_Q":
        _require_exact_keys(
            reactive,
            {"mode", "reactive_power_setpoint_pu"},
            "inverter.controls.reactive_power",
        )
        q_pu = _number(reactive, "reactive_power_setpoint_pu")
        if abs(q_pu) > 1.0:
            raise InverterSpecError(
                "CONSTANT_Q reactive_power_setpoint_pu must be in [-1, 1]"
            )
    elif mode == "CPF":
        _require_exact_keys(
            reactive,
            {"mode", "power_factor"},
            "inverter.controls.reactive_power",
        )
        pf = _number(reactive, "power_factor")
        if not -1.0 <= pf <= 1.0 or abs(pf) < 1e-12:
            raise InverterSpecError("CPF power_factor must be nonzero and in [-1, 1]")
    elif mode == "VOLT_VAR":
        _require_exact_keys(
            reactive,
            {
                "mode",
                "voltage_knees_pu",
                "reactive_power_minimum_pu",
                "reactive_power_maximum_pu",
                "reference_voltage_pu",
                "curve_smoothing_epsilon_pu",
            },
            "inverter.controls.reactive_power",
        )
        _validate_volt_var_control(reactive)
    else:
        _require_exact_keys(reactive, {"mode"}, "inverter.controls.reactive_power")
    return instance


def dispatch_from_instance(instance: dict, pv: dict | None = None) -> dict[str, object]:
    """Derive model dispatch and control settings from JSON controls."""
    s_va = float(instance["s_va"])
    active = instance["active_power"]
    reactive = instance["reactive_power"]
    p_mode = str(active["mode"]).upper()
    p_w = float(active["requested_ac_power_pu"]) * s_va
    pv_source = pv or {}
    pv_sdm = None
    if p_mode == "MPPT":
        pv_sdm = validate_sdm(pv_source)
    q_mode = str(reactive["mode"]).upper()
    q_var = (
        float(reactive["reactive_power_setpoint_pu"]) * s_va
        if q_mode == "CONSTANT_Q"
        else 0.0
    )
    volt_var = _validate_volt_var_control(reactive) if q_mode == "VOLT_VAR" else {}
    return {
        "p_w": p_w,
        "q_var": q_var,
        "p_mode": p_mode,
        "pv_sdm": pv_sdm,
        "control_mode": q_mode,
        "power_factor": reactive.get("power_factor") if q_mode == "CPF" else None,
        "weight": float(instance["weights"]["curtailment"]),
        "volt_var_knees": volt_var.get("knees_pu"),
        "volt_var_q_min_pu": volt_var.get("q_min_pu"),
        "volt_var_q_max_pu": volt_var.get("q_max_pu"),
        "volt_var_v_ref_pu": volt_var.get("v_ref_pu"),
        "volt_var_smoothing_epsilon": volt_var.get("smoothing_epsilon"),
    }


def parse_terminal(spec: str) -> tuple[str, str]:
    """Parse BUS.PHASE, accepting A/B/C as aliases for phases 1/2/3."""
    token = str(spec).strip()
    if "." not in token:
        raise ValueError(f"Invalid terminal '{spec}'. Use BUS.PHASE, for example n4.1.")
    bus_name, phase_token = token.rsplit(".", 1)
    bus_name = bus_name.strip()
    phase = PHASE_ALIASES.get(phase_token.strip().lower())
    if not bus_name or phase is None:
        raise ValueError(
            f"Invalid terminal '{spec}'. Phase must be 1, 2, 3, A, B, or C."
        )
    return bus_name, phase


_VOLTAGE_BASE_EPS = 1e-9


class Inverter:
    """PV-to-grid TSBI model attached to one phase-domain terminal.

    The JSON inverter type supplies the constant-P PV voltage, converter
    losses, switching data, and LCL filter values.  The source-side state is a
    PV/DC port.  There is no storage state or storage equation in this model.
    """

    _ids = count(0)

    def __init__(
        self,
        name: str,
        bus: Bus,
        p_w: float,
        q_var: float,
        *,
        phase=None,
        inverter_type: dict,
        rating_va: float | None = None,
        ac_terminal_voltage_v: float | None = None,
        dc_link_voltage_v: float | None = None,
        constant_p_pv_voltage_v: float | None = None,
        control_mode: str = "CONSTANT_Q",
        p_mode: str = "CONSTANT_P",
        pv_sdm: dict | None = None,
        power_factor: float | None = None,
        volt_var_knees: tuple[float, float, float, float] | None = None,
        volt_var_q_max_pu: float | None = None,
        volt_var_q_min_pu: float | None = None,
        volt_var_v_ref_pu: float | None = None,
        volt_var_smoothing_epsilon: float | None = None,
        optimization_weight: float = 1.0,
    ):
        self.id = next(self._ids)
        self.name = str(name)
        self.bus = bus
        self.bus_idx = bus.int_bus_id
        self.phase = str(phase if phase is not None else bus.NodePhase)

        pmode = str(p_mode).strip().upper()
        if pmode not in ("CONSTANT_P", "MPPT"):
            raise ValueError(
                f"Inverter {self.name}: p_mode must be 'CONSTANT_P' or "
                f"'MPPT', got {p_mode!r}."
            )
        self.p_mode = pmode
        self.inverter_type = validate_inverter_type(
            inverter_type, active_power_mode=pmode
        )
        self.raw_p_w = float(p_w)
        self.raw_q_var = float(q_var)
        if rating_va is None:
            raise ValueError(
                f"Inverter {self.name}: nameplate s_va must be provided explicitly."
            )
        self.rating_va = float(rating_va)
        self.ac_terminal_voltage_v = float(
            self.inverter_type["rated_ac_terminal_voltage_v"]
            if ac_terminal_voltage_v is None
            else ac_terminal_voltage_v
        )
        self.dc_link_voltage_v = float(
            self.inverter_type["dc_link_voltage_v"]
            if dc_link_voltage_v is None
            else dc_link_voltage_v
        )
        self.constant_p_pv_voltage_v = (
            None
            if pmode == "MPPT"
            else float(
                self.inverter_type["constant_p_pv_voltage_v"]
                if constant_p_pv_voltage_v is None
                else constant_p_pv_voltage_v
            )
        )
        self.curtailment_weight = float(optimization_weight)

        # Dispatch set point and control mode.  ``p_target`` is what the AC
        # control equation enforces (the curtailment engine moves it);
        # ``p_init`` seeds the TSBI initial state at the same dispatch.
        self.p_target = self.raw_p_w
        self.p_init = self.raw_p_w
        # Active-power side: CONSTANT_P (11b) is demand-driven, so the PV
        # setpoint is fully dispatchable. MPPT (11f) is supply-driven, so the
        # active SDM equation determines the PV
        # voltage/current curve and its calculated P_MPP; the optimization
        # may move below that point only when AC/network curtailment requires
        # it. The source-side limit comes only from the active SDM equation.
        self.pv_sdm = validate_sdm(pv_sdm) if pmode == "MPPT" else None
        if pmode == "MPPT" and self.pv_sdm is None:
            raise ValueError(
                f"Inverter {self.name}: MPPT requires active single-diode PV data."
            )
        self.pv_mpp = calculate_mpp(self.pv_sdm) if self.pv_sdm is not None else None
        self.pv_voltage_max_v = (
            None if self.pv_mpp is None else float(self.pv_mpp["v_oc_v"])
        )
        self.pv_current_max_a = (
            None if self.pv_mpp is None else float(self.pv_mpp["i_sc_a"])
        )
        mode = str(control_mode).upper()
        if mode not in ("UPF", "CPF", "CONSTANT_Q", "VOLT_VAR"):
            raise ValueError(
                f"Inverter {self.name}: control_mode must be one of "
                "('UPF', 'CPF', 'CONSTANT_Q', 'VOLT_VAR'), "
                f"got {control_mode!r}."
            )
        self.control_mode = mode
        self.power_factor = None if power_factor is None else float(power_factor)
        if self.power_factor is not None and not (-1.0 <= self.power_factor <= 1.0):
            raise ValueError(
                f"Inverter {self.name}: power_factor must be in [-1, 1]; "
                f"got {self.power_factor}."
            )
        if mode == "CPF" and self.power_factor is None:
            raise ValueError(f"Inverter {self.name}: CPF mode requires power_factor.")
        if mode == "VOLT_VAR":
            if (
                volt_var_knees is None
                or volt_var_q_max_pu is None
                or volt_var_q_min_pu is None
                or volt_var_v_ref_pu is None
                or volt_var_smoothing_epsilon is None
            ):
                raise ValueError(
                    f"Inverter {self.name}: VOLT_VAR requires all curve parameters."
                )
            self.volt_var_knees = tuple(float(knee) for knee in volt_var_knees)
            if len(self.volt_var_knees) != 4:
                raise ValueError(
                    f"Inverter {self.name}: volt_var_knees needs 4 values."
                )
            v1, v2, v3, v4 = self.volt_var_knees
            if not (v1 < v2 <= v3 < v4):
                raise ValueError(
                    f"Inverter {self.name}: knees must satisfy V1 < V2 <= V3 < V4."
                )
            self.volt_var_q_max_pu = float(volt_var_q_max_pu)
            if not 0.0 < self.volt_var_q_max_pu <= 1.0:
                raise ValueError(
                    f"Inverter {self.name}: volt_var_q_max_pu must be in (0, 1]."
                )
            self.volt_var_q_min_pu = float(volt_var_q_min_pu)
            if not -1.0 <= self.volt_var_q_min_pu < 0.0:
                raise ValueError(
                    f"Inverter {self.name}: volt_var_q_min_pu must be in [-1, 0)."
                )
            self.volt_var_v_ref_pu = float(volt_var_v_ref_pu)
            if not 0.5 < self.volt_var_v_ref_pu <= 1.5:
                raise ValueError(
                    f"Inverter {self.name}: volt_var_v_ref_pu must be in (0.5, 1.5]."
                )
        else:
            # Volt-VAR data is a scenario control, not a mandatory property
            # of every electrical inverter type.  Non-Volt-VAR models do not
            # read these attributes when building their equations.
            self.volt_var_knees = (
                tuple(float(knee) for knee in volt_var_knees)
                if volt_var_knees is not None
                else None
            )
            self.volt_var_q_max_pu = (
                None if volt_var_q_max_pu is None else float(volt_var_q_max_pu)
            )
            self.volt_var_q_min_pu = (
                None if volt_var_q_min_pu is None else float(volt_var_q_min_pu)
            )
            self.volt_var_v_ref_pu = (
                None if volt_var_v_ref_pu is None else float(volt_var_v_ref_pu)
            )

        numbers = [
            self.raw_p_w,
            self.raw_q_var,
            self.rating_va,
            self.ac_terminal_voltage_v,
            self.dc_link_voltage_v,
            self.curtailment_weight,
        ]
        if self.constant_p_pv_voltage_v is not None:
            numbers.append(self.constant_p_pv_voltage_v)
        if not all(math.isfinite(number) for number in numbers):
            raise ValueError(f"Inverter {self.name}: inputs must be finite.")
        if self.rating_va <= 0.0:
            raise ValueError(f"Inverter {self.name}: rating must be positive.")
        if self.ac_terminal_voltage_v <= 0.0:
            raise ValueError(
                f"Inverter {self.name}: AC terminal voltage must be positive."
            )
        if self.dc_link_voltage_v <= 0.0:
            raise ValueError(f"Inverter {self.name}: DC-link voltage must be positive.")
        if (
            self.constant_p_pv_voltage_v is not None
            and self.constant_p_pv_voltage_v <= 0.0
        ):
            raise ValueError(
                f"Inverter {self.name}: constant-P PV voltage must be positive."
            )
        if self.curtailment_weight <= 0.0 or not math.isfinite(self.curtailment_weight):
            raise ValueError(
                f"Inverter {self.name}: curtailment weight must be positive."
            )

        requested_va = math.hypot(self.raw_p_w, self.raw_q_var)
        if requested_va > self.rating_va + max(1e-9, self.rating_va * 1e-12):
            raise ValueError(
                f"Inverter {self.name}: requested {requested_va:.6g} VA exceeds "
                f"the {self.rating_va:.6g} VA rating."
            )

        self.P_pu = per_unit.power_to_pu(self.raw_p_w, s_base_va=bus.s_base_va)
        self.Q_pu = per_unit.power_to_pu(self.raw_q_var, s_base_va=bus.s_base_va)

        # A lossless interface transformer is mandatory when the feeder phase
        # voltage differs from the published 120 V inverter terminal.
        self.interface_turns_ratio = bus.v_base_v / self.ac_terminal_voltage_v

        # Converter and filter parameters are copied only from this JSON type.
        self.smoothing_epsilon = float(self.inverter_type["loss_smoothing_epsilon"])
        semiconductor = self.inverter_type["semiconductor"]
        mosfet = semiconductor["mosfet"]
        diode = semiconductor["diode"]
        self.mosfet_threshold_v = float(mosfet["threshold_v"])
        self.mosfet_resistance_ohm = float(mosfet["on_resistance_ohm"])
        self.diode_forward_v = float(diode["forward_v"])
        self.diode_resistance_ohm = float(diode["on_resistance_ohm"])
        first_stage = self.inverter_type["first_stage_converter"]
        second_stage = self.inverter_type["second_stage_converter"]
        self.fsc_inductor_resistance_ohm = float(first_stage["inductor_resistance_ohm"])
        self.fsc_switching_frequency_hz = float(first_stage["switching_frequency_hz"])
        self.ssc_switching_frequency_hz = float(second_stage["switching_frequency_hz"])
        self.turn_on_time_s = float(mosfet["turn_on_delay_s"] + mosfet["rise_time_s"])
        self.turn_off_time_s = float(mosfet["turn_off_delay_s"] + mosfet["fall_time_s"])
        self.diode_recovery_time_s = float(diode["recovery_time_s"])
        lcl = self.inverter_type["lcl_filter"]
        self.lcl_l1_h = float(lcl["inverter_side_inductance_h"])
        self.lcl_l2_h = float(lcl["grid_side_inductance_h"])
        self.lcl_capacitance_f = float(lcl["filter_capacitance_f"])
        self.lcl_damping_resistance_ohm = float(lcl["damping_resistance_ohm"])
        self.lcl_r1_ohm = float(lcl["inverter_side_resistance_ohm"])
        self.lcl_r2_ohm = float(lcl["grid_side_resistance_ohm"])
        self.omega_rad_s = 2.0 * math.pi * float(self.inverter_type["frequency_hz"])
        duty_limits = self.inverter_type["duty_cycle_limits"]
        self.duty_cycle_min = float(duty_limits["minimum"])
        self.duty_cycle_max = float(duty_limits["maximum"])
        self.volt_var_smoothing_epsilon = (
            None if mode != "VOLT_VAR" else float(volt_var_smoothing_epsilon)
        )

        # Mapped Pyomo variables.
        self.ipopt_vr = None
        self.ipopt_vi = None
        self.ipopt_ir = None
        self.ipopt_ii = None
        self.v_pv = None
        self.i_pv = None
        self.i_dc = None
        self.duty_cycle = None
        self.magnitude_modulation = None
        self.modulation_r = None
        self.modulation_i = None
        self.v_ac_r = None
        self.v_ac_i = None
        self.i_ac_r = None
        self.i_ac_i = None
        self.i_t2_r = None
        self.i_t2_i = None
        self.apparent_power_constraint = None
        self.pv_mpp_constraint = None

        self._model = None
        self._constraint_list = None
        self._constraints_built = False
        self._constraint_expressions = {}

    @property
    def initial_state(self) -> dict[str, float]:
        """Return a physics-informed initial point in SI units.

        If ``_warm_state`` was stored from a previous solve (the engine's
        continuation), that solved point is reused so the next solve starts
        near the previous feasible solution instead of from the flat-start
        physics guess.
        """
        warm = getattr(self, "_warm_state", None)
        if warm is not None:
            return dict(warm)
        grid_v = complex(self.bus.Vr_pu, self.bus.Vi_pu) * self.bus.v_base_v
        v_t2 = grid_v / self.interface_turns_ratio
        if abs(v_t2) <= _VOLTAGE_BASE_EPS:
            v_t2 = complex(self.ac_terminal_voltage_v, 0.0)

        q_guess = self.initial_reactive_power(v_t2)
        apparent_power = complex(self.p_init, q_guess)
        i_t2 = (apparent_power / v_t2).conjugate()

        x1 = self.omega_rad_s * self.lcl_l1_h
        x2 = self.omega_rad_s * self.lcl_l2_h
        xc = 1.0 / (self.omega_rad_s * self.lcl_capacitance_f)
        z1 = complex(self.lcl_r1_ohm, x1)
        z2 = complex(self.lcl_r2_ohm, x2)
        z_damped_cap = complex(self.lcl_damping_resistance_ohm, -xc)

        v_cap = v_t2 + z2 * i_t2
        i_ac = i_t2 + v_cap / z_damped_cap
        v_ac = v_cap + z1 * i_ac

        modulation = math.sqrt(2.0) * v_ac / self.dc_link_voltage_v
        i_ac_mag = max(abs(i_ac), math.sqrt(self.smoothing_epsilon))
        m_cos_phi = (
            modulation.real * i_ac.real + modulation.imag * i_ac.imag
        ) / i_ac_mag
        m_cos_phi_abs = math.sqrt(m_cos_phi * m_cos_phi + self.smoothing_epsilon)
        i_ac_sq = abs(i_ac) ** 2
        transistor_avg = (
            math.sqrt(2.0)
            * i_ac_mag
            * (math.pi * m_cos_phi_abs + 4.0)
            / (8.0 * math.pi)
        )
        diode_avg = (
            math.sqrt(2.0)
            * i_ac_mag
            * (4.0 - math.pi * m_cos_phi_abs)
            / (8.0 * math.pi)
        )
        transistor_rms_sq = (
            i_ac_sq * (8.0 * m_cos_phi_abs + 3.0 * math.pi) / (12.0 * math.pi)
        )
        diode_rms_sq = (
            i_ac_sq * (3.0 * math.pi - 8.0 * m_cos_phi_abs) / (12.0 * math.pi)
        )
        conduction_loss_w = 4.0 * (
            transistor_avg * self.mosfet_threshold_v
            + transistor_rms_sq * self.mosfet_resistance_ohm
            + diode_avg * self.diode_forward_v
            + diode_rms_sq * self.diode_resistance_ohm
        )
        v_conduction = conduction_loss_w * i_ac / max(i_ac_sq, self.smoothing_epsilon)
        v_ac_hat = v_ac + v_conduction
        modulation = math.sqrt(2.0) * v_ac_hat / self.dc_link_voltage_v
        if abs(modulation) > 0.95:
            modulation *= 0.95 / abs(modulation)

        ssc_switching_factor = (
            2.0
            * math.sqrt(2.0)
            / math.pi
            * self.ssc_switching_frequency_hz
            * (self.turn_on_time_s + self.turn_off_time_s + self.diode_recovery_time_s)
        )
        i_switching = ssc_switching_factor * i_ac_mag
        p_ac_hat = v_ac_hat.real * i_ac.real + v_ac_hat.imag * i_ac.imag
        i_dc = p_ac_hat / self.dc_link_voltage_v + i_switching

        if self.pv_sdm is not None:
            # Keep the initial PV state on the same SDM curve as the active
            # model.  Curtailed solves start at a small high-voltage/low-
            # current point; direct power-flow solves start near P_MPP.
            source_power_guess = min(
                self.pv_mpp["p_mpp_w"] * 0.98,
                max(5.0, self.p_init * 1.05),
            )
            pv_point = operating_point_for_power(source_power_guess, self.pv_sdm)
            v_pv = pv_point["v_pv_v"]
            i_pv = pv_point["i_pv_a"]
        else:
            v_pv = self.constant_p_pv_voltage_v
            i_pv = self.dc_link_voltage_v * i_dc / max(v_pv, 1.0)
        duty_cycle = self.dc_link_voltage_v / (self.dc_link_voltage_v + max(v_pv, 1.0))
        duty_cycle = min(
            self.duty_cycle_max,
            max(self.duty_cycle_min, duty_cycle),
        )

        return {
            "v_pv": v_pv,
            "i_pv": i_pv,
            "i_dc": i_dc,
            "duty_cycle": duty_cycle,
            "magnitude_modulation": abs(modulation),
            "modulation_r": modulation.real,
            "modulation_i": modulation.imag,
            "v_ac_r": v_ac.real,
            "v_ac_i": v_ac.imag,
            "i_ac_r": i_ac.real,
            "i_ac_i": i_ac.imag,
            "i_t2_r": i_t2.real,
            "i_t2_i": i_t2.imag,
        }

    def initial_reactive_power(self, v_t2: complex) -> float:
        """Return the reactive setpoint used by the physics initial point."""
        if self.control_mode == "CPF" and self.power_factor is not None:
            q_guess = (
                self.p_init * math.sqrt(1.0 - self.power_factor**2) / self.power_factor
            )
        elif self.control_mode == "VOLT_VAR":
            # Start at the droop commanded by the CURRENT bus voltage, not at
            # the deadband centre: at 0 W on a low-voltage feeder the droop
            # commands real VARs, and starting at zero current makes the SSC
            # initial point singular.
            v1, v2, v3, v4 = self.volt_var_knees
            q_bar = self.volt_var_q_max_pu * self.rating_va
            q_under = self.volt_var_q_min_pu * self.rating_va
            v_mag_pu = abs(v_t2) / self.ac_terminal_voltage_v
            v_ratio = v_mag_pu / self.volt_var_v_ref_pu

            def smooth_max(x):
                return (x + math.sqrt(x * x + self.smoothing_epsilon)) / 2.0

            q_guess = (
                q_bar
                - (q_bar / (v2 - v1))
                * (smooth_max(v_ratio - v1) - smooth_max(v_ratio - v2))
                + (q_under / (v4 - v3))
                * (smooth_max(v_ratio - v3) - smooth_max(v_ratio - v4))
            )
        else:
            q_guess = self.raw_q_var
        return q_guess

    def assign_ipopt_vars(self, model):
        """Map feeder and internal TSBI variables into this device."""
        self._model = model
        self.ipopt_vr = model.ipopt_vr_list[self.bus_idx]
        self.ipopt_vi = model.ipopt_vi_list[self.bus_idx]
        self.ipopt_ir = model.inv_ir_list[self.inv_idx]
        self.ipopt_ii = model.inv_ii_list[self.inv_idx]

        for attribute, variable_name in (
            ("v_pv", "inv_v_pv_list"),
            ("i_pv", "inv_i_pv_list"),
            ("i_dc", "inv_i_dc_list"),
            ("duty_cycle", "inv_duty_cycle_list"),
            ("magnitude_modulation", "inv_modulation_list"),
            ("modulation_r", "inv_modulation_r_list"),
            ("modulation_i", "inv_modulation_i_list"),
            ("v_ac_r", "inv_v_ac_r_list"),
            ("v_ac_i", "inv_v_ac_i_list"),
            ("i_ac_r", "inv_i_ac_r_list"),
            ("i_ac_i", "inv_i_ac_i_list"),
            ("i_t2_r", "inv_i_t2_r_list"),
            ("i_t2_i", "inv_i_t2_i_list"),
        ):
            setattr(self, attribute, getattr(model, variable_name)[self.inv_idx])

        if not hasattr(model, "_dx_inverter_constraints"):
            model._dx_inverter_constraints = ConstraintList()
        self._constraint_list = model._dx_inverter_constraints
        self._constraint_expressions = {}

    def _smooth_abs(self, expression):
        """Return |expression|, rounded near zero so it stays differentiable."""
        return sqrt(expression * expression + self.smoothing_epsilon)

    def _add_constraint(self, name: str, residual):
        """Record one residual and require it to vanish."""
        self._constraint_expressions[name] = residual
        self._constraint_list.add(residual == 0.0)

    def _build_terminal_constraints(self):
        """Build the complete PV-to-grid TSBI equation set."""
        if self._constraints_built:
            return
        if self._model is None:
            raise RuntimeError(
                f"Inverter {self.name}: assign_ipopt_vars() must be called first."
            )

        voltage_scale = max(self.dc_link_voltage_v, self.ac_terminal_voltage_v)
        power_scale = self.rating_va
        current_scale = self.rating_va / self.ac_terminal_voltage_v

        if self.pv_sdm is None:
            # CONSTANT_P is an explicitly commanded active-power mode.  It
            # does not claim that a PV source is being operated at its MPP.
            self._add_constraint(
                "constant_p_pv_voltage",
                (self.v_pv - self.constant_p_pv_voltage_v) / voltage_scale,
            )
        else:
            # The single-diode source equation makes I_pv and V_pv
            # optimization variables; their product is the source power
            # selected by the converter.
            sdm_module = self.pv_sdm["module"]
            series = float(self.pv_sdm["array"]["series_modules"])
            parallel = float(self.pv_sdm["array"]["parallel_strings"])
            i_ph = float(sdm_module["photocurrent_a"]) * parallel
            i_0 = float(sdm_module["saturation_current_a"]) * parallel
            r_s = float(sdm_module["series_resistance_ohm"]) * series / parallel
            r_sh = float(sdm_module["shunt_resistance_ohm"]) * series / parallel
            ideality_thermal_voltage = (
                float(sdm_module["ideality_thermal_voltage_v"]) * series
            )
            diode_argument = (self.v_pv + self.i_pv * r_s) / ideality_thermal_voltage
            # Use one implicit zero-residual equation. Putting ``-I_pv`` on
            # the right-hand side and equating that expression to ``I_pv``
            # would count the current twice and produce a false PV curve.
            # The residual is
            #
            # I_ph - I_pv - I_0*(exp((V_pv + I_pv*R_s)/V_th) - 1)
            #       - (V_pv + I_pv*R_s)/R_sh = 0.
            pv_current_residual = (
                i_ph
                - self.i_pv
                - i_0 * (exp(diode_argument) - 1.0)
                - (self.v_pv + self.i_pv * r_s) / r_sh
            )
            self._add_constraint(
                "pv_single_diode_current",
                pv_current_residual / max(self.pv_current_max_a, 1.0),
            )
            # Keep the calculated MPP visible as an explicit source limit.  It
            # is redundant with a valid SDM curve, but catches bad parameter
            # scaling and prevents numerical excursions above the curve.
            self.pv_mpp_constraint = self._constraint_list.add(
                self.v_pv * self.i_pv <= float(self.pv_mpp["p_mpp_w"])
            )

        # Non-ideal FSC, paper (4a)-(7a), represented by the controlled
        # voltage/current sources in Fig. 4 around the ideal transformer.
        abs_i_pv = self._smooth_abs(self.i_pv)
        abs_i_dc = self._smooth_abs(self.i_dc)
        sign_i_pv = self.i_pv / abs_i_pv
        sign_i_dc = self.i_dc / abs_i_dc
        fsc_series_resistance = (
            2.0 * self.mosfet_resistance_ohm + self.fsc_inductor_resistance_ohm
        )
        v_conduction_pv = self.duty_cycle * (
            2.0 * sign_i_pv * self.mosfet_threshold_v
            + self.i_pv * fsc_series_resistance
        )
        v_conduction_dc = (1.0 - self.duty_cycle) * (
            2.0 * sign_i_dc * self.mosfet_threshold_v
            + self.i_dc * fsc_series_resistance
        )
        fsc_switching_factor = self.fsc_switching_frequency_hz * (
            self.turn_on_time_s + self.turn_off_time_s
        )
        i_switching_pv = fsc_switching_factor * abs_i_pv
        i_switching_dc = fsc_switching_factor * abs_i_dc
        v_pv_effective = self.v_pv - v_conduction_pv
        v_dc_effective = self.dc_link_voltage_v + v_conduction_dc
        i_pv_effective = self.i_pv - i_switching_pv
        i_dc_effective = self.i_dc + i_switching_dc
        self._add_constraint(
            "fsc_voltage",
            (
                (1.0 - self.duty_cycle) * v_dc_effective
                - self.duty_cycle * v_pv_effective
            )
            / voltage_scale,
        )
        self._add_constraint(
            "fsc_power",
            (v_pv_effective * i_pv_effective - v_dc_effective * i_dc_effective)
            / power_scale,
        )

        # Non-ideal SSC, paper (8a)-(10o).  Equation (9g) is used directly
        # instead of (9i), avoiding division by M*cos(phi) near zero power.
        i_ac_magnitude_sq = self.i_ac_r**2 + self.i_ac_i**2
        i_ac_magnitude = sqrt(i_ac_magnitude_sq + self.smoothing_epsilon)
        m_cos_phi = (
            self.modulation_r * self.i_ac_r + self.modulation_i * self.i_ac_i
        ) / i_ac_magnitude
        abs_m_cos_phi = self._smooth_abs(m_cos_phi)
        transistor_avg_current = (
            sqrt(2.0)
            * i_ac_magnitude
            * (math.pi * abs_m_cos_phi + 4.0)
            / (8.0 * math.pi)
        )
        diode_avg_current = (
            sqrt(2.0)
            * i_ac_magnitude
            * (4.0 - math.pi * abs_m_cos_phi)
            / (8.0 * math.pi)
        )
        transistor_rms_sq = (
            i_ac_magnitude_sq * (8.0 * abs_m_cos_phi + 3.0 * math.pi) / (12.0 * math.pi)
        )
        diode_rms_sq = (
            i_ac_magnitude_sq * (3.0 * math.pi - 8.0 * abs_m_cos_phi) / (12.0 * math.pi)
        )
        conduction_loss_w = 4.0 * (
            transistor_avg_current * self.mosfet_threshold_v
            + transistor_rms_sq * self.mosfet_resistance_ohm
            + diode_avg_current * self.diode_forward_v
            + diode_rms_sq * self.diode_resistance_ohm
        )
        conduction_denominator = i_ac_magnitude_sq + self.smoothing_epsilon
        v_conduction_ac_r = conduction_loss_w * self.i_ac_r / conduction_denominator
        v_conduction_ac_i = conduction_loss_w * self.i_ac_i / conduction_denominator
        ssc_switching_factor = (
            2.0
            * sqrt(2.0)
            / math.pi
            * self.ssc_switching_frequency_hz
            * (self.turn_on_time_s + self.turn_off_time_s + self.diode_recovery_time_s)
        )
        i_switching_ssc = ssc_switching_factor * i_ac_magnitude
        v_ac_hat_r = self.v_ac_r + v_conduction_ac_r
        v_ac_hat_i = self.v_ac_i + v_conduction_ac_i
        i_dc_hat = self.i_dc - i_switching_ssc
        self._add_constraint(
            "ssc_voltage_real",
            (v_ac_hat_r - self.modulation_r * self.dc_link_voltage_v / sqrt(2.0))
            / voltage_scale,
        )
        self._add_constraint(
            "ssc_voltage_imag",
            (v_ac_hat_i - self.modulation_i * self.dc_link_voltage_v / sqrt(2.0))
            / voltage_scale,
        )
        self._add_constraint(
            "ssc_power",
            (
                self.dc_link_voltage_v * i_dc_hat
                - v_ac_hat_r * self.i_ac_r
                - v_ac_hat_i * self.i_ac_i
            )
            / power_scale,
        )
        self._add_constraint(
            "ssc_modulation",
            self.magnitude_modulation**2 - self.modulation_r**2 - self.modulation_i**2,
        )

        # Series-Rd LCL filter from Fig. 7.  The old clone treated Rd and C as
        # parallel elements; the paper explicitly places them in series.
        x1 = self.omega_rad_s * self.lcl_l1_h
        x2 = self.omega_rad_s * self.lcl_l2_h
        xc = 1.0 / (self.omega_rad_s * self.lcl_capacitance_f)
        damping_denominator = self.lcl_damping_resistance_ohm**2 + xc**2
        damping_conductance = self.lcl_damping_resistance_ohm / damping_denominator
        damping_susceptance = xc / damping_denominator
        capacitor_node_vr = (
            self.v_ac_r - self.lcl_r1_ohm * self.i_ac_r + x1 * self.i_ac_i
        )
        capacitor_node_vi = (
            self.v_ac_i - self.lcl_r1_ohm * self.i_ac_i - x1 * self.i_ac_r
        )
        damping_branch_ir = (
            damping_conductance * capacitor_node_vr
            - damping_susceptance * capacitor_node_vi
        )
        damping_branch_ii = (
            damping_susceptance * capacitor_node_vr
            + damping_conductance * capacitor_node_vi
        )
        v_t2_r = self.ipopt_vr * self.bus.v_base_v / self.interface_turns_ratio
        v_t2_i = self.ipopt_vi * self.bus.v_base_v / self.interface_turns_ratio
        self._add_constraint(
            "lcl_kcl_real",
            (self.i_t2_r - self.i_ac_r + damping_branch_ir) / current_scale,
        )
        self._add_constraint(
            "lcl_kcl_imag",
            (self.i_t2_i - self.i_ac_i + damping_branch_ii) / current_scale,
        )
        self._add_constraint(
            "lcl_kvl_real",
            (
                self.v_ac_r
                - v_t2_r
                - self.lcl_r1_ohm * self.i_ac_r
                - self.lcl_r2_ohm * self.i_t2_r
                + x1 * self.i_ac_i
                + x2 * self.i_t2_i
            )
            / self.ac_terminal_voltage_v,
        )
        self._add_constraint(
            "lcl_kvl_imag",
            (
                self.v_ac_i
                - v_t2_i
                - self.lcl_r1_ohm * self.i_ac_i
                - self.lcl_r2_ohm * self.i_t2_i
                - x1 * self.i_ac_r
                - x2 * self.i_t2_r
            )
            / self.ac_terminal_voltage_v,
        )

        # Grid control source, equivalent to paper (11a) without introducing a
        # voltage-denominator singularity. The caller supplies the active-power
        # target. For MPPT, the source-side SDM equation above determines the
        # maximum available PV power; curtailment moves the operating point
        # below that maximum when the grid requires it.
        # The reactive side follows paper Section IV-B: UPF / CONSTANT_Q (12a),
        # CPF (12b: Q*PF = P*sqrt(1-PF^2)), or the smoothed Volt-VAR droop
        # (12e) using the paper's f(x;a,b) smooth indicator.
        p_ctrl = v_t2_r * self.i_t2_r + v_t2_i * self.i_t2_i
        q_actual = v_t2_i * self.i_t2_r - v_t2_r * self.i_t2_i
        # The active-power target is the caller-set p_target (the dispatch,
        # or the curtailment engine's delivered-power set point).
        self._add_constraint(
            "control_active_power",
            (p_ctrl - self.p_target) / power_scale,
        )
        # Nameplate capability applies to every reactive control mode. Without
        # this circle, the 0.90 pu active request and a Volt-VAR command can
        # exceed the inverter rating near the droop extremes.
        self.apparent_power_constraint = self._constraint_list.add(
            (p_ctrl * p_ctrl + q_actual * q_actual) / (self.rating_va**2) <= 1.0
        )
        if self.control_mode == "CPF":
            pf = self.power_factor
            if pf is None:
                raise RuntimeError(
                    f"Inverter {self.name}: CPF mode requires power_factor."
                )
            self._add_constraint(
                "control_reactive_power",
                (q_actual * pf - p_ctrl * sqrt(1.0 - pf * pf)) / power_scale,
            )
        elif self.control_mode == "VOLT_VAR":
            v1, v2, v3, v4 = self.volt_var_knees
            q_bar = self.volt_var_q_max_pu * self.rating_va
            q_under = self.volt_var_q_min_pu * self.rating_va
            v_t2_mag_pu = (
                sqrt(v_t2_r * v_t2_r + v_t2_i * v_t2_i) / self.ac_terminal_voltage_v
            )
            v_ratio = v_t2_mag_pu / self.volt_var_v_ref_pu

            def smooth_max(x):
                return (x + sqrt(x * x + self.volt_var_smoothing_epsilon**2)) / 2.0

            # Paper (12e), applied to V/V_ref: Q_bar below V1, ramp to 0 over
            # [V1, V2], deadband [V2, V3], ramp to Q_under over [V3, V4],
            # flat at V4 and above.
            q_ctrl = (
                q_bar
                - (q_bar / (v2 - v1))
                * (smooth_max(v_ratio - v1) - smooth_max(v_ratio - v2))
                + (q_under / (v4 - v3))
                * (smooth_max(v_ratio - v3) - smooth_max(v_ratio - v4))
            )
            self._add_constraint(
                "control_reactive_power",
                (q_actual - q_ctrl) / power_scale,
            )
        else:  # UPF, CONSTANT_Q, MPPT: fixed reactive setpoint (0 for UPF/MPPT).
            q_setpoint = 0.0 if self.control_mode in ("UPF", "MPPT") else self.raw_q_var
            self._add_constraint(
                "control_reactive_power",
                (q_actual - q_setpoint) / power_scale,
            )

        # Lossless interface transformer between the paper's low-voltage T2
        # terminal and the selected feeder phase terminal.
        self._add_constraint(
            "interface_current_real",
            self.ipopt_ir
            - self.i_t2_r / (self.interface_turns_ratio * self.bus.i_base_a),
        )
        self._add_constraint(
            "interface_current_imag",
            self.ipopt_ii
            - self.i_t2_i / (self.interface_turns_ratio * self.bus.i_base_a),
        )

        self._constraints_built = True

    def add_to_eqn_list(self, kcl_real, kcl_imag):
        """Add the transformed T2 current injection to feeder KCL."""
        self._build_terminal_constraints()
        kcl_real[self.bus_idx] = kcl_real.get(self.bus_idx, 0) - self.ipopt_ir
        kcl_imag[self.bus_idx] = kcl_imag.get(self.bus_idx, 0) - self.ipopt_ii

    def max_constraint_residual(self) -> float:
        """Return the largest scaled TSBI equality residual after a solve."""
        if not self._constraint_expressions:
            return math.nan
        return max(
            abs(float(value(expression)))
            for expression in self._constraint_expressions.values()
        )
