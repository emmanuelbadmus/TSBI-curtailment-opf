"""Single-diode photovoltaic source component and operating-point helpers."""

from __future__ import annotations

import copy
import math
from functools import lru_cache


class PVSpecError(ValueError):
    """Raised when a JSON PV source definition is incomplete or invalid."""


def _number(block: dict, key: str, *, positive: bool = False) -> float:
    """Read one finite number from a block, or say which key was wrong."""
    try:
        result = float(block[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise PVSpecError(f"PV source requires numeric {key!r}") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        requirement = "positive and finite" if positive else "finite"
        raise PVSpecError(f"PV source {key!r} must be {requirement}")
    return result


def validate_sdm(raw: dict) -> dict:
    """Validate the aggregate single-diode data carried by a JSON PV source.

    The source data are module-level parameters.  The array is formed from
    ``series_modules`` modules in each of ``parallel_strings`` strings:
    series strings increase voltage, while parallel strings increase current
    and scale the equivalent resistances.
    """
    if not isinstance(raw, dict):
        raise PVSpecError("PV source must be an object")
    sdm = copy.deepcopy(raw)
    forbidden = {"model", "source", "note", "status", "active_equations"}
    present = sorted(forbidden.intersection(sdm))
    if present:
        raise PVSpecError(
            "PV source contains redundant or non-reproducible metadata; remove: "
            + ", ".join(present)
        )

    array = sdm.get("array")
    if not isinstance(array, dict):
        raise PVSpecError("PV source requires an array object")
    for key in ("series_modules", "parallel_strings"):
        value = _number(array, key, positive=True)
        if abs(value - round(value)) > 1e-12:
            raise PVSpecError(f"PV source array.{key} must be an integer")

    module = sdm.get("module")
    if not isinstance(module, dict):
        raise PVSpecError("PV source requires module parameters")
    for key in (
        "v_oc_v",
        "i_sc_a",
        "v_mp_v",
        "i_mp_a",
        "photocurrent_a",
        "saturation_current_a",
        "series_resistance_ohm",
        "shunt_resistance_ohm",
        "ideality_thermal_voltage_v",
    ):
        _number(module, key, positive=True)
    if not 0.0 < float(module["v_mp_v"]) < float(module["v_oc_v"]):
        raise PVSpecError("PV source requires 0 < module.v_mp_v < module.v_oc_v")
    if not 0.0 < float(module["i_mp_a"]) < float(module["i_sc_a"]):
        raise PVSpecError("PV source requires 0 < module.i_mp_a < module.i_sc_a")
    return sdm


def _array_parameters(sdm: dict) -> dict[str, float]:
    """Return equivalent one-diode parameters for the configured array."""
    module = sdm["module"]
    series = float(sdm["array"]["series_modules"])
    parallel = float(sdm["array"]["parallel_strings"])
    return {
        "i_ph_a": float(module["photocurrent_a"]) * parallel,
        "i_0_a": float(module["saturation_current_a"]) * parallel,
        "r_s_ohm": float(module["series_resistance_ohm"]) * series / parallel,
        "r_sh_ohm": float(module["shunt_resistance_ohm"]) * series / parallel,
        "ideality_thermal_voltage_v": (
            float(module["ideality_thermal_voltage_v"]) * series
        ),
        "v_oc_v": float(module["v_oc_v"]) * series,
        "i_sc_a": float(module["i_sc_a"]) * parallel,
        "v_mp_v": float(module["v_mp_v"]) * series,
        "i_mp_a": float(module["i_mp_a"]) * parallel,
    }


def current_at_voltage(voltage_v: float, pv: dict) -> float:
    """Solve the implicit single-diode current at an array voltage."""
    pv = validate_sdm(pv)
    parameters = _array_parameters(pv)
    voltage = float(voltage_v)
    if not math.isfinite(voltage):
        raise PVSpecError("PV source voltage must be finite")
    voltage = min(max(voltage, 0.0), parameters["v_oc_v"])

    def residual(current: float) -> float:
        argument = (voltage + current * parameters["r_s_ohm"]) / parameters[
            "ideality_thermal_voltage_v"
        ]
        diode = parameters["i_0_a"] * math.expm1(min(argument, 700.0))
        shunt = (voltage + current * parameters["r_s_ohm"]) / parameters["r_sh_ohm"]
        return parameters["i_ph_a"] - current - diode - shunt

    lower, upper = 0.0, parameters["i_sc_a"]
    if residual(lower) <= 0.0:
        return 0.0
    if residual(upper) >= 0.0:
        return upper
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _cache_key(pv: dict) -> tuple[float, ...]:
    """Return immutable PV inputs that determine the MPP."""
    pv = validate_sdm(pv)
    array = pv["array"]
    module = pv["module"]
    return (
        float(array["series_modules"]),
        float(array["parallel_strings"]),
        *(
            float(module[key])
            for key in (
                "v_oc_v",
                "i_sc_a",
                "v_mp_v",
                "i_mp_a",
                "photocurrent_a",
                "saturation_current_a",
                "series_resistance_ohm",
                "shunt_resistance_ohm",
                "ideality_thermal_voltage_v",
            )
        ),
    )


def _from_cache_key(key: tuple[float, ...]) -> dict:
    """Reconstruct the validated PV object used by the cached MPP search."""
    return {
        "array": {
            "series_modules": key[0],
            "parallel_strings": key[1],
        },
        "module": dict(
            zip(
                (
                    "v_oc_v",
                    "i_sc_a",
                    "v_mp_v",
                    "i_mp_a",
                    "photocurrent_a",
                    "saturation_current_a",
                    "series_resistance_ohm",
                    "shunt_resistance_ohm",
                    "ideality_thermal_voltage_v",
                ),
                key[2:],
                strict=True,
            )
        ),
    }


@lru_cache(maxsize=32)
def _calculate_mpp_cached(key: tuple[float, ...]) -> dict[str, float]:
    """Calculate one shared PV MPP for one distinct JSON source definition."""
    pv = _from_cache_key(key)
    parameters = _array_parameters(pv)
    voltage_step = parameters["v_oc_v"] / 256.0
    voltages = [index * voltage_step for index in range(257)]
    powers = [voltage * current_at_voltage(voltage, pv) for voltage in voltages]
    best_index = max(range(len(powers)), key=powers.__getitem__)
    left = voltages[max(best_index - 1, 0)]
    right = voltages[min(best_index + 1, len(voltages) - 1)]
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def power(voltage: float) -> float:
        return voltage * current_at_voltage(voltage, pv)

    x1 = right - golden_ratio * (right - left)
    x2 = left + golden_ratio * (right - left)
    f1, f2 = power(x1), power(x2)
    for _ in range(80):
        if f1 < f2:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden_ratio * (right - left)
            f2 = power(x2)
        else:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden_ratio * (right - left)
            f1 = power(x1)
    candidates = [
        (voltage, power(voltage)) for voltage in (x1, x2, parameters["v_mp_v"])
    ]
    voltage, _ = max(candidates, key=lambda item: item[1])
    current = current_at_voltage(voltage, pv)
    return {
        "v_mpp_v": voltage,
        "i_mpp_a": current,
        "p_mpp_w": voltage * current,
        "v_oc_v": parameters["v_oc_v"],
        "i_sc_a": parameters["i_sc_a"],
    }


def calculate_mpp(pv: dict) -> dict[str, float]:
    """Calculate the PV operating point that maximizes ``V_pv * I_pv``."""
    return dict(_calculate_mpp_cached(_cache_key(pv)))


def operating_point_for_power(power_w: float, pv: dict) -> dict[str, float]:
    """Return the high-voltage SDM point for a requested source power.

    The descending branch from ``V_MPP`` to ``V_OC`` is used for warm starts.
    The requested power is clipped at ``P_MPP`` because no point on the source
    curve can supply more.
    """
    pv = validate_sdm(pv)
    mpp = calculate_mpp(pv)
    target = float(power_w)
    if not math.isfinite(target) or target < 0.0:
        raise PVSpecError("PV warm-start power must be finite and nonnegative")
    target = min(target, mpp["p_mpp_w"])
    if target <= 0.0:
        return {
            "v_pv_v": mpp["v_oc_v"],
            "i_pv_a": 0.0,
            "p_pv_w": 0.0,
        }

    left, right = mpp["v_mpp_v"], mpp["v_oc_v"]
    for _ in range(100):
        midpoint = 0.5 * (left + right)
        midpoint_power = midpoint * current_at_voltage(midpoint, pv)
        if midpoint_power > target:
            left = midpoint
        else:
            right = midpoint
    voltage = 0.5 * (left + right)
    current = current_at_voltage(voltage, pv)
    return {
        "v_pv_v": voltage,
        "i_pv_a": current,
        "p_pv_w": voltage * current,
    }
