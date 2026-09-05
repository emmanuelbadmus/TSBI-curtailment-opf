"""Constant power, current, and impedance load component in internal p.u."""

import math
from enum import IntEnum
from itertools import count

from pyomo.common.errors import PyomoException
from pyomo.core.expr.numvalue import is_potentially_variable
from pyomo.environ import ConstraintList, Expr_if, value

from src import network_simulator as per_unit

from .bus import Bus

_LOAD_VOLTAGE_MODEL_FACTOR_MIN = 1e-12
_NUMERIC_EPS = 1e-12
_VOLTAGE_MAG_EPS = 1e-24


class LoadModel(IntEnum):
    """
    Standard Load Models (OpenDSS conventions):
    1: Constant P + jQ (standard, default).
    2: Constant impedance (Z).
    3: Constant P, quadratic Q (motor-like).
    4: Linear P, quadratic Q (feeder/CVR mix).
    5: Constant current magnitude (I).
    6: Constant P, fixed Q.
    7: Constant P, fixed reactance X.
    8: ZIPV (7-coefficient polynomial).
    9: OpenDSS Generator Model=7, constant power with low-voltage current limit.
    """

    CONSTANT_PQ = 1
    CONSTANT_Z = 2
    CONSTANT_P_QUAD_Q = 3
    LINEAR_P_QUAD_Q = 4
    CONSTANT_I = 5
    CONSTANT_P_FIXED_Q = 6
    CONSTANT_P_FIXED_X = 7
    ZIPV = 8
    OPENDSS_GENERATOR_CURRENT_LIMITED = 9


class Load:
    """One phase of a load, drawing current set by its voltage-dependent model."""

    _ids = count(0)

    def __init__(
        self,
        name,
        from_bus: Bus,
        to_bus: Bus,
        P,
        Q,
        I_const=0j,
        Z_const=0j,
        I_mag=0.0,
        I_angle=0.0,
        load_num=None,
        phase=None,
        triplex_phase=None,
        nominal_v=0.0,
        vmin_pu=None,
        vmax_pu=None,
        vlow_pu=None,
        load_model=1,
        cvr_watts=1.0,
        cvr_vars=2.0,
        zipv_coeffs=None,
        use_opendss_voltage_limits=False,
        current_follows_voltage_angle=True,
        current_reference_angle=None,
    ):
        self.id = next(self._ids)
        self.name = name
        self.phase = phase
        self.triplex_phase = triplex_phase
        self.load_num = load_num or str(self.id)

        self.from_bus = from_bus
        self.to_bus = to_bus
        self.from_bus_idx = from_bus.int_bus_id
        self.to_bus_idx = (
            to_bus.int_bus_id
            if to_bus is not None and hasattr(to_bus, "int_bus_id")
            else None
        )

        # raw data
        self.raw_p_w = float(P)
        self.raw_q_var = float(Q)
        self.raw_i_const_a = complex(I_const)
        self.raw_z_const_ohm = complex(Z_const)
        self.raw_i_mag_a = float(I_mag)
        self.raw_i_angle_rad = float(I_angle)

        self.nominal_v = float(nominal_v or 0.0)
        self.vmin_pu = None if vmin_pu is None else float(vmin_pu)
        self.vmax_pu = None if vmax_pu is None else float(vmax_pu)
        self.vlow_pu = None if vlow_pu is None else float(vlow_pu)

        self.load_model = (
            LoadModel(int(load_model or 1))
            if int(load_model or 1) in LoadModel._value2member_map_
            else int(load_model or 1)
        )
        self.cvr_watts = float(cvr_watts)
        self.cvr_vars = float(cvr_vars)
        self.zipv_coeffs = tuple(float(v) for v in (zipv_coeffs or ()))
        self.use_opendss_voltage_limits = bool(use_opendss_voltage_limits)
        self.current_follows_voltage_angle = bool(current_follows_voltage_angle)

        # branch voltage base
        self.from_v_base = from_bus.v_base_v
        self.to_v_base = (
            to_bus.v_base_v if self.to_bus_idx is not None else self.from_v_base
        )
        self.branch_v_base = per_unit.resolve_voltage_base(
            self.nominal_v if self.nominal_v > 0.0 else self.from_v_base,
            fallback_v=self.from_v_base,
        )

        self.v_from_scale = per_unit.voltage_scale(self.from_v_base, self.branch_v_base)
        self.v_to_scale = (
            per_unit.voltage_scale(self.to_v_base, self.branch_v_base)
            if self.to_bus_idx is not None
            else 0.0
        )
        self.i_from_scale = self.v_from_scale
        self.i_to_scale = self.v_to_scale if self.to_bus_idx is not None else 0.0
        self.nominal_branch_angle = self._branch_voltage_angle_from_bus_state(
            float(getattr(self.from_bus, "Vr_pu", 0.0)),
            float(getattr(self.from_bus, "Vi_pu", 0.0)),
            float(getattr(self.to_bus, "Vr_pu", 0.0))
            if self.to_bus is not None
            else 0.0,
            float(getattr(self.to_bus, "Vi_pu", 0.0))
            if self.to_bus is not None
            else 0.0,
        )
        if current_reference_angle is not None:
            self.nominal_branch_angle = float(current_reference_angle)
        self.nominal_branch_cos = math.cos(self.nominal_branch_angle)
        self.nominal_branch_sin = math.sin(self.nominal_branch_angle)

        # p.u. data
        self.P_pu = per_unit.power_to_pu(
            self.raw_p_w, s_base_va=self.from_bus.s_base_va
        )
        self.Q_pu = per_unit.power_to_pu(
            self.raw_q_var, s_base_va=self.from_bus.s_base_va
        )
        self.I_const_pu = per_unit.complex_current_to_pu(
            self.raw_i_const_a, self.branch_v_base
        )
        self.I_mag_pu = per_unit.current_to_pu(self.raw_i_mag_a, self.branch_v_base)
        self.Z_const_pu = (
            per_unit.impedance_to_pu(self.raw_z_const_ohm, self.branch_v_base)
            if abs(self.raw_z_const_ohm) > 0.0
            else 0j
        )
        self.nominal_v_pu = (
            per_unit.voltage_to_pu(self.nominal_v, self.branch_v_base)
            if self.nominal_v > 0.0
            else 0.0
        )

        # precomputed trig for constant-current model (fixed after __init__)
        self._cos_i_angle = math.cos(-self.raw_i_angle_rad)
        self._sin_i_angle = math.sin(-self.raw_i_angle_rad)

        # solver references
        self.ipopt_vr_from = None
        self.ipopt_vi_from = None
        self.ipopt_vr_to = None
        self.ipopt_vi_to = None

        self.ipopt_ir = None
        self.ipopt_ii = None

        self._model = None
        # Operating point used to pick the CVR branch, and the expression the
        # solver evaluates to find the one the solution actually lands on.
        self.branch_vpu = None
        self._branch_vpu_expr = None
        self._constraint_list = None
        self._constraints_built = False

    # -------------------------------------------------------------------------
    # model attachment
    # -------------------------------------------------------------------------
    def assign_ipopt_vars(self, model):
        """Point this load at the voltage variables of its terminals."""
        self._model = model

        self.ipopt_vr_from = model.ipopt_vr_list[self.from_bus_idx]
        self.ipopt_vi_from = model.ipopt_vi_list[self.from_bus_idx]

        if self.to_bus_idx is not None:
            self.ipopt_vr_to = model.ipopt_vr_list[self.to_bus_idx]
            self.ipopt_vi_to = model.ipopt_vi_list[self.to_bus_idx]

        if not hasattr(model, "_dx_load_constraints"):
            model._dx_load_constraints = ConstraintList()
        self._constraint_list = model._dx_load_constraints

        self.ipopt_ir = model.load_ir_list[self.load_idx]
        self.ipopt_ii = model.load_ii_list[self.load_idx]
        self.initialize_current_from_voltage()

    # -------------------------------------------------------------------------
    # voltage helpers
    # -------------------------------------------------------------------------
    def _branch_voltage_pu(self):
        """Return the voltage across the load, on the from-bus base."""
        if self.to_bus_idx is not None:
            return (
                self.ipopt_vr_from * self.v_from_scale
                - self.ipopt_vr_to * self.v_to_scale,
                self.ipopt_vi_from * self.v_from_scale
                - self.ipopt_vi_to * self.v_to_scale,
            )

        if self.phase == "1":
            return (
                self.ipopt_vr_from * self.v_from_scale,
                self.ipopt_vi_from * self.v_from_scale,
            )

        same_orientation = bool(
            getattr(self.from_bus, "triplex_same_orientation", False)
        )
        if self._uses_reversed_triplex_leg() and not same_orientation:
            return (
                -self.ipopt_vr_from * self.v_from_scale,
                -self.ipopt_vi_from * self.v_from_scale,
            )

        return (
            self.ipopt_vr_from * self.v_from_scale,
            self.ipopt_vi_from * self.v_from_scale,
        )

    def _branch_voltage_angle_from_bus_state(
        self, vr_from, vi_from, vr_to=0.0, vi_to=0.0
    ):
        """Return the angle of the branch voltage from given terminal values."""
        if self.to_bus_idx is not None:
            vr = vr_from * self.v_from_scale - vr_to * self.v_to_scale
            vi = vi_from * self.v_from_scale - vi_to * self.v_to_scale
        elif self.phase == "1":
            vr = vr_from * self.v_from_scale
            vi = vi_from * self.v_from_scale
        elif self._uses_reversed_triplex_leg() and not bool(
            getattr(self.from_bus, "triplex_same_orientation", False)
        ):
            vr = -vr_from * self.v_from_scale
            vi = -vi_from * self.v_from_scale
        else:
            vr = vr_from * self.v_from_scale
            vi = vi_from * self.v_from_scale
        if abs(vr) <= _NUMERIC_EPS and abs(vi) <= _NUMERIC_EPS:
            return 0.0
        return math.atan2(vi, vr)

    def _uses_reversed_triplex_leg(self):
        """Say whether this is the reversed second leg of a triplex service."""
        return (
            self.to_bus_idx is None
            and self.phase == "2"
            and (
                bool(getattr(self.from_bus, "is_triplex", False))
                or bool(getattr(self.to_bus, "is_triplex", False))
                or self.triplex_phase is not None
            )
        )

    @staticmethod
    def _pq_current(vr, vi, p_val, q_val, denom):
        """Return the current drawing p_val and q_val at the given voltage."""
        ir = (p_val * vr + q_val * vi) / denom
        ii = (p_val * vi - q_val * vr) / denom
        return ir, ii

    def _opendss_limit_admittance_current(self, vr, vi, base_sq, limit_pu):
        """OpenDSS voltage-limit current using the nominal Yeq branch."""
        limit = max(float(limit_pu), _LOAD_VOLTAGE_MODEL_FACTOR_MIN)

        if self.load_model == LoadModel.CONSTANT_I:
            return self._pq_current(vr, vi, self.P_pu, self.Q_pu, base_sq * limit)

        if self.load_model == LoadModel.ZIPV and len(self.zipv_coeffs) >= 6:
            pz, pi, pp, qz, qi, qp = self.zipv_coeffs[:6]
            ir_z, ii_z = self._pq_current(
                vr, vi, self.P_pu * pz, self.Q_pu * qz, base_sq
            )
            ir_i, ii_i = self._pq_current(
                vr, vi, self.P_pu * pi, self.Q_pu * qi, base_sq * limit
            )
            ir_p, ii_p = self._pq_current(
                vr, vi, self.P_pu * pp, self.Q_pu * qp, base_sq * limit**2
            )
            return ir_z + ir_i + ir_p, ii_z + ii_i + ii_p

        if self.load_model in {
            LoadModel.CONSTANT_P_FIXED_Q,
            LoadModel.CONSTANT_P_FIXED_X,
        }:
            ir_p, ii_p = self._pq_current(vr, vi, self.P_pu, 0.0, base_sq * limit**2)
            ir_q, ii_q = self._pq_current(vr, vi, 0.0, self.Q_pu, base_sq)
            return ir_p + ir_q, ii_p + ii_q

        return self._pq_current(vr, vi, self.P_pu, self.Q_pu, base_sq * limit**2)

    def _opendss_generator_current_limited_current(self, vr, vi, base_sq, limit_pu):
        """OpenDSS Generator Model=7 current limit below Vminpu."""
        limit = max(float(limit_pu), _LOAD_VOLTAGE_MODEL_FACTOR_MIN)
        model_base = max(float(base_sq) ** 0.5, _LOAD_VOLTAGE_MODEL_FACTOR_MIN)
        vmag = (vr**2 + vi**2 + _VOLTAGE_MAG_EPS) ** 0.5
        denom = vmag * model_base * limit + _VOLTAGE_MAG_EPS
        return self._pq_current(vr, vi, self.P_pu, self.Q_pu, denom)

    def _opendss_transition_admittance_scale(self, vpu, constant_current_anchor=False):
        """Return the admittance interpolation OpenDSS uses between Vlow and Vmin."""
        vlow = float(self.vlow_pu)
        vmin = float(self.vmin_pu)
        if constant_current_anchor:
            upper_current = 1.0
        else:
            upper_current = 1.0 / vmin
        current_scale = vlow + ((upper_current - vlow) * (vpu - vlow) / (vmin - vlow))
        return current_scale / vpu

    def _opendss_low_voltage_current(self, vr, vi, base_sq, d, model2_current):
        """
        OpenDSS low-voltage load current for Vlowpu < V <= Vminpu.

        This mirrors Load.pas: most load models interpolate the equivalent
        admittance between Yeq at Vlowpu and Yeq95 at Vminpu. Constant-current
        pieces interpolate to the nominal current anchor used by Model=5.
        Fixed-Q models keep the reactive admittance fixed while limiting the
        real-power branch.
        """
        if self.vmin_pu is None or self.vmin_pu <= 0.0:
            return model2_current

        if self.load_model in {
            LoadModel.CONSTANT_P_FIXED_Q,
            LoadModel.CONSTANT_P_FIXED_X,
        }:
            return self._opendss_limit_admittance_current(vr, vi, base_sq, self.vmin_pu)

        if self.vlow_pu is None or self.vlow_pu <= 0.0 or self.vlow_pu >= self.vmin_pu:
            return self._opendss_limit_admittance_current(vr, vi, base_sq, self.vmin_pu)

        model_base = max(base_sq**0.5, _LOAD_VOLTAGE_MODEL_FACTOR_MIN)
        vmag = (d + _VOLTAGE_MAG_EPS) ** 0.5
        vpu = vmag / model_base

        if self.load_model == LoadModel.CONSTANT_I:
            admittance_scale = self._opendss_transition_admittance_scale(
                vpu, constant_current_anchor=True
            )
            return self._pq_current(
                vr,
                vi,
                self.P_pu * admittance_scale,
                self.Q_pu * admittance_scale,
                base_sq,
            )

        if self.load_model == LoadModel.ZIPV and len(self.zipv_coeffs) >= 6:
            pz, pi, pp, qz, qi, qp = self.zipv_coeffs[:6]
            pq_scale = self._opendss_transition_admittance_scale(
                vpu, constant_current_anchor=False
            )
            i_scale = self._opendss_transition_admittance_scale(
                vpu, constant_current_anchor=True
            )
            ir_z, ii_z = self._pq_current(
                vr, vi, self.P_pu * pz, self.Q_pu * qz, base_sq
            )
            ir_i, ii_i = self._pq_current(
                vr, vi, self.P_pu * pi * i_scale, self.Q_pu * qi * i_scale, base_sq
            )
            ir_p, ii_p = self._pq_current(
                vr, vi, self.P_pu * pp * pq_scale, self.Q_pu * qp * pq_scale, base_sq
            )
            return ir_z + ir_i + ir_p, ii_z + ii_i + ii_p

        admittance_scale = self._opendss_transition_admittance_scale(
            vpu, constant_current_anchor=False
        )
        return self._pq_current(
            vr,
            vi,
            self.P_pu * admittance_scale,
            self.Q_pu * admittance_scale,
            base_sq,
        )

    def _opendss_voltage_limited_current(
        self,
        vr,
        vi,
        base_sq,
        d,
        normal_current,
        model2_current,
    ):
        """Return the current for the voltage band this load is operating in."""
        model_base = max(base_sq**0.5, _LOAD_VOLTAGE_MODEL_FACTOR_MIN)
        vmag = (d + _VOLTAGE_MAG_EPS) ** 0.5
        vpu = vmag / model_base

        if self.load_model == LoadModel.OPENDSS_GENERATOR_CURRENT_LIMITED:
            if self.vmin_pu is None or self.vmin_pu <= 0.0:
                return normal_current
            limited_current = self._opendss_generator_current_limited_current(
                vr,
                vi,
                base_sq,
                self.vmin_pu,
            )
            if is_potentially_variable(vpu):
                return (
                    Expr_if(
                        IF=(vpu <= self.vmin_pu),
                        THEN=limited_current[0],
                        ELSE=normal_current[0],
                    ),
                    Expr_if(
                        IF=(vpu <= self.vmin_pu),
                        THEN=limited_current[1],
                        ELSE=normal_current[1],
                    ),
                )
            if value(vpu) <= self.vmin_pu:
                return limited_current
            return normal_current

        high_current = normal_current
        if self.vmax_pu is not None and self.vmax_pu > 0.0:
            high_current = self._opendss_limit_admittance_current(
                vr,
                vi,
                base_sq,
                self.vmax_pu,
            )

        very_low_current = model2_current
        low_current = normal_current
        if self.vmin_pu is not None and self.vmin_pu > 0.0:
            low_current = self._opendss_low_voltage_current(
                vr,
                vi,
                base_sq,
                d,
                model2_current,
            )

        if self.load_model == LoadModel.LINEAR_P_QUAD_Q:
            # CVR model: fix the branch once, rather than switching on a
            # variable, because the Expr_if discontinuity at Vmin destabilises
            # IPOPT.  The branch is taken at 'branch_vpu' when a caller has
            # supplied the operating point from an earlier solve, and at the
            # initialisation voltage otherwise; NetworkSimulator iterates the
            # two to a fixed point so the branch matches where the load ends up.
            # A small tolerance absorbs floating-point representation error when
            # the operating voltage is at exactly the Vmin boundary.
            _VMIN_FP_TOL = 2e-5
            self._branch_vpu_expr = vpu
            if self.branch_vpu is not None:
                vpu_val = float(self.branch_vpu)
            else:
                vpu_val = (
                    float(value(vpu)) if is_potentially_variable(vpu) else float(vpu)
                )
            if (
                self.vmax_pu is not None
                and self.vmax_pu > 0.0
                and vpu_val > self.vmax_pu
            ):
                return high_current
            if (
                self.vlow_pu is not None
                and self.vlow_pu > 0.0
                and vpu_val <= self.vlow_pu
            ):
                return very_low_current
            if (
                self.vmin_pu is not None
                and self.vmin_pu > 0.0
                and vpu_val < self.vmin_pu - _VMIN_FP_TOL
            ):
                return low_current
            return normal_current

        if is_potentially_variable(vpu):
            ir, ii = normal_current
            if self.vmin_pu is not None and self.vmin_pu > 0.0:
                ir = Expr_if(IF=(vpu <= self.vmin_pu), THEN=low_current[0], ELSE=ir)
                ii = Expr_if(IF=(vpu <= self.vmin_pu), THEN=low_current[1], ELSE=ii)
            if self.vlow_pu is not None and self.vlow_pu > 0.0:
                ir = Expr_if(
                    IF=(vpu <= self.vlow_pu), THEN=very_low_current[0], ELSE=ir
                )
                ii = Expr_if(
                    IF=(vpu <= self.vlow_pu), THEN=very_low_current[1], ELSE=ii
                )
            if self.vmax_pu is not None and self.vmax_pu > 0.0:
                ir = Expr_if(IF=(vpu > self.vmax_pu), THEN=high_current[0], ELSE=ir)
                ii = Expr_if(IF=(vpu > self.vmax_pu), THEN=high_current[1], ELSE=ii)
            return ir, ii

        vpu_value = float(value(vpu))
        if self.vmax_pu is not None and self.vmax_pu > 0.0 and vpu_value > self.vmax_pu:
            return high_current
        if (
            self.vlow_pu is not None
            and self.vlow_pu > 0.0
            and vpu_value <= self.vlow_pu
        ):
            return very_low_current
        if (
            self.vmin_pu is not None
            and self.vmin_pu > 0.0
            and vpu_value <= self.vmin_pu
        ):
            return low_current
        return normal_current

    def cvr_branch_label(self, vpu_value) -> str:
        """Name the voltage band a CVR load is operating in."""
        if vpu_value is None:
            return "normal"
        value_ = float(vpu_value)
        if self.vmax_pu is not None and self.vmax_pu > 0.0 and value_ > self.vmax_pu:
            return "high"
        if self.vlow_pu is not None and self.vlow_pu > 0.0 and value_ <= self.vlow_pu:
            return "very_low"
        if self.vmin_pu is not None and self.vmin_pu > 0.0 and value_ < self.vmin_pu:
            return "low"
        return "normal"

    def operating_vpu(self):
        """Return the per-unit terminal voltage of the current solution."""
        if self._branch_vpu_expr is None:
            return None
        try:
            return float(value(self._branch_vpu_expr))
        except (ValueError, TypeError):
            return None

    def _power_components_at_vpu(self, vpu, vpu_sq=None):
        """Return the real and reactive power the model draws at this voltage."""
        if vpu_sq is None:
            vpu_sq = vpu**2

        if self.load_model == LoadModel.CONSTANT_Z:
            return 0.0, 0.0
        if self.load_model == LoadModel.LINEAR_P_QUAD_Q:
            p_scale = vpu**self.cvr_watts
            q_scale = vpu**self.cvr_vars
            return self.P_pu * p_scale, self.Q_pu * q_scale
        if self.load_model == LoadModel.CONSTANT_I:
            return self.P_pu * vpu, self.Q_pu * vpu
        if self.load_model == LoadModel.CONSTANT_P_QUAD_Q:
            return self.P_pu, self.Q_pu * vpu_sq
        if self.load_model == LoadModel.CONSTANT_P_FIXED_X:
            return self.P_pu, self.Q_pu * vpu_sq
        if self.load_model == LoadModel.ZIPV and len(self.zipv_coeffs) >= 6:
            pz, pi, pp, qz, qi, qp = self.zipv_coeffs[:6]
            p_scale = pz * vpu_sq + pi * vpu + pp
            q_scale = qz * vpu_sq + qi * vpu + qp
            return self.P_pu * p_scale, self.Q_pu * q_scale
        return self.P_pu, self.Q_pu

    # -------------------------------------------------------------------------
    # device law
    # -------------------------------------------------------------------------
    def _pq_component_current(self, vr, vi):
        """
        Constant-power / CVR / OpenDSS model current on branch voltage base, in p.u.
        This preserves the existing semantics, but the result is now tied to explicit
        current variables through constraints instead of being injected directly in KCL.
        """
        ir_total = 0.0
        ii_total = 0.0

        if self.P_pu == 0.0 and self.Q_pu == 0.0:
            return ir_total, ii_total
        if self.load_model == LoadModel.CONSTANT_Z:
            return ir_total, ii_total

        d = vr**2 + vi**2 + _VOLTAGE_MAG_EPS

        if self.nominal_v_pu <= 0.0:
            ir, ii = self._pq_current(vr, vi, self.P_pu, self.Q_pu, d)
            return ir, ii

        base_sq = self.nominal_v_pu**2
        vpu_sq = d / base_sq
        vpu = vpu_sq**0.5

        # Model-2 style anchor: Constant Impedance logic for low-voltage fallbacks
        ir_model2, ii_model2 = self._pq_current(vr, vi, self.P_pu, self.Q_pu, base_sq)

        p_live, q_live = self._power_components_at_vpu(vpu, vpu_sq)
        ir_use, ii_use = self._pq_current(vr, vi, p_live, q_live, d)

        if self.use_opendss_voltage_limits:
            ir_use, ii_use = self._opendss_voltage_limited_current(
                vr,
                vi,
                base_sq,
                d,
                (ir_use, ii_use),
                (ir_model2, ii_model2),
            )

        ir_total += ir_use
        ii_total += ii_use
        return ir_total, ii_total

    def _const_current_component(self, vr, vi):
        """Return the current of the constant-current part of the load."""
        ir = 0.0
        ii = 0.0

        if abs(self.I_const_pu) > 0.0:
            if self.current_follows_voltage_angle:
                vmag = (vr**2 + vi**2 + _VOLTAGE_MAG_EPS) ** 0.5
                ur = vr / vmag
                ui = vi / vmag
                rot_r = ur * self.nominal_branch_cos + ui * self.nominal_branch_sin
                rot_i = ui * self.nominal_branch_cos - ur * self.nominal_branch_sin
                ir += self.I_const_pu.real * rot_r - self.I_const_pu.imag * rot_i
                ii += self.I_const_pu.real * rot_i + self.I_const_pu.imag * rot_r
            else:
                ir += self.I_const_pu.real
                ii += self.I_const_pu.imag

        if self.I_mag_pu > 0.0:
            vmag = (vr**2 + vi**2 + _VOLTAGE_MAG_EPS) ** 0.5
            ur = vr / vmag
            ui = vi / vmag
            ir += self.I_mag_pu * (ur * self._cos_i_angle - ui * self._sin_i_angle)
            ii += self.I_mag_pu * (ui * self._cos_i_angle + ur * self._sin_i_angle)

        return ir, ii

    def _const_impedance_component(self, vr, vi):
        """Return the current of the constant-impedance part of the load."""
        if abs(self.Z_const_pu) == 0.0:
            return 0.0, 0.0

        y_pu = 1.0 / self.Z_const_pu
        ir = vr * y_pu.real - vi * y_pu.imag
        ii = vi * y_pu.real + vr * y_pu.imag
        return ir, ii

    def _target_current_expr(self):
        """Return the total current this load should draw at its terminals."""
        vr, vi = self._branch_voltage_pu()

        ir_pq, ii_pq = self._pq_component_current(vr, vi)
        ir_i, ii_i = self._const_current_component(vr, vi)
        ir_z, ii_z = self._const_impedance_component(vr, vi)

        return (
            ir_pq + ir_i + ir_z,
            ii_pq + ii_i + ii_z,
        )

    def initialize_current_from_voltage(self):
        """Seed current variables from the same device law used by constraints."""
        if self.ipopt_ir is None or self.ipopt_ii is None:
            return
        try:
            ir, ii = self._target_current_expr()
            self.ipopt_ir.value = float(value(ir))
            self.ipopt_ii.value = float(value(ii))
        except (PyomoException, TypeError, ValueError):
            return

    # -------------------------------------------------------------------------
    # explicit current variables + constraints
    # -------------------------------------------------------------------------
    def _build_current_constraints(self):
        """Tie the load's current variables to its device law, once."""
        if self._constraints_built:
            return
        if self._model is None:
            raise RuntimeError(
                f"Load {self.name}: assign_ipopt_vars() must be called first."
            )

        ir_target, ii_target = self._target_current_expr()
        self._constraint_list.add(self.ipopt_ir == ir_target)
        self._constraint_list.add(self.ipopt_ii == ii_target)

        self._constraints_built = True

    def add_to_eqn_list(self, KCL_r, KCL_i):
        """Add this load's current to the KCL rows of its terminals."""
        self._build_current_constraints()

        sign = -1 if self._uses_reversed_triplex_leg() else 1

        KCL_r[self.from_bus_idx] = (
            KCL_r.get(self.from_bus_idx, 0) + sign * self.ipopt_ir * self.i_from_scale
        )
        KCL_i[self.from_bus_idx] = (
            KCL_i.get(self.from_bus_idx, 0) + sign * self.ipopt_ii * self.i_from_scale
        )

        if self.to_bus_idx is not None:
            KCL_r[self.to_bus_idx] = (
                KCL_r.get(self.to_bus_idx, 0) - self.ipopt_ir * self.i_to_scale
            )
            KCL_i[self.to_bus_idx] = (
                KCL_i.get(self.to_bus_idx, 0) - self.ipopt_ii * self.i_to_scale
            )
