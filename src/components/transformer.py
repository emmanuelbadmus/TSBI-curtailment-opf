"""Single-phase transformer model for common winding configurations.

The model uses eight constraints per transformer. Delta windings add two
delta-KVL equations.
"""

import math
from itertools import count

from pyomo.environ import value

from src import network_simulator as per_unit

_NUMERIC_EPS = 1e-12
_VOLTAGE_BASE_EPS = 1e-9
_ADMITTANCE_VALUE_EPS = 1e-18
_IMPEDANCE_SQ_EPS = 1e-18
_IMPEDANCE_VALUE_EPS = 1e-18
# Floors for a transformer whose series impedance is given as zero, which
# happens for ideal regulators and switch-like windings. Stated in per unit at
# the winding's own base rather than in ohms, so the same negligible fraction
# applies across the voltage levels a feeder contains.
_SERIES_RESISTANCE_FLOOR_PU = 1e-8
_SERIES_REACTANCE_FLOOR_PU = 1e-6


def _kcl(r, i, key, er, ei):
    """Add one real and imaginary current term to a bus's KCL rows."""
    r[key] = r.get(key, 0) + er
    i[key] = i.get(key, 0) + ei


class Transformer:
    """One phase of a transformer: a turns ratio, a series impedance, and the winding connection that decides which terminals it spans."""

    _ids = count(0)

    def __init__(
        self,
        name,
        from_bus_pos,
        from_bus_neg,
        to_bus_pos,
        to_bus_neg,
        r,
        x,
        status,
        tr,
        ang,
        G_shunt,
        B_shunt,
        rating,
        pri_conn="Y",
        sec_conn="Y",
        anti_float_b_pri=0.0,
        anti_float_b_sec=0.0,
        is_regulator=False,
        shunt_side="pri",
    ):
        self.id = next(self._ids) if not is_regulator else None
        self.name, self.status = name, status
        self.from_bus_pos, self.from_bus_neg = from_bus_pos, from_bus_neg
        self.to_bus_pos, self.to_bus_neg = to_bus_pos, to_bus_neg
        self.power_rating = rating
        self.pri_conn = pri_conn
        self.sec_conn = sec_conn
        self.shunt_side = str(shunt_side or "pri").strip().lower()
        self.anti_float_b_pri_raw = float(anti_float_b_pri)
        self.anti_float_b_sec_raw = float(anti_float_b_sec)

        self.v_base_pri = from_bus_pos.v_base_v
        self.v_base_sec = to_bus_pos.v_base_v
        self.i_base_pri = from_bus_pos.i_base_a
        self.i_base_sec = to_bus_pos.i_base_a

        self.tr_raw = float(tr)
        self.tr = self.tr_raw * per_unit.voltage_scale(self.v_base_sec, self.v_base_pri)
        self.ang_rad = ang * math.pi / 180
        self.cos_shift, self.sin_shift = math.cos(self.ang_rad), math.sin(self.ang_rad)
        self.r_raw_ohm = float(r)
        self.x_raw_ohm = float(x)
        z_sq = r**2 + x**2
        if z_sq < _IMPEDANCE_SQ_EPS:
            # Numerical safeguard for near-ideal regulators/switch-like
            # transformers, applied at this winding's own impedance base.
            z_base_ohm = per_unit.base_for_voltage(self.v_base_sec).z_base_ohm
            if abs(r) < _NUMERIC_EPS:
                r = _SERIES_RESISTANCE_FLOOR_PU * z_base_ohm
            if abs(x) < _NUMERIC_EPS:
                x = _SERIES_REACTANCE_FLOOR_PU * z_base_ohm
            z_sq = r**2 + x**2
        y_loss_raw = complex(r / z_sq, -x / z_sq)
        y_loss_pu = per_unit.admittance_to_pu(y_loss_raw, self.v_base_sec)
        self.G_loss, self.B_loss = y_loss_pu.real, y_loss_pu.imag

        y_shunt_raw = complex(G_shunt, B_shunt)
        y_sh_pri = per_unit.admittance_to_pu(y_shunt_raw, self.v_base_pri)
        y_sh_sec = per_unit.admittance_to_pu(y_shunt_raw, self.v_base_sec)
        self.G_shunt_pri, self.B_shunt_pri = y_sh_pri.real, y_sh_pri.imag
        self.G_shunt_sec, self.B_shunt_sec = y_sh_sec.real, y_sh_sec.imag

        self.anti_float_b_pri = per_unit.admittance_to_pu(
            1j * self.anti_float_b_pri_raw, self.v_base_pri
        ).imag
        self.anti_float_b_sec = per_unit.admittance_to_pu(
            1j * self.anti_float_b_sec_raw, self.v_base_sec
        ).imag

    def _pri_is_delta(self):
        """Say whether the primary winding is connected in delta."""
        return self.pri_conn == "D"

    def _sec_is_delta(self):
        """Say whether the secondary winding is connected in delta."""
        return self.sec_conn == "D"

    def _sec_is_zigzag(self):
        """Say whether the secondary winding is connected in zigzag."""
        return self.sec_conn == "Z"

    def is_delta_wye(self):
        """Say whether this transformer is delta primary, wye secondary."""
        return self._pri_is_delta() and not self._sec_is_delta()

    def is_wye_delta(self):
        """Say whether this transformer is wye primary, delta secondary."""
        return not self._pri_is_delta() and self._sec_is_delta()

    def is_delta_delta(self):
        """Say whether both windings are connected in delta."""
        return self._pri_is_delta() and self._sec_is_delta()

    def is_wye_zigzag(self):
        """Say whether this transformer is wye primary, zigzag secondary."""
        return (not self._pri_is_delta()) and self._sec_is_zigzag()

    def assign_xfmr_ipopt_vars(self, model):
        """Point this transformer at the voltage and current variables it uses."""
        fp, tp = self.from_bus_pos.int_bus_id, self.to_bus_pos.int_bus_id
        vr, vi = model.ipopt_vr_list, model.ipopt_vi_list
        self.vr_pos, self.vi_pos = vr[fp], vi[fp]
        if self.from_bus_neg is not None:
            fn = self.from_bus_neg.int_bus_id
            self.vr_neg, self.vi_neg = vr[fn], vi[fn]
        else:
            self.vr_neg, self.vi_neg = 0.0, 0.0
        self.vr_s_pos, self.vi_s_pos = vr[tp], vi[tp]
        if self.to_bus_neg is not None:
            tn = self.to_bus_neg.int_bus_id
            self.vr_s_neg, self.vi_s_neg = vr[tn], vi[tn]
        else:
            self.vr_s_neg, self.vi_s_neg = 0.0, 0.0
        self.vr_p, self.vi_p = self._primary_branch_voltage()
        self.vr_s, self.vi_s = self._secondary_branch_voltage()
        self.vr_sp, self.vi_sp = (
            model.xfmr_vr_aux_list[self.id],
            model.xfmr_vi_aux_list[self.id],
        )
        if self._pri_is_delta() or self._sec_is_delta():
            self.ir_pri = model.xfmr_ir_pri_list[self.id]
            self.ii_pri = model.xfmr_ii_pri_list[self.id]
            self.ir_sec = model.xfmr_ir_sec_list[self.id]
            self.ii_sec = model.xfmr_ii_sec_list[self.id]
        else:
            self.ir_from = model.xfmr_ir_pri_list[self.id]
            self.ii_from = model.xfmr_ii_pri_list[self.id]
            self.ir_to = model.xfmr_ir_sec_list[self.id]
            self.ii_to = model.xfmr_ii_sec_list[self.id]

    def _primary_branch_voltage(self):
        """Return the voltage across the primary winding."""
        return self.vr_pos - self.vr_neg, self.vi_pos - self.vi_neg

    def _secondary_branch_voltage(self):
        """Return the voltage across the secondary winding."""
        return self.vr_s_pos - self.vr_s_neg, self.vi_s_pos - self.vi_s_neg

    def initialize_ipopt_vars_from_voltage(self):
        """Seed auxiliary transformer variables from the present terminal voltages."""
        if not getattr(self, "requires_aux_vars", True):
            return

        vr_p = float(value(self.vr_p))
        vi_p = float(value(self.vi_p))
        vr_s = float(value(self.vr_s))
        vi_s = float(value(self.vi_s))
        ratio = complex(
            self.tr * self.cos_shift,
            self.tr * self.sin_shift,
        )
        if abs(ratio) <= _IMPEDANCE_VALUE_EPS:
            return

        v_sp = complex(vr_p, vi_p) / ratio
        self.vr_sp.value = float(v_sp.real)
        self.vi_sp.value = float(v_sp.imag)

        series_current = complex(self.G_loss, self.B_loss) * (
            complex(vr_s, vi_s) - v_sp
        )
        primary_series_current = (
            -series_current * complex(self.cos_shift, self.sin_shift) / self.tr
            if abs(self.tr) > _IMPEDANCE_VALUE_EPS
            else 0j
        )

        if self._pri_is_delta() or self._sec_is_delta():
            secondary_terminal_current = series_current
            if self.shunt_side == "sec":
                secondary_terminal_current += complex(
                    *self._shunt_current(vr_s, vi_s, side="sec")
                )
            self.ir_sec.value = float(secondary_terminal_current.real)
            self.ii_sec.value = float(secondary_terminal_current.imag)
            self.ir_pri.value = float(primary_series_current.real)
            self.ii_pri.value = float(primary_series_current.imag)
            return

        primary_terminal_current = primary_series_current
        secondary_terminal_current = series_current
        if self.shunt_side == "sec":
            secondary_terminal_current += complex(
                *self._shunt_current(vr_s, vi_s, side="sec")
            )
        else:
            primary_terminal_current += complex(
                *self._shunt_current(vr_p, vi_p, side="pri")
            )

        self.ir_from.value = float(primary_terminal_current.real)
        self.ii_from.value = float(primary_terminal_current.imag)
        self.ir_to.value = float(secondary_terminal_current.real)
        self.ii_to.value = float(secondary_terminal_current.imag)

    # --- Constraint equations ---
    def _voltage_ratio(self, vr_p, vi_p):
        """V_p - tr·V_s' = 0"""
        c, s, tr = self.cos_shift, self.sin_shift, self.tr
        return (
            vr_p - tr * (c * self.vr_sp - s * self.vi_sp),
            vi_p - tr * (c * self.vi_sp + s * self.vr_sp),
        )

    def _series_leakage(self, vr_s, vi_s, ir_s, ii_s):
        """I_s - Y_loss·(V_s - V_s') = 0"""
        G, B = self.G_loss, self.B_loss
        dvr, dvi = vr_s - self.vr_sp, vi_s - self.vi_sp
        return ir_s - (G * dvr - B * dvi), ii_s - (G * dvi + B * dvr)

    def _current_balance(self, ir_s, ii_s, ir_p, ii_p):
        """I_s + tr*·I_p = 0"""
        c, s, tr = self.cos_shift, self.sin_shift, self.tr
        return ir_s + tr * (c * ir_p + s * ii_p), ii_s + tr * (c * ii_p - s * ir_p)

    def _shunt_current(self, vr, vi, side="pri"):
        """Return the magnetising current drawn on one side."""
        if side == "sec":
            G, B = self.G_shunt_sec, self.B_shunt_sec
        else:
            G, B = self.G_shunt_pri, self.B_shunt_pri
        return G * vr - B * vi, G * vi + B * vr

    def _anti_float_current(self, vr, vi, susceptance):
        """Return the small susceptance current that keeps a delta from floating."""
        if abs(susceptance) < _ADMITTANCE_VALUE_EPS:
            return 0.0, 0.0
        return -susceptance * vi, susceptance * vr

    def _apply_delta_antifloat(
        self,
        KCL_r,
        KCL_i,
        pos_bus,
        neg_bus,
        vr_pos,
        vi_pos,
        vr_neg,
        vi_neg,
        susceptance_raw,
    ):
        """Add the anti-float current to the terminals of a delta winding."""
        b_pos = per_unit.admittance_to_pu(1j * susceptance_raw, pos_bus.v_base_v).imag
        b_neg = per_unit.admittance_to_pu(1j * susceptance_raw, neg_bus.v_base_v).imag
        ir_pos, ii_pos = self._anti_float_current(vr_pos, vi_pos, b_pos)
        ir_neg, ii_neg = self._anti_float_current(vr_neg, vi_neg, b_neg)
        _kcl(KCL_r, KCL_i, pos_bus.int_bus_id, ir_pos, ii_pos)
        _kcl(KCL_r, KCL_i, neg_bus.int_bus_id, ir_neg, ii_neg)

    def _inject_terminal_current(self, KCL_r, KCL_i, pos_bus, neg_bus, ir, ii, side):
        """Add a winding current to its positive and negative terminals."""
        _kcl(KCL_r, KCL_i, pos_bus.int_bus_id, ir, ii)
        if neg_bus is not None:
            scale = per_unit.current_scale(
                pos_bus.v_base_v,
                neg_bus.v_base_v,
            )
            _kcl(KCL_r, KCL_i, neg_bus.int_bus_id, -ir * scale, -ii * scale)

    def _calc_constraints_yy(self):
        """Return the winding voltages and currents of a wye-wye transformer."""
        vr_p, vi_p = self._primary_branch_voltage()
        vr_s, vi_s = self._secondary_branch_voltage()
        vr_pri, vi_pri = self._voltage_ratio(vr_p, vi_p)
        if self.shunt_side == "sec":
            ir_sh, ii_sh = self._shunt_current(vr_s, vi_s, side="sec")
            ir_series = self.ir_to - ir_sh
            ii_series = self.ii_to - ii_sh
            ir_sec, ii_sec = self._series_leakage(vr_s, vi_s, ir_series, ii_series)
            ir_aux, ii_aux = self._current_balance(
                ir_series, ii_series, self.ir_from, self.ii_from
            )
            return vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec

        ir_sec, ii_sec = self._series_leakage(vr_s, vi_s, self.ir_to, self.ii_to)
        ir_sh, ii_sh = self._shunt_current(vr_p, vi_p, side="pri")
        ir_p = self.ir_from - ir_sh
        ii_p = self.ii_from - ii_sh
        ir_aux, ii_aux = self._current_balance(self.ir_to, self.ii_to, ir_p, ii_p)
        return vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec

    def _calc_constraints_dyg(self):
        """Return the winding voltages and currents of a delta-grounded-wye."""
        vr_ll, vi_ll = self._primary_branch_voltage()
        vr_pri, vi_pri = self._voltage_ratio(vr_ll, vi_ll)
        vr_s, vi_s = self._secondary_branch_voltage()
        if self.shunt_side == "sec":
            ir_sh, ii_sh = self._shunt_current(vr_s, vi_s, side="sec")
            ir_series = self.ir_sec - ir_sh
            ii_series = self.ii_sec - ii_sh
            ir_sec, ii_sec = self._series_leakage(vr_s, vi_s, ir_series, ii_series)
            ir_aux, ii_aux = self._current_balance(
                ir_series, ii_series, self.ir_pri, self.ii_pri
            )
        else:
            ir_sec, ii_sec = self._series_leakage(vr_s, vi_s, self.ir_sec, self.ii_sec)
            ir_aux, ii_aux = self._current_balance(
                self.ir_sec, self.ii_sec, self.ir_pri, self.ii_pri
            )
        return vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec

    def _calc_constraints_ygd(self):
        """Return the winding voltages and currents of a grounded-wye-delta."""
        vr_ll, vi_ll = self._secondary_branch_voltage()
        vr_p, vi_p = self._primary_branch_voltage()
        vr_pri, vi_pri = self._voltage_ratio(vr_p, vi_p)
        ir_sec, ii_sec = self._series_leakage(vr_ll, vi_ll, self.ir_sec, self.ii_sec)
        ir_aux, ii_aux = self._current_balance(
            self.ir_sec, self.ii_sec, self.ir_pri, self.ii_pri
        )
        return vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec

    def _calc_constraints_dd(self):
        """Return the winding voltages and currents of a delta-delta."""
        vr_p_ll, vi_p_ll = self._primary_branch_voltage()
        vr_s_ll, vi_s_ll = self._secondary_branch_voltage()
        vr_pri, vi_pri = self._voltage_ratio(vr_p_ll, vi_p_ll)
        ir_sec, ii_sec = self._series_leakage(
            vr_s_ll, vi_s_ll, self.ir_sec, self.ii_sec
        )
        ir_aux, ii_aux = self._current_balance(
            self.ir_sec, self.ii_sec, self.ir_pri, self.ii_pri
        )
        return vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec

    # --- KCL injection ---
    def add_to_eqn_list(
        self,
        KCL_r,
        KCL_i,
        xfmr_ir_aux,
        xfmr_ii_aux,
        xfmr_vr_pri,
        xfmr_vi_pri,
        xfmr_ir_sec,
        xfmr_ii_sec,
        xfmr_vr_kvl=None,
        xfmr_vi_kvl=None,
    ):
        """Add this transformer's currents and its voltage law to the model."""
        if self.is_delta_wye():
            self._add_dyg_eqns(
                KCL_r,
                KCL_i,
                xfmr_ir_aux,
                xfmr_ii_aux,
                xfmr_vr_pri,
                xfmr_vi_pri,
                xfmr_ir_sec,
                xfmr_ii_sec,
            )
        elif self.is_wye_delta():
            self._add_ygd_eqns(
                KCL_r,
                KCL_i,
                xfmr_ir_aux,
                xfmr_ii_aux,
                xfmr_vr_pri,
                xfmr_vi_pri,
                xfmr_ir_sec,
                xfmr_ii_sec,
                xfmr_vr_kvl,
                xfmr_vi_kvl,
            )
        elif self.is_delta_delta():
            self._add_dd_eqns(
                KCL_r,
                KCL_i,
                xfmr_ir_aux,
                xfmr_ii_aux,
                xfmr_vr_pri,
                xfmr_vi_pri,
                xfmr_ir_sec,
                xfmr_ii_sec,
                xfmr_vr_kvl,
                xfmr_vi_kvl,
            )
        elif self.is_wye_zigzag():
            self._add_yzg_eqns(
                KCL_r,
                KCL_i,
                xfmr_ir_aux,
                xfmr_ii_aux,
                xfmr_vr_pri,
                xfmr_vi_pri,
                xfmr_ir_sec,
                xfmr_ii_sec,
            )
        else:
            self._add_yy_eqns(
                KCL_r,
                KCL_i,
                xfmr_ir_aux,
                xfmr_ii_aux,
                xfmr_vr_pri,
                xfmr_vi_pri,
                xfmr_ir_sec,
                xfmr_ii_sec,
            )

    def _store_coupling(
        self,
        idx,
        c1r,
        c1i,
        c2r,
        c2i,
        c3r,
        c3i,
        vr_pri,
        vi_pri,
        ir_aux,
        ii_aux,
        ir_sec,
        ii_sec,
    ):
        """Record the coupling terms this winding contributes, for reporting."""
        _kcl(c1r, c1i, idx, vr_pri, vi_pri)
        _kcl(c2r, c2i, idx, ir_aux, ii_aux)
        _kcl(c3r, c3i, idx, ir_sec, ii_sec)

    def _add_yy_eqns(self, KCL_r, KCL_i, ia, iia, vp, vip, irs, iis):
        """State the equations of a wye-wye transformer."""
        vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec = self._calc_constraints_yy()
        self._inject_terminal_current(
            KCL_r,
            KCL_i,
            self.from_bus_pos,
            self.from_bus_neg,
            self.ir_from,
            self.ii_from,
            side="pri",
        )
        self._inject_terminal_current(
            KCL_r,
            KCL_i,
            self.to_bus_pos,
            self.to_bus_neg,
            self.ir_to,
            self.ii_to,
            side="sec",
        )
        self._store_coupling(
            self.id,
            vp,
            vip,
            ia,
            iia,
            irs,
            iis,
            vr_pri,
            vi_pri,
            ir_aux,
            ii_aux,
            ir_sec,
            ii_sec,
        )

    def _add_yzg_eqns(self, KCL_r, KCL_i, ia, iia, vp, vip, irs, iis):
        """State the equations of a wye-grounded-zigzag transformer."""
        # Grounded zigzag secondaries use the grounded-wye equations in the
        # current internal formulation while retaining their explicit type.
        self._add_yy_eqns(KCL_r, KCL_i, ia, iia, vp, vip, irs, iis)

    def _add_dyg_eqns(self, KCL_r, KCL_i, ia, iia, vp, vip, irs, iis):
        """State the equations of a delta-grounded-wye transformer."""
        vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec = self._calc_constraints_dyg()
        ir_br, ii_br = self.ir_pri, self.ii_pri
        if self.shunt_side != "sec":
            vr_ll, vi_ll = self._primary_branch_voltage()
            ir_sh, ii_sh = self._shunt_current(vr_ll, vi_ll, side="pri")
            ir_br, ii_br = self.ir_pri + ir_sh, self.ii_pri + ii_sh
        self._inject_terminal_current(
            KCL_r,
            KCL_i,
            self.from_bus_pos,
            self.from_bus_neg,
            ir_br,
            ii_br,
            side="pri",
        )
        if self.from_bus_neg is not None:
            self._apply_delta_antifloat(
                KCL_r,
                KCL_i,
                self.from_bus_pos,
                self.from_bus_neg,
                self.vr_pos,
                self.vi_pos,
                self.vr_neg,
                self.vi_neg,
                self.anti_float_b_pri_raw,
            )
        else:
            # Handle anti-float for Open DSS single-terminal delta (effectively grounded)
            irp, iip = self._anti_float_current(
                self.vr_pos, self.vi_pos, self.anti_float_b_pri
            )
            _kcl(KCL_r, KCL_i, self.from_bus_pos.int_bus_id, irp, iip)
        sec_ir, sec_ii = self.ir_sec, self.ii_sec
        if self.shunt_side == "sec":
            vr_s, vi_s = self._secondary_branch_voltage()
            ir_sh, ii_sh = self._shunt_current(vr_s, vi_s, side="sec")
            sec_ir += ir_sh
            sec_ii += ii_sh
        self._inject_terminal_current(
            KCL_r, KCL_i, self.to_bus_pos, self.to_bus_neg, sec_ir, sec_ii, side="sec"
        )
        self._store_coupling(
            self.id,
            vp,
            vip,
            ia,
            iia,
            irs,
            iis,
            vr_pri,
            vi_pri,
            ir_aux,
            ii_aux,
            ir_sec,
            ii_sec,
        )

    def _add_ygd_eqns(
        self, KCL_r, KCL_i, ia, iia, vp, vip, irs, iis, vr_kvl=None, vi_kvl=None
    ):
        """State the equations of a grounded-wye-delta transformer."""
        vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec = self._calc_constraints_ygd()
        self._inject_terminal_current(
            KCL_r,
            KCL_i,
            self.from_bus_pos,
            self.from_bus_neg,
            self.ir_pri,
            self.ii_pri,
            side="pri",
        )
        vr_ll, vi_ll = self._secondary_branch_voltage()
        ir_sh, ii_sh = self._shunt_current(vr_ll, vi_ll, side="sec")
        ir_br, ii_br = self.ir_sec + ir_sh, self.ii_sec + ii_sh
        self._inject_terminal_current(
            KCL_r,
            KCL_i,
            self.to_bus_pos,
            self.to_bus_neg,
            ir_br,
            ii_br,
            side="sec",
        )
        if self.to_bus_neg is not None:
            self._apply_delta_antifloat(
                KCL_r,
                KCL_i,
                self.to_bus_pos,
                self.to_bus_neg,
                self.vr_s_pos,
                self.vi_s_pos,
                self.vr_s_neg,
                self.vi_s_neg,
                self.anti_float_b_sec_raw,
            )
        else:
            irp, iip = self._anti_float_current(
                self.vr_s_pos, self.vi_s_pos, self.anti_float_b_sec
            )
            _kcl(KCL_r, KCL_i, self.to_bus_pos.int_bus_id, irp, iip)
        self._store_coupling(
            self.id,
            vp,
            vip,
            ia,
            iia,
            irs,
            iis,
            vr_pri,
            vi_pri,
            ir_aux,
            ii_aux,
            ir_sec,
            ii_sec,
        )

    def _add_dd_eqns(
        self, KCL_r, KCL_i, ia, iia, vp, vip, irs, iis, vr_kvl=None, vi_kvl=None
    ):
        """State the equations of a delta-delta transformer."""
        vr_pri, vi_pri, ir_aux, ii_aux, ir_sec, ii_sec = self._calc_constraints_dd()
        vr_p_ll, vi_p_ll = self._primary_branch_voltage()
        ir_sh, ii_sh = self._shunt_current(vr_p_ll, vi_p_ll, side="pri")
        ir_br_p, ii_br_p = self.ir_pri + ir_sh, self.ii_pri + ii_sh
        self._inject_terminal_current(
            KCL_r,
            KCL_i,
            self.from_bus_pos,
            self.from_bus_neg,
            ir_br_p,
            ii_br_p,
            side="pri",
        )
        if self.from_bus_neg is not None:
            self._apply_delta_antifloat(
                KCL_r,
                KCL_i,
                self.from_bus_pos,
                self.from_bus_neg,
                self.vr_pos,
                self.vi_pos,
                self.vr_neg,
                self.vi_neg,
                self.anti_float_b_pri_raw,
            )
        else:
            irp, iip = self._anti_float_current(
                self.vr_pos, self.vi_pos, self.anti_float_b_pri
            )
            _kcl(KCL_r, KCL_i, self.from_bus_pos.int_bus_id, irp, iip)

        self._inject_terminal_current(
            KCL_r,
            KCL_i,
            self.to_bus_pos,
            self.to_bus_neg,
            self.ir_sec,
            self.ii_sec,
            side="sec",
        )
        if self.to_bus_neg is not None:
            self._apply_delta_antifloat(
                KCL_r,
                KCL_i,
                self.to_bus_pos,
                self.to_bus_neg,
                self.vr_s_pos,
                self.vi_s_pos,
                self.vr_s_neg,
                self.vi_s_neg,
                self.anti_float_b_sec_raw,
            )
        else:
            irp, iip = self._anti_float_current(
                self.vr_s_pos, self.vi_s_pos, self.anti_float_b_sec
            )
            _kcl(KCL_r, KCL_i, self.to_bus_pos.int_bus_id, irp, iip)

        self._store_coupling(
            self.id,
            vp,
            vip,
            ia,
            iia,
            irs,
            iis,
            vr_pri,
            vi_pri,
            ir_aux,
            ii_aux,
            ir_sec,
            ii_sec,
        )
