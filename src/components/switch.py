"""Ideal open or closed connection between two buses."""

from enum import Enum
from itertools import count

from src import network_simulator as per_unit

from .bus import Bus

_IMPEDANCE_VALUE_EPS = 1e-18


class SwitchStatus(Enum):
    """Whether a switch carries current or breaks the connection."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Switch:
    """One phase of a switch: a short series impedance, or an open circuit."""

    switch_id_counter = count(0)

    def __init__(
        self,
        from_node: Bus,
        to_node: Bus,
        status: SwitchStatus,
        phase,
        resistance_ohm,
        reactance_ohm,
    ):
        self.status, self.phase = status, phase
        self.from_bus, self.to_bus = from_node, to_node
        self.from_bus_idx, self.to_bus_idx = from_node.int_bus_id, to_node.int_bus_id
        self.v_to_on_from = per_unit.voltage_scale(
            self.to_bus.v_base_v, self.from_bus.v_base_v
        )
        self.i_to_on_from = per_unit.current_scale(
            self.from_bus.v_base_v, self.to_bus.v_base_v
        )
        # A switch carries the series impedance the source model solves it
        # with. Nothing is substituted when it is missing: a closed switch is
        # otherwise an ideal short, and inventing a value here would put a
        # fabricated impedance in the answer.
        if resistance_ohm is None or reactance_ohm is None:
            raise ValueError(
                f"Switch on phase {phase} between '{from_node.NodeName}' and "
                f"'{to_node.NodeName}' declares no series impedance; a closed "
                "switch is an ideal short without one. Give it 'r_series' and "
                "'x_series'."
            )
        r_ohm = float(resistance_ohm)
        x_ohm = float(reactance_ohm)
        if r_ohm * r_ohm + x_ohm * x_ohm <= _IMPEDANCE_VALUE_EPS:
            raise ValueError(
                f"Switch on phase {phase} between '{from_node.NodeName}' and "
                f"'{to_node.NodeName}' has a zero series impedance, which is a "
                "short circuit rather than a branch."
            )

        self.resistance_ohm = r_ohm
        self.reactance_ohm = x_ohm
        self.impedance_pu = per_unit.impedance_to_pu(
            complex(self.resistance_ohm, self.reactance_ohm),
            self.from_bus.v_base_v,
        )
        self.resistance_pu = float(self.impedance_pu.real)
        self.reactance_pu = float(self.impedance_pu.imag)
        self.switch_idx = next(self.switch_id_counter)
        self.from_node, self.to_node, self.vs = from_node, to_node, None

    def assign_ipopt_vars(self, model):
        """Point this switch at its terminal voltages and current variables."""
        f, t = self.from_bus_idx, self.to_bus_idx
        self.ipopt_vr_from, self.ipopt_vi_from = (
            model.ipopt_vr_list[f],
            model.ipopt_vi_list[f],
        )
        self.ipopt_vr_to, self.ipopt_vi_to = (
            model.ipopt_vr_list[t],
            model.ipopt_vi_list[t],
        )
        self.ipopt_irs_from = model.switch_ir_list[self.switch_idx]
        self.ipopt_iis_from = model.switch_ii_list[self.switch_idx]

    def initialize_ipopt_vars_from_voltage(self):
        """Seed the switch current from the voltage across it."""
        if self.status.value == "OPEN":
            self.ipopt_irs_from.value = 0.0
            self.ipopt_iis_from.value = 0.0
            return

        vr_drop = (
            float(self.ipopt_vr_from.value or 0.0)
            - float(self.ipopt_vr_to.value or 0.0) * self.v_to_on_from
        )
        vi_drop = (
            float(self.ipopt_vi_from.value or 0.0)
            - float(self.ipopt_vi_to.value or 0.0) * self.v_to_on_from
        )
        z = complex(self.resistance_pu, self.reactance_pu)
        if abs(z) <= _IMPEDANCE_VALUE_EPS:
            self.ipopt_irs_from.value = 0.0
            self.ipopt_iis_from.value = 0.0
            return
        current = complex(vr_drop, vi_drop) / z
        self.ipopt_irs_from.value = float(current.real)
        self.ipopt_iis_from.value = float(current.imag)

    def cal_switch_data(self):
        """Return the voltage mismatch and the terminal currents of this switch."""
        Ir, Ii = self.ipopt_irs_from, self.ipopt_iis_from
        if self.status.value == "OPEN":
            return Ir, Ii, Ir, Ii, -Ir * self.i_to_on_from, -Ii * self.i_to_on_from
        vr_drop = self.resistance_pu * Ir - self.reactance_pu * Ii
        vi_drop = self.resistance_pu * Ii + self.reactance_pu * Ir
        return (
            self.ipopt_vr_from - self.ipopt_vr_to * self.v_to_on_from - vr_drop,
            self.ipopt_vi_from - self.ipopt_vi_to * self.v_to_on_from - vi_drop,
            Ir,
            Ii,
            -Ir * self.i_to_on_from,
            -Ii * self.i_to_on_from,
        )

    def add_to_eqn_list(self, KCL_r, KCL_i, vr_cons, vi_cons):
        """Add the switch current to its terminals and state its voltage law."""
        Vr_c, Vi_c, Ir_f, Ii_f, Ir_t, Ii_t = self.cal_switch_data()
        for bus, ir, ii in [
            (self.from_bus_idx, Ir_f, Ii_f),
            (self.to_bus_idx, Ir_t, Ii_t),
        ]:
            KCL_r[bus] = KCL_r.get(bus, 0) + ir
            KCL_i[bus] = KCL_i.get(bus, 0) + ii
        vr_cons[self.switch_idx], vi_cons[self.switch_idx] = Vr_c, Vi_c
