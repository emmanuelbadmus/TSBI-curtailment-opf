"""Bus node in the power-system network."""

import math
from itertools import count
from typing import ClassVar

from src import network_simulator as per_unit

_BUS_ANTI_FLOAT_CONDUCTANCE_PU = 1e-9


class Bus:
    """One phase terminal of one bus: its voltage variable and per-unit base."""

    bus2index_dict: ClassVar[dict] = {}
    bus_id_counter: ClassVar[count] = count(0)

    def __init__(
        self,
        bus_id,
        bus_type,
        vm_init,
        va_init,
        area,
        node_name=None,
        node_parent=None,
        node_phase=None,
        is_virtual=False,
        is_triplex=False,
    ):
        self.bus, self.Type = bus_id, bus_type
        self.NodeName = node_name or f"Bus:{bus_id}"
        self.NodeParent, self.NodePhase = node_parent, (node_phase or "NA")
        self.IsVirtual, self.V_Nominal, self.Va_init = (
            is_virtual,
            float(vm_init),
            float(va_init),
        )
        self.raw_nominal_v = float(vm_init)
        self.v_base_v = per_unit.resolve_voltage_base(
            self.raw_nominal_v, fallback_v=1.0
        )
        self.s_base_va = per_unit.s_base_1ph_va()
        self.i_base_a = self.s_base_va / self.v_base_v
        self.is_triplex = is_triplex
        self.Vr, self.Vi = (
            self.V_Nominal * math.cos(self.Va_init),
            self.V_Nominal * math.sin(self.Va_init),
        )
        self.Vr_pu = per_unit.voltage_to_pu(self.Vr, self.v_base_v)
        self.Vi_pu = per_unit.voltage_to_pu(self.Vi, self.v_base_v)
        self.vm_pu = (self.Vr_pu**2 + self.Vi_pu**2) ** 0.5
        self.vm_v = (self.Vr**2 + self.Vi**2) ** 0.5
        self.angle_deg = (
            math.degrees(math.atan2(self.Vi_pu, self.Vr_pu)) if self.vm_pu > 0 else 0.0
        )
        # Explicit solved views are updated by the solver after convergence.
        self.vr_pu, self.vi_pu = self.Vr_pu, self.Vi_pu
        self.vr_v, self.vi_v = self.Vr, self.Vi
        self.ipopt_vr_pu = self.ipopt_vi_pu = None
        self.ipopt_vr = self.ipopt_vi = None
        self.g_shunt_eps = _BUS_ANTI_FLOAT_CONDUCTANCE_PU
        if self.NodeName != "Gnd":
            self.int_bus_id = next(self.bus_id_counter)
            self.bus2index_dict[bus_id] = self.int_bus_id

    def create_ipopt_bus_vars(self, model):
        """Point this bus at its real and imaginary voltage variables."""
        idx = self.bus2index_dict[self.bus]
        self.ipopt_vr_pu = model.ipopt_vr_list[idx]
        self.ipopt_vi_pu = model.ipopt_vi_list[idx]
        self.ipopt_vr, self.ipopt_vi = self.ipopt_vr_pu, self.ipopt_vi_pu

    def initialize_voltages(self):
        """Seed the voltage variables with this bus's starting point."""
        for v, val in [(self.ipopt_vr_pu, self.Vr_pu), (self.ipopt_vi_pu, self.Vi_pu)]:
            if v is not None and hasattr(v, "value"):
                v.value = val

    def add_to_eqn_list(self, eq_real, eq_imag):
        """Add the anti-float shunt current to this bus's KCL rows."""
        idx = self.int_bus_id
        eq_real[idx] = eq_real.get(idx, 0) + self.g_shunt_eps * self.ipopt_vr_pu
        eq_imag[idx] = eq_imag.get(idx, 0) + self.g_shunt_eps * self.ipopt_vi_pu


Bus.bus2index_dict, Bus.bus_id_counter = {}, count(0)
GROUND = Bus("Gnd", "Gnd", 0.0, 0.0, 0, node_name="Gnd", node_parent="Gnd")
