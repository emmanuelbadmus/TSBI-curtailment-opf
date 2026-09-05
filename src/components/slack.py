"""Voltage-source bus (swing bus)."""

from __future__ import annotations

import math
from itertools import count

from src import network_simulator as per_unit

from .bus import Bus


class Slack:
    """A terminal held at a fixed voltage, injecting whatever current that needs."""

    slack_id_counter = count(0)

    def __init__(self, bus: Bus, Vset, ang):
        self.bus = bus
        self.Vset_v = float(Vset)
        self.Vset_pu = per_unit.voltage_to_pu(self.Vset_v, self.bus.v_base_v)
        bus.Type = 0
        self.Vr_set = self.Vset_pu * math.cos(ang)
        self.Vi_set = self.Vset_pu * math.sin(ang)
        bus.Vr_pu = bus.vr_pu = self.Vr_set
        bus.Vi_pu = bus.vi_pu = self.Vi_set
        bus.vm_pu = math.hypot(self.Vr_set, self.Vi_set)
        bus.Vr = bus.vr_v = per_unit.voltage_from_pu(self.Vr_set, bus.v_base_v)
        bus.Vi = bus.vi_v = per_unit.voltage_from_pu(self.Vi_set, bus.v_base_v)
        bus.vm_v = math.hypot(bus.Vr, bus.Vi)
        bus.angle_deg = math.degrees(ang)
        self.slack_idx = next(self.slack_id_counter)

    def assign_ipopt_vars(self, model):
        """Point this source at its bus voltage and injected-current variables."""
        bid = self.bus.int_bus_id
        self.ipopt_vr = model.ipopt_vr_list[bid]
        self.ipopt_vi = model.ipopt_vi_list[bid]
        self.ipopt_ir_slack = model.slack_vr_list[self.slack_idx]
        self.ipopt_ii_slack = model.slack_vi_list[self.slack_idx]

    def cal_slack_data(self):
        """Return the voltage mismatch and the injected current of this source."""
        return (
            self.ipopt_vr - self.Vr_set,
            self.ipopt_vi - self.Vi_set,
            self.ipopt_ir_slack,
            self.ipopt_ii_slack,
        )

    def add_to_eqn_list(self, KCL_r, KCL_i, vr_cons, vi_cons):
        """Hold the bus at its set voltage and inject the current that requires."""
        Vr_c, Vi_c, Ir, Ii = self.cal_slack_data()
        bid = self.bus.int_bus_id
        KCL_r[bid] = KCL_r.get(bid, 0) - Ir
        KCL_i[bid] = KCL_i.get(bid, 0) - Ii
        vr_cons[bid], vi_cons[bid] = Vr_c, Vi_c
