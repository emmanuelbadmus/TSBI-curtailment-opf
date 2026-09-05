"""Nodal shunt admittance matrix."""

from __future__ import annotations

import numpy as np

_CURRENT_BASE_EPS = 1e-18
_ADMITTANCE_VALUE_EPS = 1e-18


class Shunt:
    """Injects ``I = Y * V`` at a set of phase terminals."""

    def __init__(self, terminal_buses, y_matrix_s):
        self.terminal_buses = list(terminal_buses)
        self.y_matrix_s = np.array(y_matrix_s, dtype=complex)
        n = len(self.terminal_buses)
        if self.y_matrix_s.shape != (n, n):
            raise ValueError(
                f"Shunt expected Y shape {(n, n)}, got {self.y_matrix_s.shape}"
            )
        self._vr = [0.0] * n
        self._vi = [0.0] * n
        self._y_pu = np.zeros((n, n), dtype=complex)
        for i, bus_i in enumerate(self.terminal_buses):
            i_base = float(getattr(bus_i, "i_base_a", 0.0))
            if abs(i_base) < _CURRENT_BASE_EPS:
                continue
            for j, bus_j in enumerate(self.terminal_buses):
                y_ij = self.y_matrix_s[i, j]
                if abs(y_ij) < _ADMITTANCE_VALUE_EPS:
                    continue
                self._y_pu[i, j] = y_ij * (float(bus_j.v_base_v) / i_base)

    def assign_ipopt_vars(self, model):
        """Point this shunt at the voltage variables of its terminals."""
        for i, bus in enumerate(self.terminal_buses):
            idx = bus.int_bus_id
            self._vr[i] = model.ipopt_vr_list[idx]
            self._vi[i] = model.ipopt_vi_list[idx]

    def add_to_eqn_list(self, KCL_r, KCL_i):
        """Add the shunt's admittance current to its terminals' KCL rows."""
        for i, bus_i in enumerate(self.terminal_buses):
            ir = 0
            ii = 0
            for j, y_ij in enumerate(self._y_pu[i]):
                if abs(y_ij) < _ADMITTANCE_VALUE_EPS:
                    continue
                vr = self._vr[j]
                vi = self._vi[j]
                ir += y_ij.real * vr - y_ij.imag * vi
                ii += y_ij.real * vi + y_ij.imag * vr
            idx = bus_i.int_bus_id
            KCL_r[idx] = KCL_r.get(idx, 0) + ir
            KCL_i[idx] = KCL_i.get(idx, 0) + ii
