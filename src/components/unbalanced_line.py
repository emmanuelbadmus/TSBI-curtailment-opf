"""Pi-model for multiphase and triplex distribution lines.

I_series = Y_row @ (V_from - V_to), I_shunt = (Y_sh_row / 2) @ V
"""

import numpy as np


class UnbalancedLinePhase:
    """One phase of an unbalanced line."""

    def __init__(
        self,
        from_el,
        to_el,
        pair_idx,
        phase_pairs,
        admittances,
        shunt_admittances,
        network_model,
        ampacity_a=None,
    ):
        self.phase_pairs = list(phase_pairs)
        self.pair_idx = int(pair_idx)
        self.from_phase, self.to_phase = self.phase_pairs[self.pair_idx]
        self.phase = (
            self.from_phase
            if self.from_phase == self.to_phase
            else f"{self.from_phase}->{self.to_phase}"
        )
        self.admittances, self.shunt_admittances = admittances, shunt_admittances
        self.ampacity_a = None if ampacity_a is None else float(ampacity_a)
        self.is_triplex = any(
            phase in {"1", "2"} for pair in self.phase_pairs for phase in pair
        )
        bm = network_model.bus_name_map
        self.from_bus = bm[f"{from_el}_{self.from_phase}"]
        self.to_bus = bm[f"{to_el}_{self.to_phase}"]
        self.from_bus_idx, self.to_bus_idx = (
            self.from_bus.int_bus_id,
            self.to_bus.int_bus_id,
        )
        self.i_base_from = self.from_bus.i_base_a
        self.i_base_to = self.to_bus.i_base_a
        self._vr_from, self._vi_from = {}, {}
        self._vr_to, self._vi_to = {}, {}
        self._y_from_from = {}
        self._y_from_to = {}
        self._y_to_from = {}
        self._y_to_to = {}
        self._ysh_from = {}
        self._ysh_to = {}
        self._from_el, self._to_el, self._bm = from_el, to_el, bm
        row = self.pair_idx
        for j, (from_phase, to_phase) in enumerate(self.phase_pairs):
            from_bus_j = bm[f"{from_el}_{from_phase}"]
            to_bus_j = bm[f"{to_el}_{to_phase}"]
            y = self.admittances[row][j]
            self._y_from_from[j] = y * (from_bus_j.v_base_v / self.i_base_from)
            self._y_from_to[j] = y * (to_bus_j.v_base_v / self.i_base_from)
            self._y_to_from[j] = y * (from_bus_j.v_base_v / self.i_base_to)
            self._y_to_to[j] = y * (to_bus_j.v_base_v / self.i_base_to)
            if self.shunt_admittances is not None:
                # Some GLD feeders provide explicit-neutral series terms (4x4 Z)
                # while shunt charging is only defined for phase conductors (3x3 C).
                # Missing shunt entries are physically zero.
                if (
                    row < self.shunt_admittances.shape[0]
                    and j < self.shunt_admittances.shape[1]
                ):
                    y_sh = 0.5 * self.shunt_admittances[row][j]
                else:
                    y_sh = 0.0j
                self._ysh_from[j] = y_sh * (from_bus_j.v_base_v / self.i_base_from)
                self._ysh_to[j] = y_sh * (to_bus_j.v_base_v / self.i_base_to)

    def create_ipopt_vars(self, model, buses):
        """Point this line at the voltage variables of both its ends."""
        for idx, (from_phase, to_phase) in enumerate(self.phase_pairs):
            f_id = self._bm[f"{self._from_el}_{from_phase}"].int_bus_id
            t_id = self._bm[f"{self._to_el}_{to_phase}"].int_bus_id
            self._vr_from[idx], self._vi_from[idx] = (
                buses[f_id].ipopt_vr,
                buses[f_id].ipopt_vi,
            )
            self._vr_to[idx], self._vi_to[idx] = (
                buses[t_id].ipopt_vr,
                buses[t_id].ipopt_vi,
            )

    def _calc_series_current(self):
        """Return the series current entering each end of the line."""
        Ir_f = Ii_f = Ir_t = Ii_t = 0
        for j in range(len(self.phase_pairs)):
            y_ff = self._y_from_from[j]
            y_ft = self._y_from_to[j]
            y_tf = self._y_to_from[j]
            y_tt = self._y_to_to[j]
            vr_f, vi_f = self._vr_from[j], self._vi_from[j]
            vr_t, vi_t = self._vr_to[j], self._vi_to[j]

            Ir_f += (vr_f * y_ff.real - vi_f * y_ff.imag) - (
                vr_t * y_ft.real - vi_t * y_ft.imag
            )
            Ii_f += (vi_f * y_ff.real + vr_f * y_ff.imag) - (
                vi_t * y_ft.real + vr_t * y_ft.imag
            )

            Ir_t += -(vr_f * y_tf.real - vi_f * y_tf.imag) + (
                vr_t * y_tt.real - vi_t * y_tt.imag
            )
            Ii_t += -(vi_f * y_tf.real + vr_f * y_tf.imag) + (
                vi_t * y_tt.real + vr_t * y_tt.imag
            )

        return Ir_f, Ir_t, Ii_f, Ii_t

    def _calc_shunt_current(self):
        """Return the charging current at each end of the line."""
        if self.shunt_admittances is None:
            return 0, 0, 0, 0
        Ir_f = Ii_f = Ir_t = Ii_t = 0
        for j in range(len(self.phase_pairs)):
            yf = self._ysh_from[j]
            yt = self._ysh_to[j]
            vr_f, vi_f = self._vr_from[j], self._vi_from[j]
            vr_t, vi_t = self._vr_to[j], self._vi_to[j]
            Ir_f += vr_f * yf.real - vi_f * yf.imag
            Ii_f += vi_f * yf.real + vr_f * yf.imag
            Ir_t += vr_t * yt.real - vi_t * yt.imag
            Ii_t += vi_t * yt.real + vr_t * yt.imag
        return Ir_f, Ii_f, Ir_t, Ii_t

    def add_to_eqn_list(self, KCL_r, KCL_i):
        """Add the line's series and charging currents to its ends' KCL rows."""
        Ir_f, Ir_t, Ii_f, Ii_t = self._calc_series_current()
        Ir_sf, Ii_sf, Ir_st, Ii_st = self._calc_shunt_current()
        KCL_r[self.from_bus_idx] = KCL_r.get(self.from_bus_idx, 0) + Ir_f + Ir_sf
        KCL_i[self.from_bus_idx] = KCL_i.get(self.from_bus_idx, 0) + Ii_f + Ii_sf
        KCL_r[self.to_bus_idx] = KCL_r.get(self.to_bus_idx, 0) + Ir_t + Ir_st
        KCL_i[self.to_bus_idx] = KCL_i.get(self.to_bus_idx, 0) + Ii_t + Ii_st

    def find_Ir_Ii_from(self):
        """Return the series current entering the from-end of the line."""
        Ir, _, Ii, _ = self._calc_series_current()
        return Ir, Ii


def _active_square(matrix):
    """Return the matrix with all-zero rows and columns removed."""
    arr = np.array(matrix, dtype=complex)
    row_mask = ~np.all(arr == 0, axis=1)
    col_mask = ~np.all(arr == 0, axis=0)
    mask = row_mask | col_mask
    return arr[np.ix_(mask, mask)]


def _invert_sparse(Z):
    """Invert a matrix that may carry inactive phases as zero rows."""
    active = _active_square(Z)
    if np.count_nonzero(active) == 1:
        return (1.0 / active[active != 0]).reshape(1, 1)
    return np.linalg.inv(active)


def _strip_zeros(Y):
    """Return the admittance matrix without its all-zero rows and columns."""
    Y = Y[~np.all(Y == 0, axis=1)]
    return Y[:, ~np.all(Y == 0, axis=0)]


class UnbalancedLine:
    """Multiphase line segment with one phase object per conductor pair."""

    def __init__(
        self,
        network_model,
        impedances,
        shunt_admittances,
        from_element,
        to_element,
        length,
        phases,
        ampacities=None,
    ):
        Z = np.array(impedances) * length
        try:
            Y = _invert_sparse(Z)
        except np.linalg.LinAlgError:
            Y = np.linalg.inv(_active_square(Z))
        Y_sh = None
        if shunt_admittances is not None and np.count_nonzero(shunt_admittances) > 0:
            Y_sh = _strip_zeros(np.array(shunt_admittances) * length)
        if phases and isinstance(phases[0], tuple):
            phase_pairs = [(str(from_p), str(to_p)) for from_p, to_p in phases]
        else:
            phase_pairs = [(str(phase), str(phase)) for phase in phases]
        n = min(Y.shape[0], len(phase_pairs))
        if ampacities is None:
            ampacities = []
        if np.isscalar(ampacities):
            ampacities = [ampacities]
        ampacities = list(ampacities)
        if len(ampacities) == 1 and len(phase_pairs) > 1:
            ampacities *= len(phase_pairs)
        self.lines = [
            UnbalancedLinePhase(
                from_element,
                to_element,
                idx,
                phase_pairs[:n],
                Y,
                Y_sh,
                network_model,
                ampacities[idx] if idx < len(ampacities) else None,
            )
            for idx in range(n)
        ]
