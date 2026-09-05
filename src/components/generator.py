"""General generation component with explicit current variables."""

from itertools import count

from pyomo.common.errors import PyomoException
from pyomo.environ import ConstraintList, value

from src import network_simulator as per_unit

from .bus import Bus


class Generator:
    """A fixed real and reactive power injection at one terminal."""

    _ids = count(0)

    def __init__(
        self,
        name,
        bus: Bus,
        p_kw: float,
        q_kvar: float,
        phase=None,
        gen_type=None,
        model_num: int = 1,
        vpu_target=None,
        max_q_var=None,
        min_q_var=None,
        pv_factor: float = 0.1,
    ):
        self.id = next(self._ids)
        self.name = name
        self.phase = phase
        self.bus = bus
        self.bus_idx = bus.int_bus_id

        self.gen_type = gen_type
        self.model_num = int(model_num or 1)
        self.vpu_target = None if vpu_target is None else float(vpu_target)
        self.max_q_var = None if max_q_var is None else float(max_q_var)
        self.min_q_var = None if min_q_var is None else float(min_q_var)
        self.pv_factor = float(pv_factor)

        # raw power data
        self.raw_p_w = float(p_kw)
        self.raw_q_var = float(q_kvar)

        # per-unit power data
        self.P_pu = per_unit.power_to_pu(self.raw_p_w, s_base_va=bus.s_base_va)
        self.Q_pu = per_unit.power_to_pu(self.raw_q_var, s_base_va=bus.s_base_va)

        # mapped voltage vars
        self.ipopt_vr = None
        self.ipopt_vi = None

        # explicit current vars
        self.ipopt_ir = None
        self.ipopt_ii = None

        self._model = None
        self._constraint_list = None
        self._constraints_built = False

    def assign_ipopt_vars(self, model):
        """Map voltage and pre-created current variables from the model."""
        self._model = model
        self.ipopt_vr = model.ipopt_vr_list[self.bus_idx]
        self.ipopt_vi = model.ipopt_vi_list[self.bus_idx]

        if not hasattr(model, "_dx_generator_constraints"):
            model._dx_generator_constraints = ConstraintList()
        self._constraint_list = model._dx_generator_constraints

        self.ipopt_ir = model.gen_ir_list[self.gen_idx]
        self.ipopt_ii = model.gen_ii_list[self.gen_idx]
        self.initialize_current_from_voltage()

    def initialize_current_from_voltage(self):
        """Seed current variables from the generator power equation."""
        if self.ipopt_ir is None or self.ipopt_ii is None:
            return
        try:
            vr = float(value(self.ipopt_vr))
            vi = float(value(self.ipopt_vi))
            denom = vr * vr + vi * vi
            if denom <= 0.0:
                return
            self.ipopt_ir.value = (self.P_pu * vr + self.Q_pu * vi) / denom
            self.ipopt_ii.value = (self.P_pu * vi - self.Q_pu * vr) / denom
        except (PyomoException, TypeError, ValueError):
            return

    def _build_current_constraints(self):
        """Enforce generator power through bilinear current-voltage equations."""
        if self._constraints_built:
            return
        if self._model is None:
            raise RuntimeError(
                f"Generator {self.name}: assign_ipopt_vars() must be called first."
            )

        # zero-generation case: avoid underdetermined current variables
        if abs(self.P_pu) < 1e-16 and abs(self.Q_pu) < 1e-16:
            self._constraint_list.add(self.ipopt_ir == 0.0)
            self._constraint_list.add(self.ipopt_ii == 0.0)
        else:
            # Generator injection convention:
            # S = P + jQ = V * conj(I_inj)
            # => P = vr*ir + vi*ii
            # => Q = vi*ir - vr*ii
            self._constraint_list.add(
                self.ipopt_vr * self.ipopt_ir + self.ipopt_vi * self.ipopt_ii
                == self.P_pu
            )
            self._constraint_list.add(
                self.ipopt_vi * self.ipopt_ir - self.ipopt_vr * self.ipopt_ii
                == self.Q_pu
            )

        self._constraints_built = True

    def add_to_eqn_list(self, KCL_r, KCL_i):
        """
        Incorporate generated current into nodal KCL.

        Standard lines / loads are written as current leaving the bus, so
        generator injection enters KCL with a negative sign.
        """
        self._build_current_constraints()

        KCL_r[self.bus_idx] = KCL_r.get(self.bus_idx, 0) - self.ipopt_ir
        KCL_i[self.bus_idx] = KCL_i.get(self.bus_idx, 0) - self.ipopt_ii
