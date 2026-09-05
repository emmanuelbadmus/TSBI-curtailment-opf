"""Internal network containers used by the BMOPF parser and solver."""

import math
from itertools import count

from src.components.bus import Bus
from src.components.generator import Generator
from src.components.inverter import Inverter
from src.components.load import Load
from src.components.shunt import Shunt
from src.components.slack import Slack
from src.components.switch import Switch
from src.components.transformer import Transformer
from src.components.unbalanced_line import UnbalancedLine

# Shared network data model used by the BMOPF parser and optimizer.


class DxNetworkModel:
    """The network: its buses and every component attached to them."""

    GROUND_NODE = None
    FREQUENCY = 60
    OMEGA = 2 * math.pi * FREQUENCY

    def __init__(self):
        self.buses = []
        self.loads = []
        self.generators = []
        self.inverters = []
        self.slack = []
        self.matrix_shunts = []
        setattr(self, "3p_transformers", [])
        self.switches = []
        self.lines = []
        self.bus_name_map = {}
        self.load_name_map = {}
        self._bus_id_counter = count(0)

    def create_bus(
        self,
        v_mag,
        v_ang,
        node_name,
        node_parent,
        node_phase,
        is_virtual,
        is_triplex=False,
    ):
        """Add one phase terminal and return it."""
        bus = Bus(
            next(self._bus_id_counter),
            1,
            v_mag,
            v_ang,
            None,
            node_name,
            node_parent,
            node_phase,
            is_virtual,
            is_triplex=is_triplex,
        )
        self.bus_name_map[f"{node_name}_{node_phase}"] = bus
        self.buses.append(bus)
        return bus

    def create_load(
        self,
        name,
        from_bus,
        to_bus,
        p,
        q,
        load_num,
        phase,
        triplex_phase,
        i_const=0j,
        z_const=0j,
        nominal_v=0.0,
        load_model=1,
        current_follows_voltage_angle=True,
        current_reference_angle=None,
        cvr_watts=1.0,
        cvr_vars=2.0,
        vmin_pu=None,
        vmax_pu=None,
        use_opendss_voltage_limits=False,
    ):
        """Add one phase of a load and return it."""
        pq = Load(
            name,
            from_bus,
            to_bus,
            P=p,
            Q=q,
            I_const=i_const,
            Z_const=z_const,
            load_num=load_num,
            phase=phase,
            triplex_phase=triplex_phase,
            nominal_v=nominal_v,
            load_model=load_model,
            current_follows_voltage_angle=current_follows_voltage_angle,
            current_reference_angle=current_reference_angle,
            cvr_watts=cvr_watts,
            cvr_vars=cvr_vars,
            vmin_pu=vmin_pu,
            vmax_pu=vmax_pu,
            use_opendss_voltage_limits=use_opendss_voltage_limits,
        )
        self.loads.append(pq)
        self.load_name_map[f"{name}_{phase}"] = pq
        return pq

    def create_shunt(self, terminal_buses, y_matrix_s, name=None):
        """Add a shunt admittance across a set of terminals and return it."""
        shunt = Shunt(terminal_buses, y_matrix_s)
        shunt.name = name
        self.matrix_shunts.append(shunt)
        return shunt

    def create_generator(self, name, bus, p_kw, q_kvar, phase, gen_type=None):
        """Add a fixed power injection and return it."""
        gen = Generator(name, bus, p_kw, q_kvar, phase=phase, gen_type=gen_type)
        gen.gen_idx = len(self.generators)
        self.generators.append(gen)
        return gen

    def create_inverter(
        self,
        name,
        bus,
        p_w,
        q_var,
        phase,
        rating_va=None,
        **kwargs,
    ):
        """Add a two-stage inverter with its PV array and return it."""
        inverter = Inverter(
            name,
            bus,
            p_w,
            q_var,
            phase=phase,
            rating_va=rating_va,
            **kwargs,
        )
        inverter.inv_idx = len(self.inverters)
        self.inverters.append(inverter)
        return inverter

    def create_switch(
        self,
        fb,
        tb,
        st,
        ph,
        resistance_ohm,
        reactance_ohm,
        name=None,
    ):
        """Add one phase of a switch and return it."""
        sw = Switch(
            fb,
            tb,
            st,
            ph,
            resistance_ohm=resistance_ohm,
            reactance_ohm=reactance_ohm,
        )
        sw.name = name
        self.switches.append(sw)
        return sw

    def create_unbalanced_line(
        self, imp, shunt, frm, to, length, phases, amps, name=None
    ):
        """Add a multi-phase line with its series and shunt matrices."""
        line = UnbalancedLine(self, imp, shunt, frm, to, length, phases, amps)
        line.name = name
        self.lines.append(line)
        return line


# Clear static counters before parsing a new JSON document.


def reset_component_state():
    """Reset the static state of all component classes."""
    Bus.bus2index_dict = {}
    Bus.bus_id_counter = count(0)

    Slack.slack_id_counter = count(0)

    Load._ids = count(0)
    Generator._ids = count(0)
    Inverter._ids = count(0)
    Transformer._ids = count(0)
    Switch.switch_id_counter = count(0)
