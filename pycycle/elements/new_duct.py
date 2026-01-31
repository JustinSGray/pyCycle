"""
NewDuct - A duct element using JaxElement for automatic differentiation.

This is a single ExplicitComponent that replaces the Duct Group.
It uses functional thermo interfaces (CEAThermo or TabularThermo) and
provides analytical derivatives via JAX automatic differentiation.
"""

import time
import jax.numpy as jnp

from pycycle.jax_element_base import JaxElement
from pycycle.functional_thermo.jax_wrappers import (
    TotalPropsIdx as TPI, StaticPropsIdx as SPI
)

# Module-level timing accumulators
_new_duct_timing_stats = {
    'compute_calls': 0,
    'compute_time': 0.0,
    'partials_calls': 0,
    'partials_time': 0.0,
}

def reset_new_duct_timing_stats():
    """Reset all timing statistics."""
    for key in _new_duct_timing_stats:
        _new_duct_timing_stats[key] = 0.0 if 'time' in key else 0

def print_new_duct_timing_stats():
    """Print timing statistics."""
    stats = _new_duct_timing_stats
    print("\n=== NewDuct Timing Stats ===")
    print(f"  compute() calls: {stats['compute_calls']}")
    print(f"  compute() total time: {stats['compute_time']*1000:.3f} ms")
    if stats['compute_calls'] > 0:
        print(f"  compute() avg time: {stats['compute_time']*1000/stats['compute_calls']:.3f} ms")
    print(f"  compute_partials() calls: {stats['partials_calls']}")
    print(f"  compute_partials() total time: {stats['partials_time']*1000:.3f} ms")
    if stats['partials_calls'] > 0:
        print(f"  compute_partials() avg time: {stats['partials_time']*1000/stats['partials_calls']:.3f} ms")
    print("============================\n")


class NewDuct(JaxElement):
    """
    Duct element using functional thermodynamic interfaces with JAX derivatives.

    Calculates flow through a duct with specified MN (on design) or Area (off-design),
    including pressure loss and heat addition.
    """

    def initialize(self):
        super().initialize()

        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')
        self.options.declare('expMN', default=0.0,
                             desc='Mach number exponent for dPqP_MN calculations.')

        self.default_des_od_conns = [('Fl_O:stat:area', 'area')]

    def _get_jit_config_key(self):
        """
        Return config key including NewDuct-specific options.

        Includes design, statics, expMN, and jax_thermo identity to ensure
        instances with different configurations get separate JIT compilations.
        """
        return (
            type(self).__name__,
            self.options['design'],
            self.options['statics'],
            self.options['expMN'],
            id(self._jax_thermo) if self._jax_thermo is not None else None,
        )

    def pyc_setup_output_ports(self):
        self.copy_flow('Fl_I', 'Fl_O')

    def setup(self):
        design = self.options['design']
        statics = self.options['statics']
        expMN = self.options['expMN']

        # Configure flow ports for base class pre-linearization
        self._flow_in_port = 'Fl_I'
        self._flow_out_port = 'Fl_O'

        # --- Inputs ---
        # Add all flow inputs for pyCycle flow connections
        self.add_flow_input('Fl_I')

        # Register which inputs are used in compute_physics
        self.add_primal_input('Fl_I:tot:P', 'Pt_in')
        self.add_primal_input('Fl_I:tot:h', 'ht_in')
        self.add_primal_input('Fl_I:stat:W', 'W_in')
        self.add_primal_input('Fl_I:stat:MN', 'MN_in')
        self.add_primal_input('Fl_I:tot:composition', 'composition')

        self.add_input('Q_dot', val=0.0, units='Btu/s',
                       desc='Heat flow rate into (positive) or out of (negative) the air')
        self.add_primal_input('Q_dot', 'Q_dot')

        if expMN > 1e-10:
            if design:
                self.add_input('dPqP', val=0.0,
                               desc='Pressure differential as fraction of inlet pressure')
                self.add_primal_input('dPqP', 'dPqP_or_s')
            else:
                self.add_input('s_dPqP', val=0.0, desc='Pressure loss scalar')
                self.add_primal_input('s_dPqP', 'dPqP_or_s')
        else:
            self.add_input('dPqP', val=0.0,
                           desc='Pressure differential as fraction of inlet pressure')
            self.add_primal_input('dPqP', 'dPqP_or_s')

        if statics:
            if design:
                self.add_input('MN', val=0.5, desc='Exit Mach number')
                self.add_primal_input('MN', 'MN_or_area')
            else:
                self.add_input('area', val=1.0, units='inch**2', desc='Exit flow area')
                self.add_primal_input('area', 'MN_or_area')

        # --- Outputs ---
        # Add all flow outputs
        self.add_flow_output('Fl_O', statics=statics)

        # Register which outputs are computed by compute_physics
        # Order must match compute_physics return order
        self.add_flow_total_primal_outputs('Fl_O')
        self.add_primal_output('Fl_O:stat:W', 'W_out')

        if expMN > 1e-10:
            if design:
                self.add_output('s_dPqP', val=0.0, desc='Pressure loss scalar')
                self.add_primal_output('s_dPqP', 's_dPqP_out')
            else:
                self.add_output('dPqP', val=0.0,
                                desc='Pressure differential as fraction of inlet pressure')
                self.add_primal_output('dPqP', 'dPqP_out')

        if statics:
            self.add_flow_static_primal_outputs('Fl_O')

        # Declare partials between primal inputs and outputs
        super().setup_partials()

    def compute_physics(self, Pt_in, ht_in, W_in, MN_in, composition, Q_dot, dPqP_or_s, MN_or_area=None):
        """
        Pure JAX physics computation.

        Parameters
        ----------
        Pt_in : float
            Inlet total pressure
        ht_in : float
            Inlet total enthalpy
        W_in : float
            Inlet mass flow rate
        MN_in : float
            Inlet Mach number
        composition : array
            Flow composition. For TABULAR, composition[0] = FAR.
            For CEA, this is elemental fractions.
        Q_dot : float
            Heat flow rate
        dPqP_or_s : float
            Pressure loss (dPqP in design, s_dPqP in off-design)
        MN_or_area : float, optional
            Exit Mach number (design) or exit area (off-design)

        Returns
        -------
        tuple
            All output values in the order registered by add_primal_output
        """
        design = self.options['design']
        statics = self.options['statics']
        expMN = self.options['expMN']
        jt = self.jax_thermo

        # Pressure loss calculation
        if expMN > 1e-10:
            if design:
                dPqP = dPqP_or_s
                s_dPqP = jnp.where(MN_in > 1e-10, dPqP / MN_in**expMN, 0.0)
            else:
                s_dPqP = dPqP_or_s
                dPqP = s_dPqP * MN_in**expMN
        else:
            dPqP = dPqP_or_s
            s_dPqP = 0.0

        # Total properties - pass composition through (thermo extracts FAR internally)
        Pt_out = Pt_in * (1.0 - dPqP)
        ht_out = jnp.where(W_in > 1e-10, ht_in + Q_dot / W_in, ht_in)
        Tt_out = jt.T_from_hP(ht_out, Pt_out, composition)
        props = jt.props_TP(Tt_out, Pt_out, composition)

        # Build output list - must match add_primal_output order
        outputs = [
            Pt_out, Tt_out, ht_out,
            props[TPI.S], props[TPI.gamma], props[TPI.Cp],
            props[TPI.Cv], props[TPI.rho], props[TPI.R],
            W_in,  # Mass flow passthrough
        ]

        # s_dPqP or dPqP output (if expMN > 0)
        if expMN > 1e-10:
            outputs.append(s_dPqP if design else dPqP)

        # Static properties - pass composition through
        if statics:
            if design:
                static_props = jt.static_from_MN(Tt_out, Pt_out, MN_or_area, W_in, composition)
            else:
                static_props = jt.static_from_area(Tt_out, Pt_out, MN_or_area, W_in, composition)

            # Corrected flow
            Wc = W_in * jnp.sqrt(Tt_out / 518.67) / (Pt_out / 14.696)

            outputs.extend([
                static_props[SPI.hs], static_props[SPI.Ts], static_props[SPI.Ps],
                static_props[SPI.rhos], static_props[SPI.gamma], static_props[SPI.Cp],
                static_props[SPI.Cv], static_props[SPI.S], static_props[SPI.R],
                static_props[SPI.V], static_props[SPI.Vsonic], static_props[SPI.MN],
                static_props[SPI.area], Wc,
            ])

        return tuple(outputs)

    def compute(self, inputs, outputs):
        """Timed wrapper around base class compute."""
        t_start = time.perf_counter()

        super().compute(inputs, outputs)

        _new_duct_timing_stats['compute_calls'] += 1
        _new_duct_timing_stats['compute_time'] += (time.perf_counter() - t_start)

    def compute_partials(self, inputs, partials):
        """Timed wrapper around base class compute_partials."""
        t_start = time.perf_counter()

        super().compute_partials(inputs, partials)

        _new_duct_timing_stats['partials_calls'] += 1
        _new_duct_timing_stats['partials_time'] += (time.perf_counter() - t_start)


# Backward compatibility alias
Duct = NewDuct


if __name__ == "__main__":
    import openmdao.api as om
    from pycycle.mp_cycle import Cycle
    from pycycle.elements.flow_start import FlowStart
    from pycycle.thermo.cea import species_data

    p = om.Problem()
    cycle = p.model = Cycle()
    cycle.options['thermo_method'] = 'CEA'
    cycle.options['thermo_data'] = species_data.janaf

    cycle.add_subsystem('flow_start', FlowStart(), promotes=['MN', 'P', 'T'])
    cycle.add_subsystem('duct', NewDuct(), promotes=['MN'])

    cycle.pyc_connect_flow('flow_start.Fl_O', 'duct.Fl_I')

    cycle.set_input_defaults('MN', 0.5)
    cycle.set_input_defaults('duct.dPqP', 0.02)
    cycle.set_input_defaults('P', 17., units='psi')
    cycle.set_input_defaults('T', 500., units='degR')
    cycle.set_input_defaults('flow_start.W', 500., units='lbm/s')

    p.setup(check=False, force_alloc_complex=True)
    p.set_solver_print(level=-1)

    p.run_model()

    print("NewDuct test:")
    print(f"  Pt_out = {p['duct.Fl_O:tot:P'][0]:.4f} psi")
    print(f"  Tt_out = {p['duct.Fl_O:tot:T'][0]:.4f} degR")
    print(f"  ht_out = {p['duct.Fl_O:tot:h'][0]:.4f} Btu/lbm")

    print("\nChecking partials...")
    partial_data = p.check_partials(method='fd', compact_print=True,
                                    includes=['duct.*'], excludes=['*.base_thermo.*'])
