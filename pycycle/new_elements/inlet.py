"""
NewInlet - An inlet element using JaxElement for automatic differentiation.

This is a single ExplicitComponent that replaces the Inlet Group.
It computes ram drag, exit total pressure (via ram recovery), and
exit static properties using JAX automatic differentiation.
"""

import jax.numpy as jnp

from pycycle.new_elements.jax_element_base import JaxElement
from pycycle.constants import g_c

# Standard reference conditions for corrected flow calculation
_T_REF = 518.67   # Reference temperature (degR)
_P_REF = 14.696   # Reference pressure (psi)


class NewInlet(JaxElement):
    """
    Inlet element using functional thermodynamic interfaces with JAX derivatives.

    Calculates ram drag and exit flow conditions for an inlet with
    specified MN (on design) or area (off-design). The total temperature
    passes through unchanged; total pressure is reduced by the ram recovery factor.
    """

    def initialize(self):
        super().initialize()

        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')

        self.default_des_od_conns = [('Fl_O:stat:area', 'area')]

    def pyc_setup_output_ports(self):
        self.copy_flow('Fl_I', 'Fl_O')

    def setup(self):
        design = self.options['design']
        statics = self.options['statics']

        # --- Inputs ---
        self.add_flow_input('Fl_I')

        self.add_input('ram_recovery', val=1.0,
                       desc='Inlet ram recovery factor')

        if statics:
            if design:
                self.add_input('MN', val=0.5, desc='Exit Mach number')
            else:
                self.add_input('area', val=1.0, units='inch**2', desc='Exit flow area')

        # --- Outputs ---
        self.add_flow_output('Fl_O', statics=statics)

        self.add_output('F_ram', val=1.0, units='lbf', desc='Ram drag')

        # Build index mappings and declare partials
        self.setup_partials()

    def compute_physics(self, inputs):
        """
        Pure JAX physics computation with vector I/O.

        Parameters
        ----------
        inputs : jnp.ndarray
            Flat input vector. Use self.inp(inputs, 'name') to access values.

        Returns
        -------
        jnp.ndarray
            Flat output vector matching add_primal_output order.
        """
        design = self.options['design']
        statics = self.options['statics']
        thermo = self.jax_thermo

        # --- Extract inputs ---
        Pt_in = self.inp(inputs, 'Fl_I:tot:P')
        Tt_in = self.inp(inputs, 'Fl_I:tot:T')
        W_in = self.inp(inputs, 'Fl_I:stat:W')
        V_in = self.inp(inputs, 'Fl_I:stat:V')
        composition = self.inp(inputs, 'Fl_I:tot:composition')
        ram_recovery = self.inp(inputs, 'ram_recovery')

        # --- Inlet calculations ---
        Pt_out = Pt_in * ram_recovery
        F_ram = W_in * V_in / g_c

        # Total properties: temperature passes through, pressure changes
        props = thermo.set_total_TP(Tt_in, Pt_out, composition)

        # Total properties: h, T, P, rho, gamma, Cp, Cv, S, R
        outputs = [
            props.h, Tt_in, Pt_out,
            props.rho, props.gamma, props.Cp,
            props.Cv, props.S, props.R,
        ]

        # Static properties
        if statics:
            if design:
                MN_exit = self.inp(inputs, 'MN')
                static_props = thermo.set_static_MN(Tt_in, Pt_out, MN_exit, W_in, composition)
            else:
                area_exit = self.inp(inputs, 'area')
                static_props = thermo.set_static_area(Tt_in, Pt_out, area_exit, W_in, composition)

            # Corrected flow
            Wc = W_in * jnp.sqrt(Tt_in / _T_REF) / (Pt_out / _P_REF)

            # Static properties: h, T, P, rho, gamma, Cp, Cv, S, R, V, Vsonic, MN, area, Wc
            outputs.extend([
                static_props.hs, static_props.Ts, static_props.Ps,
                static_props.rhos, static_props.gamma, static_props.Cp,
                static_props.Cv, static_props.S, static_props.R,
                static_props.V, static_props.Vsonic, static_props.MN,
                static_props.area, Wc,
            ])

        # W passthrough and F_ram
        outputs.append(W_in)
        outputs.append(F_ram)

        return jnp.array(outputs)
