"""
NewDuct - A duct element using JaxElement for automatic differentiation.

This is a single ExplicitComponent that replaces the Duct Group.
It uses JaxTabularThermo for thermodynamic calculations and provides
analytical derivatives via JAX automatic differentiation.
"""

import jax.numpy as jnp

from pycycle.jax_element_base import JaxElement
from pycycle.functional_thermo.jax_wrappers import (
    TotalPropsIdx as TPI, StaticPropsIdx as SPI
)


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
            Flow composition. composition[0] = FAR (fuel-to-air ratio).
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
        thermo = self.jax_thermo

        # Extract FAR from composition array
        FAR = composition[0]

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

        # Total properties
        Pt_out = Pt_in * (1.0 - dPqP)
        ht_out = jnp.where(W_in > 1e-10, ht_in + Q_dot / W_in, ht_in)
        Tt_out = thermo.T_from_hP(ht_out, Pt_out, FAR)
        props = thermo.props_TP(Tt_out, Pt_out, FAR)

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

        # Static properties
        if statics:
            if design:
                static_props = thermo.static_from_MN(Tt_out, Pt_out, MN_or_area, W_in, FAR)
            else:
                static_props = thermo.static_from_area(Tt_out, Pt_out, MN_or_area, W_in, FAR)

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
