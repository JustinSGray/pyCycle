"""
NewDuct - A duct element using JaxElement for automatic differentiation.

This is a single ExplicitComponent that replaces the Duct Group.
It uses JaxTabularThermo for thermodynamic calculations and provides
analytical derivatives via JAX automatic differentiation.
"""

import jax.numpy as jnp

from pycycle.jax_element_base import JaxElement

# Threshold for treating expMN as effectively zero
_EXPMN_THRESHOLD = 1e-10

# Standard reference conditions for corrected flow calculation
_T_REF = 518.67   # Reference temperature (degR)
_P_REF = 14.696   # Reference pressure (psi)


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

        if expMN > _EXPMN_THRESHOLD:
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

        if expMN > _EXPMN_THRESHOLD:
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
            Pressure loss parameter. When expMN > 0: dPqP in design mode,
            s_dPqP in off-design mode. When expMN == 0: always dPqP.
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
        if expMN > _EXPMN_THRESHOLD:
            if design:
                dPqP = dPqP_or_s
                s_dPqP = jnp.where(MN_in > _EXPMN_THRESHOLD, dPqP / MN_in**expMN, 0.0)
            else:
                s_dPqP = dPqP_or_s
                dPqP = s_dPqP * MN_in**expMN
        else:
            dPqP = dPqP_or_s
            s_dPqP = 0.0

        # Total properties
        Pt_out = Pt_in * (1.0 - dPqP)
        ht_out = jnp.where(W_in > _EXPMN_THRESHOLD, ht_in + Q_dot / W_in, ht_in)
        Tt_out = thermo.set_total_hP(ht_out, Pt_out, FAR)
        props = thermo.set_total_TP(Tt_out, Pt_out, FAR)

        # Build output list - must match add_primal_output order
        outputs = [
            Pt_out, Tt_out, ht_out,
            props.S, props.gamma, props.Cp,
            props.Cv, props.rho, props.R,
            W_in,  # Mass flow passthrough
        ]

        # s_dPqP or dPqP output (if expMN > 0)
        if expMN > _EXPMN_THRESHOLD:
            outputs.append(s_dPqP if design else dPqP)

        # Static properties
        if statics:
            if design:
                static_props = thermo.set_static_MN(Tt_out, Pt_out, MN_or_area, W_in, FAR)
            else:
                static_props = thermo.set_static_area(Tt_out, Pt_out, MN_or_area, W_in, FAR)

            # Corrected flow (normalized to standard day conditions)
            Wc = W_in * jnp.sqrt(Tt_out / _T_REF) / (Pt_out / _P_REF)

            outputs.extend([
                static_props.hs, static_props.Ts, static_props.Ps,
                static_props.rhos, static_props.gamma, static_props.Cp,
                static_props.Cv, static_props.S, static_props.R,
                static_props.V, static_props.Vsonic, static_props.MN,
                static_props.area, Wc,
            ])

        return tuple(outputs)
