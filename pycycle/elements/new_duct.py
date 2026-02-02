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

    # used for initialization purposes in pycycle
    # TODO: refactor to make this not needed later
    # maybe combine behavior into `add_flow_output`
    def pyc_setup_output_ports(self):
        self.copy_flow('Fl_I', 'Fl_O')

    def setup(self):
        design = self.options['design']
        statics = self.options['statics']
        expMN = self.options['expMN']

        # --- Inputs ---
        # Add all flow inputs for pyCycle flow connections
        self.add_flow_input('Fl_I')

        self.add_input('Q_dot', val=0.0, units='Btu/s',
                       desc='Heat flow rate into (positive) or out of (negative) the air')

        if expMN > _EXPMN_THRESHOLD:
            if design:
                self.add_input('dPqP', val=0.0,
                               desc='Pressure differential as fraction of inlet pressure')
            else:
                self.add_input('s_dPqP', val=0.0, desc='Pressure loss scalar')
        else:
            self.add_input('dPqP', val=0.0,
                           desc='Pressure differential as fraction of inlet pressure')

        if statics:
            if design:
                self.add_input('MN', val=0.5, desc='Exit Mach number')
            else:
                self.add_input('area', val=1.0, units='inch**2', desc='Exit flow area')

        # --- Outputs ---
        self.add_flow_output('Fl_O', statics=statics)

        if expMN > _EXPMN_THRESHOLD:
            if design:
                self.add_output('s_dPqP', val=0.0, desc='Pressure loss scalar')
            else:
                self.add_output('dPqP', val=0.0,
                                desc='Pressure differential as fraction of inlet pressure')

        # --- Register primal inputs (order defines input vector layout) ---
        self.add_primal_input('Fl_I:tot:P')
        self.add_primal_input('Fl_I:tot:h')
        self.add_primal_input('Fl_I:stat:W')
        self.add_primal_input('Fl_I:stat:MN')
        self.add_primal_input('Fl_I:tot:composition', size='dynamic')  # shape_by_conn
        self.add_primal_input('Q_dot')

        if expMN > _EXPMN_THRESHOLD:
            if design:
                self.add_primal_input('dPqP')
            else:
                self.add_primal_input('s_dPqP')
        else:
            self.add_primal_input('dPqP')

        if statics:
            if design:
                self.add_primal_input('MN')
            else:
                self.add_primal_input('area')

        # --- Register primal outputs (order must match add_output order) ---
        # add_flow_output order: total props, static props (if statics), W, FAR
        # Then we add s_dPqP/dPqP after add_flow_output
        self.add_flow_total_primal_outputs('Fl_O')

        if statics:
            self.add_flow_static_primal_outputs('Fl_O')

        self.add_primal_output('Fl_O:stat:W')

        if expMN > _EXPMN_THRESHOLD:
            if design:
                self.add_primal_output('s_dPqP')
            else:
                self.add_primal_output('dPqP')

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
        expMN = self.options['expMN']
        thermo = self.jax_thermo

        # --- Extract inputs using OpenMDAO names ---
        Pt_in = self.inp(inputs, 'Fl_I:tot:P')
        ht_in = self.inp(inputs, 'Fl_I:tot:h')
        W_in = self.inp(inputs, 'Fl_I:stat:W')
        MN_in = self.inp(inputs, 'Fl_I:stat:MN')
        composition = self.inp(inputs, 'Fl_I:tot:composition')
        Q_dot = self.inp(inputs, 'Q_dot')

        # Extract FAR from composition array
        FAR = composition[0]

        # --- Pressure loss calculation ---
        if expMN > _EXPMN_THRESHOLD:
            if design:
                dPqP = self.inp(inputs, 'dPqP')
                s_dPqP = jnp.where(MN_in > _EXPMN_THRESHOLD, dPqP / MN_in**expMN, 0.0)
            else:
                s_dPqP = self.inp(inputs, 's_dPqP')
                dPqP = s_dPqP * MN_in**expMN
        else:
            dPqP = self.inp(inputs, 'dPqP')
            s_dPqP = 0.0

        # --- Total properties ---
        Pt_out = Pt_in * (1.0 - dPqP)
        ht_out = jnp.where(W_in > _EXPMN_THRESHOLD, ht_in + Q_dot / W_in, ht_in)
        Tt_out = thermo.set_total_hP(ht_out, Pt_out, FAR)
        props = thermo.set_total_TP(Tt_out, Pt_out, FAR)

        # --- Build output list (must match add_output order from add_flow_output) ---
        # Order follows add_flow_output: total props, then static props, then W, then FAR
        # We only include primal outputs (FAR and composition are passthrough)

        # Total properties: h, T, P, rho, gamma, Cp, Cv, S, R (from _TOTAL_PROPS order)
        outputs = [
            ht_out, Tt_out, Pt_out,
            props.rho, props.gamma, props.Cp,
            props.Cv, props.S, props.R,
        ]

        # Static properties (if enabled) - come before W in add_output order
        if statics:
            if design:
                MN_exit = self.inp(inputs, 'MN')
                static_props = thermo.set_static_MN(Tt_out, Pt_out, MN_exit, W_in, FAR)
            else:
                area_exit = self.inp(inputs, 'area')
                static_props = thermo.set_static_area(Tt_out, Pt_out, area_exit, W_in, FAR)

            # Corrected flow (normalized to standard day conditions)
            Wc = W_in * jnp.sqrt(Tt_out / _T_REF) / (Pt_out / _P_REF)

            # Static properties order from _STATIC_PROPS:
            # h, T, P, rho, gamma, Cp, Cv, S, R, V, Vsonic, MN, area, Wc
            outputs.extend([
                static_props.hs, static_props.Ts, static_props.Ps,
                static_props.rhos, static_props.gamma, static_props.Cp,
                static_props.Cv, static_props.S, static_props.R,
                static_props.V, static_props.Vsonic, static_props.MN,
                static_props.area, Wc,
            ])

        # Fl_O:stat:W comes after static props in add_flow_output
        outputs.append(W_in)

        # s_dPqP or dPqP output (if expMN > 0) - added after add_flow_output
        if expMN > _EXPMN_THRESHOLD:
            outputs.append(s_dPqP if design else dPqP)

        return jnp.array(outputs)
