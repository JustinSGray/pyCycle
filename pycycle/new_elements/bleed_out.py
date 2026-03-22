"""
NewBleedOut - A bleed extraction element using JaxElement for automatic differentiation.

Extracts bleed flows from the incoming flow. Each bleed port gets a fraction
of the inlet mass flow at the same total conditions (T, P).
"""

import jax.numpy as jnp

from pycycle.new_elements.jax_element_base import JaxElement

# Standard reference conditions for corrected flow calculation
_T_REF = 518.67   # Reference temperature (degR)
_P_REF = 14.696   # Reference pressure (psi)


class NewBleedOut(JaxElement):
    """
    Bleed extraction from the incoming flow.

    Flow Stations
    -------------
    Fl_I : primary input flow
    Fl_O : primary output flow
    {bleed_name} : bleed output flows (one per name in bleed_names)

    Design: inputs are bleed frac_W values and MN (if statics)
    Off-Design: inputs are bleed frac_W values and area (if statics)
    """

    def initialize(self):
        super().initialize()

        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')
        self.options.declare('bleed_names', types=(list, tuple),
                             desc='List of names for the bleed ports',
                             default=[])

        self.default_des_od_conns = [('Fl_O:stat:area', 'area')]

    def pyc_setup_output_ports(self):
        self.copy_flow('Fl_I', 'Fl_O')
        for b_name in self.options['bleed_names']:
            self.copy_flow('Fl_I', b_name)

    def setup(self):
        design = self.options['design']
        statics = self.options['statics']
        bleeds = self.options['bleed_names']

        # --- Inputs ---
        self.add_flow_input('Fl_I')

        for bn in bleeds:
            self.add_input(f'{bn}:frac_W', val=0.0,
                           desc=f'Bleed mass flow fraction for {bn}')

        if statics:
            if design:
                self.add_input('MN', val=0.5, desc='Exit Mach number')
            else:
                self.add_input('area', val=1.0, units='inch**2', desc='Exit flow area')

        # --- Outputs ---
        # Main flow output (with statics if enabled)
        self.add_flow_output('Fl_O', statics=statics)

        # Bleed port outputs (total properties only, no statics)
        for bn in bleeds:
            self.add_flow_output(bn, statics=False)

        # Build index mappings and declare partials
        self.setup_partials()

    def compute_physics(self, inputs):
        design = self.options['design']
        statics = self.options['statics']
        bleeds = self.options['bleed_names']
        thermo = self.jax_thermo

        # --- Extract inputs ---
        Tt_in = self.inp(inputs, 'Fl_I:tot:T')
        Pt_in = self.inp(inputs, 'Fl_I:tot:P')
        W_in = self.inp(inputs, 'Fl_I:stat:W')
        composition = self.inp(inputs, 'Fl_I:tot:composition')

        # --- Compute total properties at inlet conditions ---
        # Outlet and all bleeds share the same total state as inlet
        props = thermo.set_total_TP(Tt_in, Pt_in, composition)

        # --- Main outlet total properties ---
        outputs = [
            props.h, Tt_in, Pt_in,
            props.rho, props.gamma, props.Cp,
            props.Cv, props.S, props.R,
        ]

        # --- Static properties for main outlet ---
        if statics:
            # Compute W_out after bleed extraction
            W_out = W_in
            for bn in bleeds:
                frac_W = self.inp(inputs, f'{bn}:frac_W')
                W_out = W_out - W_in * frac_W

            if design:
                MN_exit = self.inp(inputs, 'MN')
                static_props = thermo.set_static_MN(Tt_in, Pt_in, MN_exit, W_out, composition)
            else:
                area_exit = self.inp(inputs, 'area')
                static_props = thermo.set_static_area(Tt_in, Pt_in, area_exit, W_out, composition)

            Wc = W_out * jnp.sqrt(Tt_in / _T_REF) / (Pt_in / _P_REF)

            outputs.extend([
                static_props.hs, static_props.Ts, static_props.Ps,
                static_props.rhos, static_props.gamma, static_props.Cp,
                static_props.Cv, static_props.S, static_props.R,
                static_props.V, static_props.Vsonic, static_props.MN,
                static_props.area, Wc,
            ])

        # Fl_O:stat:W (mass flow out after bleeds)
        W_out = W_in
        for bn in bleeds:
            frac_W = self.inp(inputs, f'{bn}:frac_W')
            W_out = W_out - W_in * frac_W
        outputs.append(W_out)

        # --- Bleed port outputs ---
        for bn in bleeds:
            frac_W = self.inp(inputs, f'{bn}:frac_W')
            W_bld = W_in * frac_W

            # Total properties (same as inlet)
            outputs.extend([
                props.h, Tt_in, Pt_in,
                props.rho, props.gamma, props.Cp,
                props.Cv, props.S, props.R,
            ])

            # {bn}:stat:W
            outputs.append(W_bld)

        return jnp.array(outputs)
