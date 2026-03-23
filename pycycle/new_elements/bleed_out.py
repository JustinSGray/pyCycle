"""
NewBleedOut - A bleed extraction element using JaxElement for automatic differentiation.

Extracts bleed flows from the incoming flow. Each bleed port gets a fraction
of the inlet mass flow at the same total conditions (T, P).
"""

import jax.numpy as jnp

from pycycle.new_elements.jax_element_base import JaxElement, _TOTAL_PROPS, _STATIC_PROPS

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

        # Performance optimization: exclude unused flow inputs from the primal set.
        # add_flow_input registers ALL ~27 flow properties as primal by default,
        # but compute_physics only reads a handful via self.inp(). Each unused
        # primal input adds a zero-column to the Jacobian and wastes a JVP
        # evaluation in jacfwd. Removing them cuts the Jacobian width significantly.
        # NOTE: if you modify compute_physics to use additional flow inputs,
        # you must add them to this set.
        used_flow_inputs = {'Fl_I:tot:T', 'Fl_I:tot:P', 'Fl_I:stat:W',
                            'Fl_I:tot:composition'}
        for name in list(self._primal_input_set):
            if name.startswith('Fl_I:') and name not in used_flow_inputs:
                del self._primal_input_set[name]

        for bn in bleeds:
            self.add_input(f'{bn}:frac_W', val=0.0,
                           desc=f'Bleed mass flow fraction for {bn}')

        if statics:
            if design:
                self.add_input('MN', val=0.5, desc='Exit Mach number')
            else:
                self.add_input('area', val=1.0, units='inch**2', desc='Exit flow area')

        # --- Outputs ---
        # Performance optimization: total properties for both the main outlet and
        # bleed ports are IDENTICAL to the inlet (same T, P, composition).
        # Instead of recomputing them via set_total_TP (which adds an expensive
        # thermo call to the JAX computation and enlarges the Jacobian), we pass
        # them through directly from the inlet with identity derivatives.
        # This eliminates the set_total_TP call entirely and removes
        # (1 + n_bleeds) × 9 = many primal outputs from the Jacobian.

        # Main outlet total properties — passthrough from Fl_I
        if not hasattr(self, '_passthrough_vars'):
            self._passthrough_vars = []

        for prop, val, units in _TOTAL_PROPS:
            kwargs = {'val': val, 'primal': False}
            if units is not None:
                kwargs['units'] = units
            if prop == 'P':
                kwargs['lower'] = 1e-4
            self.add_output(f'Fl_O:tot:{prop}', **kwargs)
            self._passthrough_vars.append((f'Fl_I:tot:{prop}', f'Fl_O:tot:{prop}'))

        # Fl_O composition and FAR — passthrough
        self.add_output('Fl_O:tot:composition',
                        copy_shape='Fl_I:tot:composition', primal=False)
        self._passthrough_vars.append(('Fl_I:tot:composition', 'Fl_O:tot:composition'))

        # Static properties (computed by JAX — these DO depend on W_out)
        if statics:
            for prop, val, units in _STATIC_PROPS:
                kwargs = {'val': val}
                if units is not None:
                    kwargs['units'] = units
                self.add_output(f'Fl_O:stat:{prop}', **kwargs)

        # W_out (primal — computed from W_in minus bleed extractions)
        self.add_output('Fl_O:stat:W', val=1.0, units='lbm/s')

        self.add_output('Fl_O:FAR', val=0.0, primal=False)
        self._passthrough_vars.append(('Fl_I:FAR', 'Fl_O:FAR'))

        # Bleed port outputs — total properties are passthroughs, W is primal
        for bn in bleeds:
            for prop, val, units in _TOTAL_PROPS:
                kwargs = {'val': val, 'primal': False}
                if units is not None:
                    kwargs['units'] = units
                self.add_output(f'{bn}:tot:{prop}', **kwargs)
                self._passthrough_vars.append((f'Fl_I:tot:{prop}', f'{bn}:tot:{prop}'))

            self.add_output(f'{bn}:tot:composition',
                            copy_shape='Fl_I:tot:composition', primal=False)
            self._passthrough_vars.append(('Fl_I:tot:composition', f'{bn}:tot:composition'))

            self.add_output(f'{bn}:stat:W', val=1.0, units='lbm/s')

            self.add_output(f'{bn}:FAR', val=0.0, primal=False)
            self._passthrough_vars.append(('Fl_I:FAR', f'{bn}:FAR'))

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

        # --- Compute W_out after bleed extraction ---
        W_out = W_in
        for bn in bleeds:
            frac_W = self.inp(inputs, f'{bn}:frac_W')
            W_out = W_out - W_in * frac_W

        outputs = []

        # --- Static properties for main outlet ---
        # (total properties are handled as passthroughs, not computed here)
        if statics:
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

        # Fl_O:stat:W
        outputs.append(W_out)

        # --- Bleed port W outputs ---
        for bn in bleeds:
            frac_W = self.inp(inputs, f'{bn}:frac_W')
            outputs.append(W_in * frac_W)

        return jnp.array(outputs)
