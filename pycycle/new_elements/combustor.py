"""
NewCombustor - A combustor element using JaxElement for automatic differentiation.

Adds fuel to the incoming flow at a specified fuel-to-air ratio (FAR),
applies a pressure loss, and computes the vitiated (burned) flow properties.
"""

import jax.numpy as jnp

from pycycle.new_elements.jax_element_base import JaxElement, _TOTAL_PROPS, _STATIC_PROPS

# Standard reference conditions for corrected flow calculation
_T_REF = 518.67   # Reference temperature (degR)
_P_REF = 14.696   # Reference pressure (psi)


class NewCombustor(JaxElement):
    """
    Combustor element using functional thermodynamic interfaces with JAX derivatives.

    Mixes fuel into the incoming air flow at a specified fuel-to-air ratio,
    applies a pressure loss, and computes the vitiated total and static
    properties. The output composition includes fuel elements (e.g., H for
    hydrocarbon fuels).

    Flow Stations
    -------------
    Fl_I : incoming air flow
    Fl_O : vitiated (burned) output flow
    """

    def initialize(self):
        super().initialize()

        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')
        self.options.declare('fuel_type', default='JP-7',
                             desc='Type of fuel.')

        self.default_des_od_conns = [('Fl_O:stat:area', 'area')]

    def pyc_setup_output_ports(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        fuel_type = self.options['fuel_type']
        inflow_composition = self.Fl_I_data['Fl_I']

        # Determine the output composition (method-agnostic factory function).
        # Must use standalone function here since thermo object hasn't been
        # created yet (it needs the output composition to be set first).
        from pycycle.functional_thermo import get_mixed_output_composition
        output_composition = get_mixed_output_composition(
            thermo_method, thermo_data, inflow_composition, fuel_type)
        self.init_output_flow('Fl_O', output_composition)

    def _get_thermo_composition(self):
        """Override to use the mixed (air+fuel) composition for thermo.

        The Combustor's output flow has a different species set than the input
        (fuel adds H-containing species). The thermo must be initialized with
        the mixed composition to include all combustion product species.
        """
        if 'Fl_O' in self.Fl_O_data:
            return self.Fl_O_data['Fl_O']
        return super()._get_thermo_composition()

    def setup(self):
        design = self.options['design']
        statics = self.options['statics']
        thermo_data = self.options['thermo_data']
        fuel_type = self.options['fuel_type']

        inflow_composition = self.Fl_I_data['Fl_I']

        # Create composition mixer (handles all method-specific mixing logic)
        from pycycle.functional_thermo import create_composition_mixer
        self._mixer = create_composition_mixer(
            self.options['thermo_method'], thermo_data, inflow_composition, fuel_type)
        self._out_comp_size = self._mixer.comp_size

        # --- Inputs ---
        self.add_flow_input('Fl_I')

        # Performance optimization: exclude unused flow inputs from the primal set.
        used_flow_inputs = {'Fl_I:tot:P', 'Fl_I:tot:h', 'Fl_I:stat:W',
                            'Fl_I:tot:composition', 'Fl_I:FAR'}
        for name in list(self._primal_input_set):
            if name.startswith('Fl_I:') and name not in used_flow_inputs:
                del self._primal_input_set[name]

        # Fl_I:FAR is already added by add_flow_input
        self.add_input('dPqP', val=0.0, desc='Pressure loss as fraction of inlet total pressure')

        if statics:
            if design:
                self.add_input('MN', val=0.5, desc='Exit Mach number')
            else:
                self.add_input('area', val=1.0, units='inch**2', desc='Exit flow area')

        # --- Outputs ---
        # Manually add flow outputs because the output composition may differ
        # in size from the input composition (fuel adds new elements for CEA).
        for prop, val, units in _TOTAL_PROPS:
            kwargs = {'val': val}
            if units is not None:
                kwargs['units'] = units
            if prop == 'P':
                kwargs['lower'] = 1e-4
            self.add_output(f'Fl_O:tot:{prop}', **kwargs)

        # Composition output (mixed, possibly different size from input)
        self.add_output('Fl_O:tot:composition', val=self._mixer.base_b0)

        # Static properties
        if statics:
            for prop, val, units in _STATIC_PROPS:
                kwargs = {'val': val}
                if units is not None:
                    kwargs['units'] = units
                self.add_output(f'Fl_O:stat:{prop}', **kwargs)

        # Mass flow output (total = air + fuel)
        self.add_output('Fl_O:stat:W', val=1.0, units='lbm/s')

        # FAR output
        self.add_output('Fl_O:FAR', val=0.0)

        # Fuel mass flow
        self.add_output('Wfuel', val=0.0, units='lbm/s')

        self.setup_partials()

    def compute_physics(self, inputs):
        design = self.options['design']
        statics = self.options['statics']
        thermo = self.jax_thermo

        # --- Extract inputs ---
        Pt_in = self.inp(inputs, 'Fl_I:tot:P')
        ht_in = self.inp(inputs, 'Fl_I:tot:h')
        W_in = self.inp(inputs, 'Fl_I:stat:W')
        b0_in = self.inp(inputs, 'Fl_I:tot:composition')
        FAR = self.inp(inputs, 'Fl_I:FAR')

        # --- Fuel mixing ---
        Wfuel = W_in * FAR
        Wout = W_in + Wfuel

        # Use the mixer to compute output composition (method-agnostic)
        composition = self._mixer.mix_jax(b0_in, W_in, Wfuel)

        # --- Mass-averaged enthalpy (fuel h defaults to 0) ---
        mass_avg_h = ht_in * W_in / Wout

        # --- Pressure loss ---
        dPqP = self.inp(inputs, 'dPqP')
        Pt_out = Pt_in * (1.0 - dPqP)

        # --- Vitiated total properties ---
        Tt_out = thermo.set_total_hP(mass_avg_h, Pt_out, composition)
        props = thermo.set_total_TP(Tt_out, Pt_out, composition)

        # --- Build output vector ---
        # Total properties: h, T, P, rho, gamma, Cp, Cv, S, R
        outputs = [
            props.h, Tt_out, Pt_out,
            props.rho, props.gamma, props.Cp,
            props.Cv, props.S, props.R,
        ]

        # Composition array elements
        for i in range(self._out_comp_size):
            outputs.append(composition[i])

        # Static properties
        if statics:
            if design:
                MN_exit = self.inp(inputs, 'MN')
                static_props = thermo.set_static_MN(Tt_out, Pt_out, MN_exit, Wout, composition)
            else:
                area_exit = self.inp(inputs, 'area')
                static_props = thermo.set_static_area(Tt_out, Pt_out, area_exit, Wout, composition)

            Wc = Wout * jnp.sqrt(Tt_out / _T_REF) / (Pt_out / _P_REF)

            outputs.extend([
                static_props.hs, static_props.Ts, static_props.Ps,
                static_props.rhos, static_props.gamma, static_props.Cp,
                static_props.Cv, static_props.S, static_props.R,
                static_props.V, static_props.Vsonic, static_props.MN,
                static_props.area, Wc,
            ])

        # Fl_O:stat:W (total mass flow after mixing)
        outputs.append(Wout)

        # Fl_O:FAR
        outputs.append(FAR)

        # Wfuel
        outputs.append(Wfuel)

        return jnp.array(outputs)
