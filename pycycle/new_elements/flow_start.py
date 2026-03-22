"""
NewFlowStart - A flow start element using JaxElement for automatic differentiation.

Creates a flow from specified total conditions (T, P) and flow parameters (W, MN).
Optionally mixes in a reactant (e.g., water) at a specified mass ratio.
"""

import numpy as np
import jax.numpy as jnp

from pycycle.new_elements.jax_element_base import JaxElement, _TOTAL_PROPS, _STATIC_PROPS
from pycycle.constants import THERMO_DEFAULT_COMPOSITIONS

# Standard reference conditions for corrected flow calculation
_T_REF = 518.67   # Reference temperature (degR)
_P_REF = 14.696   # Reference pressure (psi)


class NewFlowStart(JaxElement):
    """
    Flow start element using functional thermodynamic interfaces with JAX derivatives.

    Creates a flow at specified total temperature and pressure conditions with
    a given Mach number and mass flow rate. Optionally mixes in a reactant
    (e.g., water for WAR) at a specified mass ratio that modifies the flow composition.
    """

    def initialize(self):
        super().initialize()

        self.options.declare('composition', default=None,
                             desc='Composition of the flow. None uses the default for the thermo package.')
        self.options.declare('reactant', default=False, types=(bool, str),
                             desc='If False, flow matches base composition. If a string, that reactant '
                                  'is mixed into the flow at the ratio set by the mix_ratio input.')
        self.options.declare('mix_ratio_name', default='mix:ratio',
                             desc='The name of the input that governs the mix ratio of the reactant.')

    def pyc_setup_output_ports(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        composition = self.options['composition']
        reactant = self.options['reactant']

        if reactant is not False:
            from pycycle.thermo.thermo import ThermoAdd
            self._thermo_add_obj = ThermoAdd(method=thermo_method, mix_mode='reactant',
                                             thermo_kwargs={'spec': thermo_data,
                                                            'inflow_composition': composition,
                                                            'mix_composition': reactant})
            self.init_output_flow('Fl_O', self._thermo_add_obj.output_port_data())
        else:
            if composition is None:
                composition = THERMO_DEFAULT_COMPOSITIONS[thermo_method]
            self.init_output_flow('Fl_O', composition)

    def setup(self):
        reactant = self.options['reactant']
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']

        composition = self.Fl_O_data['Fl_O']

        # Compute composition array and its size
        if thermo_method == 'CEA':
            from pycycle.thermo.cea.species_data import Properties
            mixed_props = Properties(thermo_data, init_elements=composition)
            self._composition_b0 = mixed_props.b0.copy()
            self._comp_size = len(mixed_props.b0)

            if reactant is not False:
                # Precompute constants for reactant mixing in compute_physics
                inflow_composition = self.options['composition']
                if inflow_composition is None:
                    inflow_composition = THERMO_DEFAULT_COMPOSITIONS['CEA']
                inflow_props = Properties(thermo_data, init_elements=inflow_composition)

                # Remap inflow b0 to mixed element ordering
                in_out_map = np.zeros((self._comp_size, len(inflow_props.b0)))
                for i, e in enumerate(inflow_props.elements):
                    j = mixed_props.elements.index(e)
                    in_out_map[j, i] = 1.0

                b0_remapped = in_out_map @ inflow_props.b0
                b0_mass = b0_remapped * mixed_props.element_wt
                self._b0_base_mass_norm = b0_mass / np.sum(b0_mass)

                # Fuel composition for 1kg
                mix_comp = reactant if isinstance(reactant, str) else reactant
                if isinstance(mix_comp, str):
                    mix_comp = (mix_comp,)

                self._init_fuel_1kg = np.zeros(self._comp_size)
                for reactant_name in mix_comp:
                    for i, e in enumerate(mixed_props.elements):
                        self._init_fuel_1kg[i] = (thermo_data.reactants[reactant_name].get(e, 0)
                                                  * thermo_data.element_wts[e])
                self._init_fuel_1kg /= np.sum(self._init_fuel_1kg)

                self._mixed_wt_mole = mixed_props.element_wt.copy()

        else:  # TABULAR
            comp_values = list(composition.values())
            self._composition_b0 = np.array(comp_values)
            self._comp_size = len(comp_values)

        # --- Inputs ---
        self.add_input('T', val=518., units='degR')
        self.add_input('P', val=14.696, units='lbf/inch**2')
        self.add_input('W', val=1.0, units='lbm/s')
        self.add_input('MN', val=0.5)

        if reactant is not False:
            mix_ratio_name = self.options['mix_ratio_name']
            self.add_input(mix_ratio_name, val=0.0)

        # --- Outputs ---
        # Total properties
        for prop, val, units in _TOTAL_PROPS:
            kwargs = {'val': val}
            if units is not None:
                kwargs['units'] = units
            if prop == 'P':
                kwargs['lower'] = 1e-4
            self.add_output(f'Fl_O:tot:{prop}', **kwargs)

        # Composition output (known size from setup)
        self.add_output('Fl_O:tot:composition', val=self._composition_b0)

        # Static properties
        for prop, val, units in _STATIC_PROPS:
            kwargs = {'val': val}
            if units is not None:
                kwargs['units'] = units
            self.add_output(f'Fl_O:stat:{prop}', **kwargs)

        # Mass flow output
        self.add_output('Fl_O:stat:W', val=1.0, units='lbm/s')

        # FAR output (always 0 for FlowStart, not part of JAX computation)
        self.add_output('Fl_O:FAR', val=0.0, primal=False)

        self.setup_partials()

    def compute_physics(self, inputs):
        thermo = self.jax_thermo
        reactant = self.options['reactant']

        T = self.inp(inputs, 'T')
        P = self.inp(inputs, 'P')
        W = self.inp(inputs, 'W')
        MN = self.inp(inputs, 'MN')

        # Compute composition
        if reactant is not False:
            mix_ratio_name = self.options['mix_ratio_name']
            ratio = self.inp(inputs, mix_ratio_name)

            # Mix base composition with reactant (all constants except W and ratio)
            b0_base = jnp.array(self._b0_base_mass_norm)
            fuel_1kg = jnp.array(self._init_fuel_1kg)
            wt_mole = jnp.array(self._mixed_wt_mole)

            b0_out = b0_base * W + fuel_1kg * W * ratio
            b0_out = b0_out / jnp.sum(b0_out)
            composition = b0_out / wt_mole
        else:
            composition = jnp.array(self._composition_b0)

        # Total properties from T, P
        props = thermo.set_total_TP(T, P, composition)

        outputs = [
            props.h, T, P,
            props.rho, props.gamma, props.Cp,
            props.Cv, props.S, props.R,
        ]

        # Composition array elements
        for i in range(self._comp_size):
            outputs.append(composition[i])

        # Static properties
        static_props = thermo.set_static_MN(T, P, MN, W, composition)

        Wc = W * jnp.sqrt(T / _T_REF) / (P / _P_REF)

        outputs.extend([
            static_props.hs, static_props.Ts, static_props.Ps,
            static_props.rhos, static_props.gamma, static_props.Cp,
            static_props.Cv, static_props.S, static_props.R,
            static_props.V, static_props.Vsonic, static_props.MN,
            static_props.area, Wc,
        ])

        # W output
        outputs.append(W)

        return jnp.array(outputs)
