"""
NewPerformance - Engine performance calculations using JaxElement.

Replaces the old Performance ExplicitComponent with JAX-based automatic
differentiation instead of hand-coded partials.
"""

import jax.numpy as jnp

from pycycle.new_elements.jax_element_base import JaxElement


class NewPerformance(JaxElement):
    """
    Calculates overall engine performance parameters: OPR, thrust, fuel consumption.

    Supports variable numbers of nozzles and burners via options.
    """

    def initialize(self):
        super().initialize()

        self.options.declare('num_nozzles', default=1, types=int)
        self.options.declare('num_burners', default=1, types=int)

    def setup(self):
        num_nozzles = self.options['num_nozzles']
        num_burners = self.options['num_burners']

        # --- Inputs ---
        self.add_input('Pt2', val=14.696, units='lbf/inch**2',
                       desc='Pressure at inlet of first compressor')
        self.add_input('Pt3', val=14.696, units='lbf/inch**2',
                       desc='Pressure at exit of last compressor')
        self.add_input('ram_drag', val=0.0, units='lbf',
                       desc='Ram drag from inlet')
        self.add_input('power', val=1.0, units='hp',
                       desc='Shaft power')

        # Dynamic nozzle thrust inputs
        for i in range(num_nozzles):
            self.add_input(f'Fg_{i}', val=0.0, units='lbf',
                           desc=f'Gross thrust from nozzle {i}')

        # Dynamic burner fuel flow inputs
        for i in range(num_burners):
            self.add_input(f'Wfuel_{i}', val=0.0, units='lbm/s',
                           desc=f'Fuel flow rate entering combustor {i}')

        # --- Outputs ---
        self.add_output('OPR', val=1.0,
                        desc='Overall pressure ratio, Pt3/Pt2')
        self.add_output('Fg', val=10000.0, units='lbf',
                        desc='Gross thrust of all nozzles')
        self.add_output('Fn', val=10000.0, units='lbf',
                        desc='Net thrust of the engine')

        if num_burners > 0:
            self.add_output('TSFC', val=1.0, units='lbm/(h*lbf)',
                            desc='Thrust specific fuel consumption')
            self.add_output('PSFC', val=1.0, units='lbm/(h*lbf)',
                            desc='Power specific fuel consumption')
            self.add_output('Wfuel', val=0.001, units='lbm/s',
                            desc='Total fuel flow rate')

        self.setup_partials()

    def compute_physics(self, inputs):
        num_nozzles = self.options['num_nozzles']
        num_burners = self.options['num_burners']

        Pt2 = self.inp(inputs, 'Pt2')
        Pt3 = self.inp(inputs, 'Pt3')
        ram_drag = self.inp(inputs, 'ram_drag')
        power = self.inp(inputs, 'power')

        OPR = Pt3 / Pt2

        # Sum gross thrust from all nozzles
        Fg = 0.0
        for i in range(num_nozzles):
            Fg = Fg + self.inp(inputs, f'Fg_{i}')

        Fn = Fg - ram_drag

        outputs = [OPR, Fg, Fn]

        if num_burners > 0:
            # Sum fuel flow from all burners
            Wfuel = 0.0
            for i in range(num_burners):
                Wfuel = Wfuel + self.inp(inputs, f'Wfuel_{i}')

            TSFC = Wfuel * 3600.0 / (Fn + 1e-10)
            PSFC = Wfuel * 3600.0 / power

            outputs.extend([TSFC, PSFC, Wfuel])

        return jnp.array(outputs)
