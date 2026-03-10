"""
NewShaft - A shaft element using JaxElement for automatic differentiation.

This is a single ExplicitComponent that replaces the old Shaft ExplicitComponent.
It uses JAX for automatic derivatives instead of hand-coded partials.
"""

import jax.numpy as jnp

from pycycle.new_elements.jax_element_base import JaxElement

# Conversion factor: RPM * ft*lbf -> hp
# 2*pi/60 converts RPM to rad/s, then divide by 550 ft*lbf/s per hp
_HP_PER_RPM_FT_LBF = 2.0 * jnp.pi / 60.0 / 550.0


class NewShaft(JaxElement):
    """
    Shaft element that calculates power balance for a rotating shaft.

    Sums torques from multiple ports, applies fractional loss and
    auxiliary power extraction (HPX), and computes net torque and power.
    """

    def initialize(self):
        super().initialize()

        self.options.declare('num_ports', default=2,
                             desc='Number of shaft connections to make')

    def setup(self):
        num_ports = self.options['num_ports']

        # --- Inputs ---
        self.add_input('Nmech', val=1000.0, units='rpm')
        self.add_input('HPX', val=0.0, units='hp')
        self.add_input('fracLoss', val=0.0)

        # Dynamic torque inputs (one per port)
        for i in range(num_ports):
            self.add_input(f'trq_{i}', val=0.0, units='ft*lbf')

        # --- Outputs ---
        self.add_output('trq_in', val=1.0, units='ft*lbf')
        self.add_output('trq_out', val=1.0, units='ft*lbf')
        self.add_output('trq_net', val=1.0, units='ft*lbf')
        self.add_output('pwr_in', val=1.0, units='hp')
        self.add_output('pwr_in_real', val=1.0, units='hp')
        self.add_output('pwr_out', val=1.0, units='hp')
        self.add_output('pwr_out_real', val=1.0, units='hp')
        self.add_output('pwr_net', val=1.0, units='hp')

        # Build index mappings and declare partials
        self.setup_partials()

    def compute_physics(self, inputs):
        """
        Pure JAX physics computation for shaft power balance.

        Parameters
        ----------
        inputs : jnp.ndarray
            Flat input vector. Use self.inp(inputs, 'name') to access values.

        Returns
        -------
        jnp.ndarray
            Flat output vector matching add_output order.
        """
        num_ports = self.options['num_ports']

        Nmech = self.inp(inputs, 'Nmech')
        HPX = self.inp(inputs, 'HPX')
        fracLoss = self.inp(inputs, 'fracLoss')

        # Sum positive torques (in) and negative torques (out)
        trq_in = 0.0
        trq_out = 0.0
        for i in range(num_ports):
            trq = self.inp(inputs, f'trq_{i}')
            trq_in = jnp.where(trq >= 0, trq_in + trq, trq_in)
            trq_out = jnp.where(trq < 0, trq_out + trq, trq_out)

        convert = _HP_PER_RPM_FT_LBF

        trq_net = trq_in * (1.0 - fracLoss) + trq_out - HPX / (Nmech * convert)
        pwr_in = trq_in * Nmech * convert
        pwr_in_real = trq_in * (1.0 - fracLoss) * Nmech * convert
        pwr_out = trq_out * Nmech * convert
        pwr_out_real = trq_out * Nmech * convert - HPX
        pwr_net = trq_net * Nmech * convert

        # Return outputs in add_output order
        return jnp.array([
            trq_in, trq_out, trq_net,
            pwr_in, pwr_in_real, pwr_out, pwr_out_real, pwr_net,
        ])
