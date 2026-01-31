"""
Functional interface for mixing thermodynamic streams.

This module provides a functional interface for computing mixed flow properties
when combining an inflow stream with reactants (like fuel) or other flow streams.
"""

import numpy as np
from collections import namedtuple

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from pycycle.constants import CEA_AIR_COMPOSITION
from pycycle.thermo.cea.species_data import Properties, janaf


# Named tuple for returning mixed flow results (matches OpenMDAO outputs)
ThermoAddOutput = namedtuple('ThermoAddOutput', [
    'mass_avg_h',       # Mass-averaged enthalpy
    'Wout',             # Total mass flow rate out
    'composition_out',  # Mixed composition (b0 array)
    'W_mix',            # Array of mass flow rates for each mix stream, ordered by mix_names
])


class ThermoAdd:
    """
    Functional interface for mixing thermodynamic streams.

    This class computes the mixed composition and mass-averaged enthalpy
    when combining an inflow stream with reactants or other flow streams.

    Parameters
    ----------
    inflow_composition : dict, optional
        Elemental composition of the inflow. Default is CEA_AIR_COMPOSITION.
    mix_mode : str, optional
        'reactant' to mix by fuel-to-air ratio, 'flow' to mix flow streams.
        Default is 'reactant'.
    mix_composition : str, dict, or list, optional
        For 'reactant' mode: reactant name(s) like 'JP-7'.
        For 'flow' mode: elemental composition dict(s).
        Default is 'JP-7'.
    mix_names : str or list, optional
        Name(s) for the mix streams. Default is 'mix'.
    thermo_data : module, optional
        Thermodynamic data module. Default is janaf.

    Examples
    --------
    Reactant mode (adding fuel):

    >>> mixer = ThermoAdd(mix_mode='reactant', mix_composition='JP-7')
    >>> result = mixer.compute(W=38.8, h=181.38, composition=air_b0,
    ...                        ratio={'mix': 0.02673}, h_mix={'mix': 0.0})

    Flow mode (mixing two streams):

    >>> mixer = ThermoAdd(mix_mode='flow',
    ...                   mix_composition=other_composition,
    ...                   mix_names='bleed')
    >>> result = mixer.compute(W=62.0, h=10.0, composition=main_b0,
    ...                        W_mix={'bleed': 4.5}, h_mix={'bleed': 5.0},
    ...                        composition_mix={'bleed': bleed_b0})
    """

    def __init__(self, inflow_composition=None, mix_mode='reactant',
                 mix_composition='JP-7', mix_names='mix', thermo_data=None):

        if inflow_composition is None:
            inflow_composition = CEA_AIR_COMPOSITION
        if thermo_data is None:
            thermo_data = janaf

        self.thermo_data = thermo_data
        self.mix_mode = mix_mode

        # Normalize mix_composition and mix_names to tuples
        if isinstance(mix_composition, (str, dict)):
            mix_composition = (mix_composition,)
        if isinstance(mix_names, str):
            mix_names = (mix_names,)

        self.mix_composition = mix_composition
        self.mix_names = mix_names

        # Compute the output element set (union of inflow and mix elements)
        mixed_elements = inflow_composition.copy()
        if mix_mode == 'reactant':
            for reactant in mix_composition:
                mixed_elements.update(thermo_data.reactants[reactant])
        else:  # flow mode
            for flow_elements in mix_composition:
                mixed_elements.update(flow_elements)

        self.mixed_elements = mixed_elements

        # Create Properties objects for inflow and mixed output
        self.inflow_thermo = Properties(thermo_data, init_elements=inflow_composition)
        self.mixed_thermo = Properties(thermo_data, init_elements=mixed_elements)

        # Store useful attributes
        self.inflow_elements = self.inflow_thermo.elements
        self.inflow_wt_mole = self.inflow_thermo.element_wt
        self.num_inflow_elements = len(self.inflow_elements)

        self.output_elements = self.mixed_thermo.elements
        self.output_wt_mole = self.mixed_thermo.element_wt
        self.num_output_elements = len(self.output_elements)

        # Create mapping matrix from inflow composition to output composition
        self.in_out_map = np.zeros((self.num_output_elements, self.num_inflow_elements))
        for i, e in enumerate(self.inflow_elements):
            j = self.output_elements.index(e)
            self.in_out_map[j, i] = 1.0

        # Pre-compute fuel amounts per kg for reactant mode
        if mix_mode == 'reactant':
            self.fuel_amounts_1kg = {}
            for reactant in mix_composition:
                amounts = np.zeros(self.num_output_elements)
                for i, e in enumerate(self.output_elements):
                    amounts[i] = thermo_data.reactants[reactant].get(e, 0) * thermo_data.element_wts[e]
                # Normalize to 1 kg of fuel
                amounts /= np.sum(amounts)
                self.fuel_amounts_1kg[reactant] = amounts
        else:
            # Flow mode: create mapping matrices for each mix stream
            self.mix_thermos = {}
            self.mix_wt_mole = {}
            self.mix_out_maps = {}

            for name, elements in zip(mix_names, mix_composition):
                thermo = Properties(thermo_data, init_elements=elements)
                self.mix_thermos[name] = thermo
                self.mix_wt_mole[name] = thermo.element_wt

                # Mapping from mix composition to output composition
                mix_map = np.zeros((self.num_output_elements, thermo.num_element))
                for i, e in enumerate(thermo.elements):
                    j = self.output_elements.index(e)
                    mix_map[j, i] = 1.0
                self.mix_out_maps[name] = mix_map

    @property
    def b0(self):
        """Return the default composition (b0) for the mixed output."""
        return self.mixed_thermo.b0

    @property
    def inflow_b0(self):
        """Return the default composition (b0) for the inflow."""
        return self.inflow_thermo.b0

    def compute(self, W, h, composition, ratio=None, W_mix=None, h_mix=None, composition_mix=None):
        """
        Compute mixed flow properties.

        Parameters
        ----------
        W : float
            Inflow mass flow rate
        h : float
            Inflow total enthalpy
        composition : ndarray
            Inflow composition (b0 array from thermo Properties)
        ratio : dict, optional
            For reactant mode: {name: ratio} where ratio is reactant-to-inflow mass ratio
        W_mix : dict, optional
            For flow mode: {name: mass_flow} for each mix stream
        h_mix : dict, optional
            Enthalpy of each mix stream. Required for both modes.
        composition_mix : dict, optional
            For flow mode: {name: b0_array} composition of each mix stream

        Returns
        -------
        ThermoAddOutput
            Named tuple with (mass_avg_h, Wout, composition_out, W_mix).
            W_mix is an array ordered by self.mix_names.
        """
        if h_mix is None:
            h_mix = {name: 0.0 for name in self.mix_names}

        # Map inflow composition to output element set
        b0_out = self.in_out_map @ composition

        # Convert to mass units, normalize to 1 kg, then scale to inflow mass
        b0_out *= self.output_wt_mole
        b0_out /= np.sum(b0_out)
        b0_out *= W

        # Initialize mass-averaged enthalpy and total mass flow
        mass_avg_h = h * W
        Wout = float(W)

        # Array to store computed mix mass flows (ordered by mix_names)
        W_mix_out = np.zeros(len(self.mix_names))

        if self.mix_mode == 'reactant':
            if ratio is None:
                ratio = {name: 0.0 for name in self.mix_names}

            for idx, (name, reactant) in enumerate(zip(self.mix_names, self.mix_composition)):
                r = ratio.get(name, 0.0)
                W_reactant = W * r
                W_mix_out[idx] = W_reactant

                # Add reactant contribution to composition
                b0_out += self.fuel_amounts_1kg[reactant] * W_reactant

                # Add to mass-averaged enthalpy
                mass_avg_h += h_mix.get(name, 0.0) * W_reactant
                Wout += W_reactant

        else:  # flow mode
            if W_mix is None:
                W_mix = {name: 0.0 for name in self.mix_names}
            if composition_mix is None:
                composition_mix = {name: self.mix_thermos[name].b0 for name in self.mix_names}

            for idx, name in enumerate(self.mix_names):
                W_stream = W_mix.get(name, 0.0)
                W_mix_out[idx] = W_stream
                comp = composition_mix.get(name)

                # Convert mix composition to mass units
                mix_mass = comp.copy() * self.mix_wt_mole[name]
                mix_mass /= np.sum(mix_mass)  # Normalize to 1 kg
                mix_mass *= W_stream  # Scale to actual mass flow

                # Map to output element set and add
                b0_out += self.mix_out_maps[name] @ mix_mass

                # Add to mass-averaged enthalpy
                mass_avg_h += h_mix.get(name, 0.0) * W_stream
                Wout += W_stream

        # Normalize output composition back to per-kg basis
        b0_out /= np.sum(b0_out)
        composition_out = b0_out / self.output_wt_mole

        # Compute mass-averaged enthalpy
        mass_avg_h /= Wout

        return ThermoAddOutput(
            mass_avg_h=mass_avg_h,
            Wout=Wout,
            composition_out=composition_out,
            W_mix=W_mix_out
        )

    # =========================================================================
    # JAX-compatible methods for automatic differentiation
    # =========================================================================

    def _compute_jax(self, W, h, composition, ratio_or_W_mix, h_mix, composition_mix_flat=None):
        """
        JAX-traceable compute method using array inputs only.

        Parameters
        ----------
        W : scalar
            Inflow mass flow rate
        h : scalar
            Inflow total enthalpy
        composition : array (num_inflow_elements,)
            Inflow composition
        ratio_or_W_mix : array (num_mix,)
            For reactant mode: ratios. For flow mode: W_mix values.
        h_mix : array (num_mix,)
            Enthalpies of mix streams
        composition_mix_flat : array, optional
            For flow mode only: flattened mix compositions.
            Shape: (num_mix * num_mix_elements,) where num_mix_elements varies per stream.

        Returns
        -------
        tuple
            (mass_avg_h, Wout, composition_out, W_mix_out)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for _compute_jax. Install with: pip install jax jaxlib")

        # Use JAX numpy
        xp = jnp

        # Map inflow composition to output element set
        b0_out = xp.asarray(self.in_out_map) @ composition

        # Convert to mass units, normalize to 1 kg, then scale to inflow mass
        b0_out = b0_out * xp.asarray(self.output_wt_mole)
        b0_out = b0_out / xp.sum(b0_out)
        b0_out = b0_out * W

        # Initialize mass-averaged enthalpy and total mass flow
        mass_avg_h = h * W
        Wout = W

        # Array to store computed mix mass flows
        num_mix = len(self.mix_names)
        W_mix_out = xp.zeros(num_mix)

        if self.mix_mode == 'reactant':
            for idx, (name, reactant) in enumerate(zip(self.mix_names, self.mix_composition)):
                r = ratio_or_W_mix[idx]
                W_reactant = W * r
                W_mix_out = W_mix_out.at[idx].set(W_reactant)

                # Add reactant contribution to composition
                b0_out = b0_out + xp.asarray(self.fuel_amounts_1kg[reactant]) * W_reactant

                # Add to mass-averaged enthalpy
                mass_avg_h = mass_avg_h + h_mix[idx] * W_reactant
                Wout = Wout + W_reactant

        else:  # flow mode
            # For flow mode, we need to handle composition_mix
            # composition_mix_flat contains flattened compositions for each mix stream
            offset = 0
            for idx, name in enumerate(self.mix_names):
                W_stream = ratio_or_W_mix[idx]
                W_mix_out = W_mix_out.at[idx].set(W_stream)

                # Get mix composition from flattened array
                mix_num_elem = len(self.mix_wt_mole[name])
                comp = composition_mix_flat[offset:offset + mix_num_elem]
                offset += mix_num_elem

                # Convert mix composition to mass units
                mix_mass = comp * xp.asarray(self.mix_wt_mole[name])
                mix_mass = mix_mass / xp.sum(mix_mass)  # Normalize to 1 kg
                mix_mass = mix_mass * W_stream  # Scale to actual mass flow

                # Map to output element set and add
                b0_out = b0_out + xp.asarray(self.mix_out_maps[name]) @ mix_mass

                # Add to mass-averaged enthalpy
                mass_avg_h = mass_avg_h + h_mix[idx] * W_stream
                Wout = Wout + W_stream

        # Normalize output composition back to per-kg basis
        b0_out = b0_out / xp.sum(b0_out)
        composition_out = b0_out / xp.asarray(self.output_wt_mole)

        # Compute mass-averaged enthalpy
        mass_avg_h = mass_avg_h / Wout

        return mass_avg_h, Wout, composition_out, W_mix_out

    def _flatten_inputs(self, W, h, composition, ratio=None, W_mix=None, h_mix=None, composition_mix=None):
        """Convert dict-based inputs to flat arrays for JAX."""
        if h_mix is None:
            h_mix = {name: 0.0 for name in self.mix_names}

        h_mix_arr = np.array([h_mix.get(name, 0.0) for name in self.mix_names])

        if self.mix_mode == 'reactant':
            if ratio is None:
                ratio = {name: 0.0 for name in self.mix_names}
            ratio_arr = np.array([ratio.get(name, 0.0) for name in self.mix_names])
            return W, h, np.asarray(composition), ratio_arr, h_mix_arr, None
        else:
            if W_mix is None:
                W_mix = {name: 0.0 for name in self.mix_names}
            if composition_mix is None:
                composition_mix = {name: self.mix_thermos[name].b0 for name in self.mix_names}

            W_mix_arr = np.array([W_mix.get(name, 0.0) for name in self.mix_names])

            # Flatten composition_mix
            comp_mix_flat = np.concatenate([
                np.asarray(composition_mix.get(name, self.mix_thermos[name].b0))
                for name in self.mix_names
            ])

            return W, h, np.asarray(composition), W_mix_arr, h_mix_arr, comp_mix_flat

    def linearize(self, W, h, composition, ratio=None, W_mix=None, h_mix=None, composition_mix=None):
        """
        Compute and cache Jacobians at the given operating point using JAX.

        Must be called before using jvp() or vjp().

        Parameters
        ----------
        Same as compute()
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for linearize(). Install with: pip install jax jaxlib")

        # Store the linearization point
        self._lin_W = W
        self._lin_h = h
        self._lin_composition = np.asarray(composition)

        # Flatten inputs
        W, h, composition, ratio_or_W_mix, h_mix_arr, comp_mix_flat = self._flatten_inputs(
            W, h, composition, ratio, W_mix, h_mix, composition_mix
        )

        self._lin_ratio_or_W_mix = ratio_or_W_mix
        self._lin_h_mix = h_mix_arr
        self._lin_comp_mix_flat = comp_mix_flat

        # Define the function to differentiate
        if self.mix_mode == 'reactant':
            def f(W, h, composition, ratio_arr, h_mix_arr):
                mass_avg_h, Wout, composition_out, W_mix_out = self._compute_jax(
                    W, h, composition, ratio_arr, h_mix_arr, None
                )
                # Return as flat array for Jacobian computation
                return jnp.concatenate([
                    jnp.array([mass_avg_h, Wout]),
                    composition_out,
                    W_mix_out
                ])

            # Compute Jacobians with respect to each input using jacfwd
            # For scalar inputs, jacfwd returns shape (num_outputs,)
            self._jac_W = jax.jacfwd(
                lambda W: f(W, h, composition, ratio_or_W_mix, h_mix_arr)
            )(W)
            self._jac_h = jax.jacfwd(
                lambda h: f(W, h, composition, ratio_or_W_mix, h_mix_arr)
            )(h)

            # Full Jacobian for array inputs: shape (num_outputs, input_size)
            self._jac_composition = jax.jacfwd(
                lambda comp: f(W, h, comp, ratio_or_W_mix, h_mix_arr)
            )(composition)

            self._jac_ratio = jax.jacfwd(
                lambda r: f(W, h, composition, r, h_mix_arr)
            )(ratio_or_W_mix)

            self._jac_h_mix = jax.jacfwd(
                lambda hm: f(W, h, composition, ratio_or_W_mix, hm)
            )(h_mix_arr)

        else:  # flow mode
            def f(W, h, composition, W_mix_arr, h_mix_arr, comp_mix_flat):
                mass_avg_h, Wout, composition_out, W_mix_out = self._compute_jax(
                    W, h, composition, W_mix_arr, h_mix_arr, comp_mix_flat
                )
                return jnp.concatenate([
                    jnp.array([mass_avg_h, Wout]),
                    composition_out,
                    W_mix_out
                ])

            # Compute Jacobians with respect to each input using jacfwd
            self._jac_W = jax.jacfwd(
                lambda W: f(W, h, composition, ratio_or_W_mix, h_mix_arr, comp_mix_flat)
            )(W)
            self._jac_h = jax.jacfwd(
                lambda h: f(W, h, composition, ratio_or_W_mix, h_mix_arr, comp_mix_flat)
            )(h)

            self._jac_composition = jax.jacfwd(
                lambda comp: f(W, h, comp, ratio_or_W_mix, h_mix_arr, comp_mix_flat)
            )(composition)

            self._jac_W_mix = jax.jacfwd(
                lambda wm: f(W, h, composition, wm, h_mix_arr, comp_mix_flat)
            )(ratio_or_W_mix)

            self._jac_h_mix = jax.jacfwd(
                lambda hm: f(W, h, composition, ratio_or_W_mix, hm, comp_mix_flat)
            )(h_mix_arr)

            self._jac_comp_mix = jax.jacfwd(
                lambda cm: f(W, h, composition, ratio_or_W_mix, h_mix_arr, cm)
            )(comp_mix_flat)

        # Store output sizes for unpacking
        self._num_outputs = 2 + self.num_output_elements + len(self.mix_names)

    def _unpack_output_tangent(self, out_dot):
        """Unpack flattened output tangent to named components."""
        idx = 0
        mass_avg_h_dot = out_dot[idx]
        idx += 1
        Wout_dot = out_dot[idx]
        idx += 1
        composition_out_dot = out_dot[idx:idx + self.num_output_elements]
        idx += self.num_output_elements
        W_mix_dot = out_dot[idx:]
        return mass_avg_h_dot, Wout_dot, composition_out_dot, W_mix_dot

    def jvp(self, W_dot=0.0, h_dot=0.0, composition_dot=None, ratio_dot=None,
            W_mix_dot=None, h_mix_dot=None, composition_mix_dot=None):
        """
        Compute Jacobian-vector product (forward-mode autodiff).

        Computes the directional derivative of outputs in the direction
        specified by the tangent vectors. Must call linearize() first.

        Parameters
        ----------
        W_dot : float
            Tangent for W
        h_dot : float
            Tangent for h
        composition_dot : array, optional
            Tangent for composition
        ratio_dot : dict, optional
            Tangent for ratio (reactant mode)
        W_mix_dot : dict, optional
            Tangent for W_mix (flow mode)
        h_mix_dot : dict, optional
            Tangent for h_mix
        composition_mix_dot : dict, optional
            Tangent for composition_mix (flow mode)

        Returns
        -------
        dict
            Tangents for outputs: {'mass_avg_h': ..., 'Wout': ...,
                                   'composition_out': ..., 'W_mix': ...}
        """
        if not hasattr(self, '_jac_W'):
            raise RuntimeError("Must call linearize() before jvp()")

        # Initialize output tangent
        out_dot = np.zeros(self._num_outputs)

        # Contribution from W
        out_dot += float(W_dot) * np.asarray(self._jac_W)

        # Contribution from h
        out_dot += float(h_dot) * np.asarray(self._jac_h)

        # Contribution from composition
        if composition_dot is not None:
            out_dot += np.asarray(self._jac_composition) @ np.asarray(composition_dot)

        # Contribution from h_mix
        if h_mix_dot is not None:
            h_mix_dot_arr = np.array([h_mix_dot.get(name, 0.0) for name in self.mix_names])
            out_dot += np.asarray(self._jac_h_mix) @ h_mix_dot_arr

        if self.mix_mode == 'reactant':
            # Contribution from ratio
            if ratio_dot is not None:
                ratio_dot_arr = np.array([ratio_dot.get(name, 0.0) for name in self.mix_names])
                out_dot += np.asarray(self._jac_ratio) @ ratio_dot_arr
        else:
            # Contribution from W_mix
            if W_mix_dot is not None:
                W_mix_dot_arr = np.array([W_mix_dot.get(name, 0.0) for name in self.mix_names])
                out_dot += np.asarray(self._jac_W_mix) @ W_mix_dot_arr

            # Contribution from composition_mix
            if composition_mix_dot is not None:
                comp_mix_dot_flat = np.concatenate([
                    np.asarray(composition_mix_dot.get(name, np.zeros(len(self.mix_wt_mole[name]))))
                    for name in self.mix_names
                ])
                out_dot += np.asarray(self._jac_comp_mix) @ comp_mix_dot_flat

        # Unpack to named outputs
        mass_avg_h_dot, Wout_dot, composition_out_dot, W_mix_out_dot = self._unpack_output_tangent(out_dot)

        return {
            'mass_avg_h': float(mass_avg_h_dot),
            'Wout': float(Wout_dot),
            'composition_out': np.asarray(composition_out_dot),
            'W_mix': np.asarray(W_mix_out_dot)
        }

    def vjp(self, mass_avg_h_bar=0.0, Wout_bar=0.0, composition_out_bar=None, W_mix_bar=None):
        """
        Compute vector-Jacobian product (reverse-mode autodiff).

        Computes the gradient of a scalar loss with respect to inputs,
        given the gradient of the loss with respect to outputs.
        Must call linearize() first.

        Parameters
        ----------
        mass_avg_h_bar : float
            Cotangent for mass_avg_h
        Wout_bar : float
            Cotangent for Wout
        composition_out_bar : array, optional
            Cotangent for composition_out
        W_mix_bar : array, optional
            Cotangent for W_mix

        Returns
        -------
        dict
            Cotangents for inputs. Keys depend on mix_mode:
            - Always: 'W', 'h', 'composition', 'h_mix'
            - Reactant mode: 'ratio'
            - Flow mode: 'W_mix', 'composition_mix'
        """
        if not hasattr(self, '_jac_W'):
            raise RuntimeError("Must call linearize() before vjp()")

        # Build output cotangent vector
        out_bar = np.zeros(self._num_outputs)
        out_bar[0] = float(mass_avg_h_bar)
        out_bar[1] = float(Wout_bar)
        if composition_out_bar is not None:
            out_bar[2:2 + self.num_output_elements] = np.asarray(composition_out_bar)
        if W_mix_bar is not None:
            out_bar[2 + self.num_output_elements:] = np.asarray(W_mix_bar)

        # Compute input cotangents via transpose of Jacobians
        W_bar = float(np.dot(out_bar, np.asarray(self._jac_W)))
        h_bar = float(np.dot(out_bar, np.asarray(self._jac_h)))
        composition_bar = np.asarray(self._jac_composition).T @ out_bar
        h_mix_bar_arr = np.asarray(self._jac_h_mix).T @ out_bar

        result = {
            'W': W_bar,
            'h': h_bar,
            'composition': composition_bar,
            'h_mix': {name: float(h_mix_bar_arr[i]) for i, name in enumerate(self.mix_names)}
        }

        if self.mix_mode == 'reactant':
            ratio_bar_arr = np.asarray(self._jac_ratio).T @ out_bar
            result['ratio'] = {name: float(ratio_bar_arr[i]) for i, name in enumerate(self.mix_names)}
        else:
            W_mix_bar_arr = np.asarray(self._jac_W_mix).T @ out_bar
            result['W_mix'] = {name: float(W_mix_bar_arr[i]) for i, name in enumerate(self.mix_names)}

            comp_mix_bar_flat = np.asarray(self._jac_comp_mix).T @ out_bar
            # Unflatten composition_mix gradients
            result['composition_mix'] = {}
            offset = 0
            for name in self.mix_names:
                num_elem = len(self.mix_wt_mole[name])
                result['composition_mix'][name] = comp_mix_bar_flat[offset:offset + num_elem]
                offset += num_elem

        return result


if __name__ == "__main__":
    # Quick test
    print("Testing ThermoAdd functional interface")
    print("=" * 50)

    # Test reactant mode (fuel addition)
    print("\n1. Reactant mode (fuel addition):")
    mixer = ThermoAdd(mix_mode='reactant', mix_composition='JP-7', mix_names='fuel')

    air_thermo = Properties(janaf, init_elements=CEA_AIR_COMPOSITION)
    W = 38.8
    h = 181.381769
    composition = air_thermo.b0
    ratio = 0.02673

    result = mixer.compute(W=W, h=h, composition=composition,
                          ratio={'fuel': ratio}, h_mix={'fuel': 0.0})

    print(f"  Inflow: W={W}, h={h}")
    print(f"  Fuel ratio: {ratio}")
    print(f"  Output: Wout={result.Wout:.4f}, mass_avg_h={result.mass_avg_h:.4f}")
    print(f"  Fuel W: {result.W_mix[0]:.4f}")  # W_mix is now an array, index 0 for 'fuel'
    print(f"  Composition: {result.composition_out}")

    # Test flow mode
    print("\n2. Flow mode (stream mixing):")
    mixer2 = ThermoAdd(
        inflow_composition={'C': 0.000314, 'N': 0.00211, 'O': 0.00421, 'Ar': 0.0523, 'H': 0.0141},
        mix_mode='flow',
        mix_composition=CEA_AIR_COMPOSITION,
        mix_names='mix'
    )

    W_main = 62.15
    h_main = 10.0
    comp_main = np.array([0.000313780313538, 0.0021127831122, 0.004208814234964,
                          0.052325087161902, 0.014058631311261])

    W_bleed = 4.44635
    h_bleed = 5.0

    result2 = mixer2.compute(W=W_main, h=h_main, composition=comp_main,
                            W_mix={'mix': W_bleed}, h_mix={'mix': h_bleed},
                            composition_mix={'mix': air_thermo.b0})

    print(f"  Main: W={W_main}, h={h_main}")
    print(f"  Mix: W={W_bleed}, h={h_bleed}")
    print(f"  Output: Wout={result2.Wout:.4f}, mass_avg_h={result2.mass_avg_h:.4f}")
