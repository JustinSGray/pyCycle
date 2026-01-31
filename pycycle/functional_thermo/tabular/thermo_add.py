"""
Functional interface for mixing thermodynamic streams using tabular thermodynamics.

This module provides a functional interface for computing mixed flow properties
when combining an inflow stream with reactants (like fuel) or other flow streams.

Units Convention (SI):
    W - Mass flow rate (kg/s)
    h - Enthalpy (J/kg)
    composition - Array of component ratios (e.g., [FAR])
"""

import numpy as np
from collections import namedtuple

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from pycycle.constants import TAB_AIR_FUEL_COMPOSITION


# Named tuple for returning mixed flow results (matches OpenMDAO outputs)
ThermoAddOutput = namedtuple('ThermoAddOutput', [
    'mass_avg_h',       # Mass-averaged enthalpy (J/kg)
    'Wout',             # Total mass flow rate out (kg/s)
    'composition_out',  # Mixed composition (array of ratios)
    'W_mix',            # Array of mass flow rates for each mix stream (kg/s)
])


class ThermoAdd:
    """
    Functional interface for mixing thermodynamic streams using tabular data.

    This class computes the mixed composition and mass-averaged enthalpy
    when combining an inflow stream with reactants or other flow streams.

    The tabular implementation uses a simplified composition model where
    composition is represented as a vector of component-to-air ratios
    (e.g., [FAR] for fuel-to-air ratio).

    Parameters
    ----------
    inflow_composition : dict, optional
        Composition of the inflow as {component: ratio}.
        Default is TAB_AIR_FUEL_COMPOSITION {'FAR': 0.0}.
    mix_mode : str, optional
        'reactant' to mix by fuel-to-air ratio, 'flow' to mix flow streams.
        Default is 'reactant'.
    mix_composition : str, optional
        For 'reactant' mode: name of reactant to add (must be key in inflow_composition).
        For 'flow' mode: not used (composition comes from input).
        Default is 'FAR'.
    mix_names : str or list, optional
        Name(s) for the mix streams. Default is 'mix'.

    Examples
    --------
    Reactant mode (adding fuel):

    >>> mixer = ThermoAdd(mix_mode='reactant', mix_composition='FAR')
    >>> result = mixer.compute(W=17.6, h=422000.0, composition=np.array([0.0]),
    ...                        ratio={'mix': 0.02673}, h_mix={'mix': 0.0})

    Flow mode (mixing two streams):

    >>> mixer = ThermoAdd(mix_mode='flow', mix_names='bleed')
    >>> result = mixer.compute(W=28.0, h=422000.0, composition=np.array([0.01]),
    ...                        W_mix={'bleed': 2.0}, h_mix={'bleed': 300000.0},
    ...                        composition_mix={'bleed': np.array([0.0])})
    """

    def __init__(self, inflow_composition=None, mix_mode='reactant',
                 mix_composition='FAR', mix_names='mix'):

        if inflow_composition is None:
            inflow_composition = TAB_AIR_FUEL_COMPOSITION

        self.mix_mode = mix_mode

        # Store inflow composition as sorted list of keys and values
        self.sorted_compo = sorted(inflow_composition.keys())
        self.inflow_composition = inflow_composition
        self.num_composition = len(self.sorted_compo)

        # Normalize mix_names to tuple
        if isinstance(mix_names, str):
            mix_names = (mix_names,)
        self.mix_names = mix_names

        # For reactant mode, store the index of the reactant in composition
        if mix_mode == 'reactant':
            if mix_composition is None:
                mix_composition = 'FAR'
            self.mix_composition = mix_composition
            self.idx_compo = self.sorted_compo.index(mix_composition)
        else:
            self.mix_composition = mix_composition

    @property
    def inflow_composition_vec(self):
        """Return the default inflow composition as an array."""
        return np.array([self.inflow_composition[k] for k in self.sorted_compo])

    def _mix_core(self, W, h, compo_in, ratio_or_W_mix_arr, h_mix_arr, comp_mix_flat, xp):
        """
        Core mixing logic that works with both NumPy and JAX arrays.

        Parameters
        ----------
        W : scalar
            Inflow mass flow rate (kg/s)
        h : scalar
            Inflow total enthalpy (J/kg)
        compo_in : array
            Inflow composition
        ratio_or_W_mix_arr : array (num_mix,)
            For reactant mode: ratios. For flow mode: W_mix values.
        h_mix_arr : array (num_mix,)
            Enthalpies of mix streams (J/kg)
        comp_mix_flat : array or None
            For flow mode: flattened mix compositions (num_mix * num_composition,)
        xp : module
            Array module (np or jnp)

        Returns
        -------
        tuple
            (mass_avg_h, W_out, composition_out, W_mix_out)
        """
        n_compo = self.num_composition
        num_mix = len(self.mix_names)

        # Composition vector is given as vector of <something>-to-air ratios
        # W_air_in is the mass of pure air in the inflow
        W_air_in = W / (1 + xp.sum(compo_in))
        W_other_in = W_air_in * compo_in

        W_out = W
        W_other_out = W_other_in
        W_air_out = W_air_in
        W_times_h = W * h

        W_mix_out = xp.zeros(num_mix)

        if self.mix_mode == 'reactant':
            for idx in range(num_mix):
                r = ratio_or_W_mix_arr[idx]
                # For reactant mode, ratio is relative to air mass
                W_other_mix = W_air_in * r

                if xp is jnp:
                    W_mix_out = W_mix_out.at[idx].set(W_other_mix)
                    W_other_out = W_other_out.at[self.idx_compo].add(W_other_mix)
                else:
                    W_mix_out[idx] = W_other_mix
                    W_other_out[self.idx_compo] += W_other_mix

                W_out = W_out + W_other_mix
                W_times_h = W_times_h + W_other_mix * h_mix_arr[idx]

            composition_out = W_other_out / W_air_in

        else:  # flow mode
            for idx in range(num_mix):
                W_stream = ratio_or_W_mix_arr[idx]

                if xp is jnp:
                    W_mix_out = W_mix_out.at[idx].set(W_stream)
                else:
                    W_mix_out[idx] = W_stream

                # Get composition from flattened array
                compo_mix = comp_mix_flat[idx * n_compo:(idx + 1) * n_compo]

                W_air_mix = W_stream / (1 + xp.sum(compo_mix))
                W_other_out = W_other_out + W_air_mix * compo_mix
                W_out = W_out + W_stream
                W_air_out = W_air_out + W_air_mix
                W_times_h = W_times_h + W_stream * h_mix_arr[idx]

            composition_out = W_other_out / W_air_out

        mass_avg_h = W_times_h / W_out

        return mass_avg_h, W_out, composition_out, W_mix_out

    def compute(self, W, h, composition, ratio=None, W_mix=None, h_mix=None, composition_mix=None):
        """
        Compute mixed flow properties.

        Parameters
        ----------
        W : float
            Inflow mass flow rate (kg/s)
        h : float
            Inflow total enthalpy (J/kg)
        composition : ndarray
            Inflow composition as array of ratios (e.g., [FAR])
        ratio : dict, optional
            For reactant mode: {name: ratio} where ratio is reactant-to-inflow mass ratio
        W_mix : dict, optional
            For flow mode: {name: mass_flow} for each mix stream (kg/s)
        h_mix : dict, optional
            Enthalpy of each mix stream (J/kg). Required for both modes.
        composition_mix : dict, optional
            For flow mode: {name: composition_array} composition of each mix stream

        Returns
        -------
        ThermoAddOutput
            Named tuple with (mass_avg_h, Wout, composition_out, W_mix).
            W_mix is an array ordered by self.mix_names.
        """
        # Convert dict inputs to arrays
        if h_mix is None:
            h_mix = {name: 0.0 for name in self.mix_names}
        h_mix_arr = np.array([h_mix.get(name, 0.0) for name in self.mix_names])

        compo_in = np.asarray(composition).copy()  # copy for in-place updates in reactant mode

        if self.mix_mode == 'reactant':
            if ratio is None:
                ratio = {name: 0.0 for name in self.mix_names}
            ratio_arr = np.array([ratio.get(name, 0.0) for name in self.mix_names])
            comp_mix_flat = None
        else:
            if W_mix is None:
                W_mix = {name: 0.0 for name in self.mix_names}
            if composition_mix is None:
                composition_mix = {name: self.inflow_composition_vec for name in self.mix_names}
            ratio_arr = np.array([W_mix.get(name, 0.0) for name in self.mix_names])
            comp_mix_flat = np.concatenate([
                np.asarray(composition_mix.get(name, self.inflow_composition_vec))
                for name in self.mix_names
            ])

        mass_avg_h, W_out, composition_out, W_mix_out = self._mix_core(
            W, h, compo_in, ratio_arr, h_mix_arr, comp_mix_flat, np
        )

        return ThermoAddOutput(
            mass_avg_h=mass_avg_h,
            Wout=float(W_out),
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
            Inflow mass flow rate (kg/s)
        h : scalar
            Inflow total enthalpy (J/kg)
        composition : array (num_composition,)
            Inflow composition
        ratio_or_W_mix : array (num_mix,)
            For reactant mode: ratios. For flow mode: W_mix values.
        h_mix : array (num_mix,)
            Enthalpies of mix streams (J/kg)
        composition_mix_flat : array, optional
            For flow mode only: flattened mix compositions.
            Shape: (num_mix * num_composition,)

        Returns
        -------
        tuple
            (mass_avg_h, Wout, composition_out, W_mix_out)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for _compute_jax. Install with: pip install jax jaxlib")

        return self._mix_core(W, h, composition, ratio_or_W_mix, h_mix, composition_mix_flat, jnp)

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
                composition_mix = {name: self.inflow_composition_vec for name in self.mix_names}

            W_mix_arr = np.array([W_mix.get(name, 0.0) for name in self.mix_names])

            # Flatten composition_mix - all have same size for tabular
            comp_mix_flat = np.concatenate([
                np.asarray(composition_mix.get(name, self.inflow_composition_vec))
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

        # Flatten inputs
        W, h, composition, ratio_or_W_mix, h_mix_arr, comp_mix_flat = self._flatten_inputs(
            W, h, composition, ratio, W_mix, h_mix, composition_mix
        )

        # Define the function to differentiate
        if self.mix_mode == 'reactant':
            def f(W, h, composition, ratio_arr, h_mix_arr):
                mass_avg_h, Wout, composition_out, W_mix_out = self._compute_jax(
                    W, h, composition, ratio_arr, h_mix_arr, None
                )
                return jnp.concatenate([
                    jnp.array([mass_avg_h, Wout]),
                    composition_out,
                    W_mix_out
                ])

            self._jac_W = jax.jacfwd(
                lambda W: f(W, h, composition, ratio_or_W_mix, h_mix_arr)
            )(W)
            self._jac_h = jax.jacfwd(
                lambda h: f(W, h, composition, ratio_or_W_mix, h_mix_arr)
            )(h)
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
        self._num_outputs = 2 + self.num_composition + len(self.mix_names)

    def _unpack_output_tangent(self, out_dot):
        """Unpack flattened output tangent to named components."""
        idx = 0
        mass_avg_h_dot = out_dot[idx]
        idx += 1
        Wout_dot = out_dot[idx]
        idx += 1
        composition_out_dot = out_dot[idx:idx + self.num_composition]
        idx += self.num_composition
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

        out_dot = np.zeros(self._num_outputs)

        out_dot += float(W_dot) * np.asarray(self._jac_W)
        out_dot += float(h_dot) * np.asarray(self._jac_h)

        if composition_dot is not None:
            out_dot += np.asarray(self._jac_composition) @ np.asarray(composition_dot)

        if h_mix_dot is not None:
            h_mix_dot_arr = np.array([h_mix_dot.get(name, 0.0) for name in self.mix_names])
            out_dot += np.asarray(self._jac_h_mix) @ h_mix_dot_arr

        if self.mix_mode == 'reactant':
            if ratio_dot is not None:
                ratio_dot_arr = np.array([ratio_dot.get(name, 0.0) for name in self.mix_names])
                out_dot += np.asarray(self._jac_ratio) @ ratio_dot_arr
        else:
            if W_mix_dot is not None:
                W_mix_dot_arr = np.array([W_mix_dot.get(name, 0.0) for name in self.mix_names])
                out_dot += np.asarray(self._jac_W_mix) @ W_mix_dot_arr

            if composition_mix_dot is not None:
                comp_mix_dot_flat = np.concatenate([
                    np.asarray(composition_mix_dot.get(name, np.zeros(self.num_composition)))
                    for name in self.mix_names
                ])
                out_dot += np.asarray(self._jac_comp_mix) @ comp_mix_dot_flat

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

        out_bar = np.zeros(self._num_outputs)
        out_bar[0] = float(mass_avg_h_bar)
        out_bar[1] = float(Wout_bar)
        if composition_out_bar is not None:
            out_bar[2:2 + self.num_composition] = np.asarray(composition_out_bar)
        if W_mix_bar is not None:
            out_bar[2 + self.num_composition:] = np.asarray(W_mix_bar)

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
            result['composition_mix'] = {}
            for idx, name in enumerate(self.mix_names):
                start = idx * self.num_composition
                end = start + self.num_composition
                result['composition_mix'][name] = comp_mix_bar_flat[start:end]

        return result
