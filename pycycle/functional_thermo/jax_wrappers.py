"""
JAX-compatible wrappers for functional thermodynamic interfaces.

This module provides:
- JaxThermo: A wrapper class that makes functional thermo objects fully JAX-traceable
- Named index classes for accessing property arrays

The wrappers use custom_jvp with pure_callback to isolate non-JAX code while
allowing JAX to trace through the full computation graph. This enables
efficient JIT-compiled Jacobian computation via jacfwd/jacrev.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import custom_jvp, pure_callback


# =============================================================================
# Timing Functions
# =============================================================================

import time

_jax_thermo_timing = {
    'linearize_at_calls': 0,
    'linearize_at_time': 0.0,
    'linearize_T_from_hP_time': 0.0,
    'linearize_props_TP_time': 0.0,
    'linearize_static_time': 0.0,
}

def print_jax_thermo_timing():
    """Print timing stats."""
    print("\n=== JaxThermo Timing ===")
    print(f"  linearize_at() calls: {_jax_thermo_timing['linearize_at_calls']}")
    print(f"  linearize_at() total time: {_jax_thermo_timing['linearize_at_time']*1000:.3f} ms")
    if _jax_thermo_timing['linearize_at_calls'] > 0:
        print(f"  linearize_at() avg time: {_jax_thermo_timing['linearize_at_time']/_jax_thermo_timing['linearize_at_calls']*1000:.3f} ms")
    print(f"  linearize_at() breakdown:")
    print(f"    T_from_hP: {_jax_thermo_timing['linearize_T_from_hP_time']*1000:.3f} ms")
    if 'T_solve_time' in _jax_thermo_timing:
        print(f"      T_solve: {_jax_thermo_timing['T_solve_time']*1000:.3f} ms")
        print(f"      linearize: {_jax_thermo_timing['linearize_call_time']*1000:.3f} ms")
        print(f"      jvp_calls: {_jax_thermo_timing['jvp_calls_time']*1000:.3f} ms")
    print(f"    props_TP: {_jax_thermo_timing['linearize_props_TP_time']*1000:.3f} ms")
    print(f"    static: {_jax_thermo_timing['linearize_static_time']*1000:.3f} ms")
    print("========================\n")

def reset_jax_thermo_timing():
    """Reset timing stats."""
    for key in _jax_thermo_timing:
        _jax_thermo_timing[key] = 0.0 if 'time' in key else 0


# =============================================================================
# Named Index Classes for Property Arrays
# =============================================================================

class TotalPropsIdx:
    """Named indices for total property arrays returned by JaxThermo.props_TP."""
    h = 0
    S = 1
    gamma = 2
    Cp = 3
    Cv = 4
    rho = 5
    R = 6

    @classmethod
    def count(cls):
        return 7


class StaticPropsIdx:
    """Named indices for static property arrays returned by JaxThermo.static_from_*."""
    Ts = 0
    Ps = 1
    hs = 2
    rhos = 3
    MN = 4
    V = 5
    Vsonic = 6
    area = 7
    gamma = 8
    Cp = 9
    Cv = 10
    S = 11
    R = 12

    @classmethod
    def count(cls):
        return 13


# =============================================================================
# JaxThermo Wrapper Class
# =============================================================================

class JaxThermo:
    """
    Wrapper that provides fully JAX-traceable interface to functional thermo objects.

    Uses custom_jvp with pure_callback to isolate non-JAX code (scipy interpolation)
    while enabling JAX to trace through the computation graph. This allows:
    - JIT compilation of the full computation
    - Efficient Jacobian computation via jacfwd/jacrev
    - One-time tracing with cached compiled functions

    Supports pre-linearization via linearize_at() to avoid redundant linearization
    calls during Jacobian computation.

    All methods accept composition as an array parameter:
    - For TabularThermo: composition = [FAR], extracts FAR = composition[0]
    - For CEAThermo: composition = elemental fractions, FAR is not used (always 0)

    Note: This class is designed to be shareable across multiple JaxElement instances.
    The linearization cache is stored externally and set via set_cache() before
    computing Jacobians. This allows multiple instances to share the compiled
    JAX functions while maintaining separate operating-point state.
    """

    def __init__(self, thermo):
        self._thermo = thermo
        self._cache = {}  # Linearization cache - can be pointed to external dict via set_cache()

        # Check if this is a tabular thermo that supports FAR as a parameter
        # TabularThermo has FAR as a direct parameter; CEAThermo uses composition
        from pycycle.functional_thermo.tabular import TabularThermo
        self._supports_FAR = isinstance(thermo, TabularThermo)

        self._setup_wrappers()

    def set_cache(self, cache):
        """
        Set the cache to an external dict for subsequent linearization and JVP calls.

        This makes self._cache point to the caller's dict, allowing shared
        JaxThermo objects to use instance-specific caches.

        Parameters
        ----------
        cache : dict
            External cache dict to store linearized derivatives
        """
        self._cache = cache

    def clear_cache(self):
        """
        Reset the cache to an empty dict.

        This unbinds from any external cache and resets to a fresh empty dict.
        """
        self._cache = {}

    def _extract_FAR(self, composition):
        """
        Extract FAR from composition array based on thermo type.

        For TABULAR: composition = [FAR], so FAR = composition[0]
        For CEA: FAR is not in composition, return 0.0
        """
        if self._supports_FAR:
            return float(composition[0]) if len(composition) > 0 else 0.0
        else:
            return 0.0

    def linearize_at(self, ht, Pt, W, MN_or_area, is_design, statics, composition=None, T=None, props=None, static_props=None):
        """
        Pre-linearize all thermo functions at the current operating point.

        Call this before computing Jacobians to avoid redundant linearization
        during JVP evaluations.

        Parameters
        ----------
        ht : float
            Total enthalpy at the operating point
        Pt : float
            Total pressure at the operating point
        W : float
            Mass flow rate
        MN_or_area : float or None
            Mach number (design) or area (off-design), or None if no statics
        is_design : bool
            True for design mode (MN input), False for off-design (area input)
        statics : bool
            Whether static properties are being computed
        composition : array-like, optional
            Composition array. For TABULAR, composition[0] = FAR.
            For CEA, this is elemental fractions (FAR not used).
        T : float, optional
            Temperature from forward pass. If provided, skips T_from_hP solve.
        props : TotalProps, optional
            Properties from forward pass (h, S, gamma, Cp, Cv, rho, R).
            If provided, skips property lookups in linearize().
        static_props : StaticProps, optional
            Static properties from forward pass. If provided, skips static
            property lookups in linearize_static_MN/area().
        """
        t_start = time.perf_counter()
        thermo = self._thermo
        supports_FAR = self._supports_FAR

        # Extract FAR from composition
        FAR = self._extract_FAR(composition) if composition is not None else 0.0

        # T_from_hP: compute T, then linearize (which also returns props)
        t0 = time.perf_counter()
        if supports_FAR:
            t_solve = time.perf_counter()
            if T is None:
                T = thermo.T_from_hP(ht, Pt, FAR)
            _jax_thermo_timing.setdefault('T_solve_time', 0.0)
            _jax_thermo_timing['T_solve_time'] += (time.perf_counter() - t_solve)

            t_lin = time.perf_counter()
            # linearize() computes gradients; if props passed, skips lookups
            props = thermo.linearize(T, Pt, FAR, props=props)
            _jax_thermo_timing.setdefault('linearize_call_time', 0.0)
            _jax_thermo_timing['linearize_call_time'] += (time.perf_counter() - t_lin)

            t_jvp = time.perf_counter()
            jvp_T = thermo.jvp(1.0, 0.0, 0.0)
            jvp_P = thermo.jvp(0.0, 1.0, 0.0)
            jvp_FAR = thermo.jvp(0.0, 0.0, 1.0)
            _jax_thermo_timing.setdefault('jvp_calls_time', 0.0)
            _jax_thermo_timing['jvp_calls_time'] += (time.perf_counter() - t_jvp)
            dh_dFAR = jvp_FAR['h']
        else:
            t_solve = time.perf_counter()
            if T is None:
                T = thermo.T_from_hP(ht, Pt)
            _jax_thermo_timing.setdefault('T_solve_time', 0.0)
            _jax_thermo_timing['T_solve_time'] += (time.perf_counter() - t_solve)

            t_lin = time.perf_counter()
            # linearize() now returns props AND computes gradients in one pass
            props = thermo.linearize(T, Pt)
            _jax_thermo_timing.setdefault('linearize_call_time', 0.0)
            _jax_thermo_timing['linearize_call_time'] += (time.perf_counter() - t_lin)

            t_jvp = time.perf_counter()
            jvp_T = thermo.jvp(1.0, 0.0)
            jvp_P = thermo.jvp(0.0, 1.0)
            _jax_thermo_timing.setdefault('jvp_calls_time', 0.0)
            _jax_thermo_timing['jvp_calls_time'] += (time.perf_counter() - t_jvp)
            jvp_FAR = None
            dh_dFAR = 0.0
        dh_dT = jvp_T['h']
        dh_dP = jvp_P['h']
        dT_dh = 1.0 / dh_dT if abs(dh_dT) > 1e-20 else 0.0
        dT_dP = -dh_dP / dh_dT if abs(dh_dT) > 1e-20 else 0.0
        dT_dFAR = -dh_dFAR / dh_dT if abs(dh_dT) > 1e-20 else 0.0

        # Store with FAR for cache comparison (extracted from composition)
        self._cache['T_from_hP'] = (ht, Pt, FAR, T, dT_dh, dT_dP, dT_dFAR)
        _jax_thermo_timing['linearize_T_from_hP_time'] += (time.perf_counter() - t0)

        # props_TP: use props already returned by linearize() above
        t0 = time.perf_counter()
        keys = ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']
        dprops_dT = np.array([jvp_T[k] for k in keys])
        dprops_dP = np.array([jvp_P[k] for k in keys])
        if supports_FAR:
            dprops_dFAR = np.array([jvp_FAR[k] for k in keys])
        else:
            dprops_dFAR = np.zeros(len(keys))
        props_arr = np.array([props.h, props.S, props.gamma, props.Cp,
                              props.Cv, props.rho, props.R])
        self._cache['props_TP'] = (T, Pt, FAR, props_arr, dprops_dT, dprops_dP, dprops_dFAR)
        _jax_thermo_timing['linearize_props_TP_time'] += (time.perf_counter() - t0)

        # Static properties - use pre-computed Jacobians from functional thermo
        t0 = time.perf_counter()
        if statics and MN_or_area is not None:
            static_keys = ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic',
                          'area', 'gamma', 'Cp', 'Cv', 'S', 'R']
            if is_design:
                try:
                    # linearize_static_MN returns props AND computes Jacobian in one pass
                    # Pass static_props to skip value lookups if available
                    if supports_FAR:
                        sprops = thermo.linearize_static_MN(T, Pt, MN_or_area, W, FAR, sprops=static_props)
                    else:
                        sprops = thermo.linearize_static_MN(T, Pt, MN_or_area, W)
                    # Get full Jacobian directly (no 5x jvp calls needed)
                    jac = thermo.get_jacobian_static_MN()
                    sprops_arr = np.array([sprops.Ts, sprops.Ps, sprops.hs, sprops.rhos,
                                          sprops.MN, sprops.V, sprops.Vsonic, sprops.area,
                                          sprops.gamma, sprops.Cp, sprops.Cv, sprops.S, sprops.R])
                    # Extract columns of Jacobian: [dTt, dPt, dMN, dW, dFAR]
                    derivs = [np.array([jac[k][i] for k in static_keys]) for i in range(5)]
                    self._cache['static_MN'] = (T, Pt, MN_or_area, W, FAR, sprops_arr, *derivs)
                except (NotImplementedError, AttributeError):
                    pass  # No analytical derivatives available, will use FD fallback
            else:
                try:
                    # linearize_static_area returns props AND computes Jacobian in one pass
                    if supports_FAR:
                        sprops = thermo.linearize_static_area(T, Pt, MN_or_area, W, FAR)
                    else:
                        sprops = thermo.linearize_static_area(T, Pt, MN_or_area, W)
                    # Get full Jacobian directly (no 5x jvp calls needed)
                    jac = thermo.get_jacobian_static_area()
                    sprops_arr = np.array([sprops.Ts, sprops.Ps, sprops.hs, sprops.rhos,
                                          sprops.MN, sprops.V, sprops.Vsonic, sprops.area,
                                          sprops.gamma, sprops.Cp, sprops.Cv, sprops.S, sprops.R])
                    # Extract columns of Jacobian: [dTt, dPt, darea, dW, dFAR]
                    derivs = [np.array([jac[k][i] for k in static_keys]) for i in range(5)]
                    self._cache['static_area'] = (T, Pt, MN_or_area, W, FAR, sprops_arr, *derivs)
                except (NotImplementedError, AttributeError):
                    pass  # No analytical derivatives available, will use FD fallback
        _jax_thermo_timing['linearize_static_time'] += (time.perf_counter() - t0)

        _jax_thermo_timing['linearize_at_calls'] += 1
        _jax_thermo_timing['linearize_at_time'] += (time.perf_counter() - t_start)

    def _setup_wrappers(self):
        """Create fully JAX-traceable wrappers for thermo methods."""
        thermo = self._thermo

        # ---------------------------------------------------------------------
        # T_from_hP wrapper - accepts composition as third argument
        # For TabularThermo: FAR = composition[0], affects properties and derivatives
        # For CEAThermo: FAR is ignored (composition is elemental fractions)
        # ---------------------------------------------------------------------
        supports_FAR = self._supports_FAR
        extract_FAR = self._extract_FAR

        def _T_from_hP_impl(args):
            """Pure Python implementation - called via pure_callback."""
            h, P = float(args[0]), float(args[1])
            composition = np.array(args[2:])
            FAR = extract_FAR(composition)
            if supports_FAR:
                return np.array([thermo.T_from_hP(h, P, FAR)])
            else:
                return np.array([thermo.T_from_hP(h, P)])

        def _T_from_hP_derivs(args):
            """Compute T and its derivatives - called via pure_callback."""
            h, P = float(args[0]), float(args[1])
            composition = np.array(args[2:])
            FAR = extract_FAR(composition)

            # Check cache first
            if 'T_from_hP' in self._cache:
                ch, cP, cFAR, T, dT_dh, dT_dP, dT_dFAR = self._cache['T_from_hP']
                if abs(ch - h) < 1e-10 and abs(cP - P) < 1e-10 and abs(cFAR - FAR) < 1e-10:
                    return np.array([T, dT_dh, dT_dP, dT_dFAR])

            # Compute fresh (fallback)
            if supports_FAR:
                T = thermo.T_from_hP(h, P, FAR)
                thermo.linearize(T, P, FAR)
                jvp_T = thermo.jvp(1.0, 0.0, 0.0)
                jvp_P = thermo.jvp(0.0, 1.0, 0.0)
                jvp_FAR = thermo.jvp(0.0, 0.0, 1.0)
                dh_dT = jvp_T['h']
                dh_dP = jvp_P['h']
                dh_dFAR = jvp_FAR['h']
                dT_dh = 1.0 / dh_dT if abs(dh_dT) > 1e-20 else 0.0
                dT_dP = -dh_dP / dh_dT if abs(dh_dT) > 1e-20 else 0.0
                dT_dFAR = -dh_dFAR / dh_dT if abs(dh_dT) > 1e-20 else 0.0
            else:
                T = thermo.T_from_hP(h, P)
                thermo.linearize(T, P)
                jvp_T = thermo.jvp(1.0, 0.0)
                jvp_P = thermo.jvp(0.0, 1.0)
                dh_dT = jvp_T['h']
                dh_dP = jvp_P['h']
                dT_dh = 1.0 / dh_dT if abs(dh_dT) > 1e-20 else 0.0
                dT_dP = -dh_dP / dh_dT if abs(dh_dT) > 1e-20 else 0.0
                dT_dFAR = 0.0
            return np.array([T, dT_dh, dT_dP, dT_dFAR])

        @custom_jvp
        def T_from_hP_jax(h, P, composition):
            # Pack h, P, and composition into a single array for pure_callback
            args = jnp.concatenate([jnp.array([h, P]), composition])
            result = pure_callback(_T_from_hP_impl,
                                   jax.ShapeDtypeStruct((1,), jnp.float64),
                                   args)
            return result[0]

        @T_from_hP_jax.defjvp
        def T_from_hP_jvp(primals, tangents):
            h, P, composition = primals
            h_dot, P_dot, composition_dot = tangents

            # Pack args for callback
            args = jnp.concatenate([jnp.array([h, P]), composition])
            result = pure_callback(_T_from_hP_derivs,
                                   jax.ShapeDtypeStruct((4,), jnp.float64),
                                   args)
            T = result[0]
            dT_dh = result[1]
            dT_dP = result[2]
            dT_dFAR = result[3]

            # For TABULAR: FAR = composition[0], so FAR_dot = composition_dot[0]
            # For CEA: dT_dFAR = 0, so this term vanishes regardless
            FAR_dot = composition_dot[0] if supports_FAR else 0.0

            T_dot = dT_dh * h_dot + dT_dP * P_dot + dT_dFAR * FAR_dot
            return T, T_dot

        self.T_from_hP = T_from_hP_jax

        # ---------------------------------------------------------------------
        # props_TP wrapper - returns [h, S, gamma, Cp, Cv, rho, R]
        # Accepts composition as third argument
        # ---------------------------------------------------------------------
        def _props_TP_impl(args):
            """Pure Python implementation."""
            T, P = float(args[0]), float(args[1])
            composition = np.array(args[2:])
            FAR = extract_FAR(composition)
            if supports_FAR:
                props = thermo.props_TP(T, P, FAR)
            else:
                props = thermo.props_TP(T, P)
            return np.array([props.h, props.S, props.gamma, props.Cp,
                            props.Cv, props.rho, props.R])

        def _props_TP_derivs(args):
            """Compute props and Jacobian (7 outputs x 3 inputs)."""
            T, P = float(args[0]), float(args[1])
            composition = np.array(args[2:])
            FAR = extract_FAR(composition)

            # Check cache first
            if 'props_TP' in self._cache:
                cT, cP, cFAR, props_arr, dprops_dT, dprops_dP, dprops_dFAR = self._cache['props_TP']
                if abs(cT - T) < 1e-10 and abs(cP - P) < 1e-10 and (cFAR is None or abs(cFAR - FAR) < 1e-10):
                    return np.concatenate([props_arr, dprops_dT, dprops_dP, dprops_dFAR])

            # Compute fresh (fallback)
            if supports_FAR:
                props = thermo.props_TP(T, P, FAR)
                thermo.linearize(T, P, FAR)
                jvp_T = thermo.jvp(1.0, 0.0, 0.0)
                jvp_P = thermo.jvp(0.0, 1.0, 0.0)
                jvp_FAR = thermo.jvp(0.0, 0.0, 1.0)
            else:
                props = thermo.props_TP(T, P)
                thermo.linearize(T, P)
                jvp_T = thermo.jvp(1.0, 0.0)
                jvp_P = thermo.jvp(0.0, 1.0)
                jvp_FAR = {'h': 0.0, 'S': 0.0, 'gamma': 0.0, 'Cp': 0.0, 'Cv': 0.0, 'rho': 0.0, 'R': 0.0}
            # Return: primal(7) + dT(7) + dP(7) + dFAR(7) = 28 values
            return np.array([
                props.h, props.S, props.gamma, props.Cp, props.Cv, props.rho, props.R,
                jvp_T['h'], jvp_T['S'], jvp_T['gamma'], jvp_T['Cp'], jvp_T['Cv'], jvp_T['rho'], jvp_T['R'],
                jvp_P['h'], jvp_P['S'], jvp_P['gamma'], jvp_P['Cp'], jvp_P['Cv'], jvp_P['rho'], jvp_P['R'],
                jvp_FAR['h'], jvp_FAR['S'], jvp_FAR['gamma'], jvp_FAR['Cp'], jvp_FAR['Cv'], jvp_FAR['rho'], jvp_FAR['R'],
            ])

        @custom_jvp
        def props_TP_jax(T, P, composition):
            args = jnp.concatenate([jnp.array([T, P]), composition])
            return pure_callback(_props_TP_impl,
                                 jax.ShapeDtypeStruct((7,), jnp.float64),
                                 args)

        @props_TP_jax.defjvp
        def props_TP_jvp(primals, tangents):
            T, P, composition = primals
            T_dot, P_dot, composition_dot = tangents

            args = jnp.concatenate([jnp.array([T, P]), composition])
            result = pure_callback(_props_TP_derivs,
                                   jax.ShapeDtypeStruct((28,), jnp.float64),
                                   args)

            primal = result[0:7]
            dprops_dT = result[7:14]
            dprops_dP = result[14:21]
            dprops_dFAR = result[21:28]

            # For TABULAR: FAR = composition[0], so FAR_dot = composition_dot[0]
            # For CEA: dprops_dFAR = 0, so this term vanishes regardless
            FAR_dot = composition_dot[0] if supports_FAR else 0.0

            props_dot = dprops_dT * T_dot + dprops_dP * P_dot + dprops_dFAR * FAR_dot
            return primal, props_dot

        self.props_TP = props_TP_jax

        # ---------------------------------------------------------------------
        # static_from_MN wrapper
        # Returns: [Ts, Ps, hs, rhos, MN, V, Vsonic, area, gamma, Cp, Cv, S, R]
        # Accepts composition as fifth argument
        # ---------------------------------------------------------------------
        def _static_MN_impl(args):
            """Pure Python implementation."""
            Tt, Pt, MN, W = float(args[0]), float(args[1]), float(args[2]), float(args[3])
            composition = np.array(args[4:])
            FAR = extract_FAR(composition)
            if supports_FAR:
                props = thermo.static_from_MN(Tt, Pt, MN, W, FAR)
            else:
                props = thermo.static_from_MN(Tt, Pt, MN, W)
            return np.array([props.Ts, props.Ps, props.hs, props.rhos,
                            props.MN, props.V, props.Vsonic, props.area,
                            props.gamma, props.Cp, props.Cv, props.S, props.R])

        def _static_MN_derivs(args):
            """Compute static props and Jacobian (13 outputs x 5 inputs)."""
            Tt, Pt, MN, W = float(args[0]), float(args[1]), float(args[2]), float(args[3])
            composition = np.array(args[4:])
            FAR = extract_FAR(composition)

            # Check cache first
            if 'static_MN' in self._cache:
                cTt, cPt, cMN, cW, cFAR, sprops_arr, dTt, dPt, dMN, dW, dFAR = self._cache['static_MN']
                if (abs(cTt - Tt) < 1e-10 and abs(cPt - Pt) < 1e-10 and
                    abs(cMN - MN) < 1e-10 and abs(cW - W) < 1e-10 and (cFAR is None or abs(cFAR - FAR) < 1e-10)):
                    return np.concatenate([sprops_arr, dTt, dPt, dMN, dW, dFAR])

            # Compute fresh (fallback)
            if supports_FAR:
                props = thermo.static_from_MN(Tt, Pt, MN, W, FAR)
            else:
                props = thermo.static_from_MN(Tt, Pt, MN, W)
            primal = [props.Ts, props.Ps, props.hs, props.rhos,
                      props.MN, props.V, props.Vsonic, props.area,
                      props.gamma, props.Cp, props.Cv, props.S, props.R]

            keys = ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                    'gamma', 'Cp', 'Cv', 'S', 'R']

            try:
                if supports_FAR:
                    thermo.linearize_static_MN(Tt, Pt, MN, W, FAR)
                    jvp_Tt = thermo.jvp_static_MN(1.0, 0.0, 0.0, 0.0, 0.0)
                    jvp_Pt = thermo.jvp_static_MN(0.0, 1.0, 0.0, 0.0, 0.0)
                    jvp_MN = thermo.jvp_static_MN(0.0, 0.0, 1.0, 0.0, 0.0)
                    jvp_W = thermo.jvp_static_MN(0.0, 0.0, 0.0, 1.0, 0.0)
                    jvp_FAR = thermo.jvp_static_MN(0.0, 0.0, 0.0, 0.0, 1.0)
                else:
                    thermo.linearize_static_MN(Tt, Pt, MN, W)
                    jvp_Tt = thermo.jvp_static_MN(1.0, 0.0, 0.0, 0.0)
                    jvp_Pt = thermo.jvp_static_MN(0.0, 1.0, 0.0, 0.0)
                    jvp_MN = thermo.jvp_static_MN(0.0, 0.0, 1.0, 0.0)
                    jvp_W = thermo.jvp_static_MN(0.0, 0.0, 0.0, 1.0)
                    jvp_FAR = {k: 0.0 for k in keys}  # FAR derivative is zero for CEA
                dTt = [jvp_Tt[k] for k in keys]
                dPt = [jvp_Pt[k] for k in keys]
                dMN = [jvp_MN[k] for k in keys]
                dW = [jvp_W[k] for k in keys]
                dFAR = [jvp_FAR[k] for k in keys]
            except NotImplementedError:
                # Finite difference fallback
                eps = 1e-6
                if supports_FAR:
                    def compute(Tt_, Pt_, MN_, W_, FAR_):
                        p = thermo.static_from_MN(Tt_, Pt_, MN_, W_, FAR_)
                        return [p.Ts, p.Ps, p.hs, p.rhos, p.MN, p.V, p.Vsonic, p.area,
                                p.gamma, p.Cp, p.Cv, p.S, p.R]
                    base = np.array(primal)
                    dTt = (np.array(compute(Tt+eps, Pt, MN, W, FAR)) - base) / eps
                    dPt = (np.array(compute(Tt, Pt+eps, MN, W, FAR)) - base) / eps
                    dMN = (np.array(compute(Tt, Pt, MN+eps, W, FAR)) - base) / eps
                    dW = (np.array(compute(Tt, Pt, MN, W+eps, FAR)) - base) / eps
                    dFAR = (np.array(compute(Tt, Pt, MN, W, FAR+eps)) - base) / eps
                else:
                    def compute(Tt_, Pt_, MN_, W_):
                        p = thermo.static_from_MN(Tt_, Pt_, MN_, W_)
                        return [p.Ts, p.Ps, p.hs, p.rhos, p.MN, p.V, p.Vsonic, p.area,
                                p.gamma, p.Cp, p.Cv, p.S, p.R]
                    base = np.array(primal)
                    dTt = (np.array(compute(Tt+eps, Pt, MN, W)) - base) / eps
                    dPt = (np.array(compute(Tt, Pt+eps, MN, W)) - base) / eps
                    dMN = (np.array(compute(Tt, Pt, MN+eps, W)) - base) / eps
                    dW = (np.array(compute(Tt, Pt, MN, W+eps)) - base) / eps
                    dFAR = np.zeros(len(keys))  # FAR derivative is zero for CEA

            # Return: primal(13) + dTt(13) + dPt(13) + dMN(13) + dW(13) + dFAR(13) = 78 values
            return np.concatenate([primal, dTt, dPt, dMN, dW, dFAR])

        @custom_jvp
        def static_from_MN_jax(Tt, Pt, MN, W, composition):
            args = jnp.concatenate([jnp.array([Tt, Pt, MN, W]), composition])
            return pure_callback(_static_MN_impl,
                                 jax.ShapeDtypeStruct((13,), jnp.float64),
                                 args)

        @static_from_MN_jax.defjvp
        def static_from_MN_jvp(primals, tangents):
            Tt, Pt, MN, W, composition = primals
            Tt_dot, Pt_dot, MN_dot, W_dot, composition_dot = tangents

            args = jnp.concatenate([jnp.array([Tt, Pt, MN, W]), composition])
            result = pure_callback(_static_MN_derivs,
                                   jax.ShapeDtypeStruct((78,), jnp.float64),
                                   args)

            primal = result[0:13]
            dprops_dTt = result[13:26]
            dprops_dPt = result[26:39]
            dprops_dMN = result[39:52]
            dprops_dW = result[52:65]
            dprops_dFAR = result[65:78]

            # For TABULAR: FAR = composition[0], so FAR_dot = composition_dot[0]
            # For CEA: dprops_dFAR = 0, so this term vanishes regardless
            FAR_dot = composition_dot[0] if supports_FAR else 0.0

            props_dot = (dprops_dTt * Tt_dot + dprops_dPt * Pt_dot +
                        dprops_dMN * MN_dot + dprops_dW * W_dot +
                        dprops_dFAR * FAR_dot)
            return primal, props_dot

        self.static_from_MN = static_from_MN_jax

        # ---------------------------------------------------------------------
        # static_from_area wrapper
        # Returns: [Ts, Ps, hs, rhos, MN, V, Vsonic, area, gamma, Cp, Cv, S, R]
        # Accepts composition as fifth argument
        # ---------------------------------------------------------------------
        def _static_area_impl(args):
            """Pure Python implementation."""
            Tt, Pt, area, W = float(args[0]), float(args[1]), float(args[2]), float(args[3])
            composition = np.array(args[4:])
            FAR = extract_FAR(composition)
            if supports_FAR:
                props = thermo.static_from_area(Tt, Pt, area, W, FAR=FAR)
            else:
                props = thermo.static_from_area(Tt, Pt, area, W)
            return np.array([props.Ts, props.Ps, props.hs, props.rhos,
                            props.MN, props.V, props.Vsonic, props.area,
                            props.gamma, props.Cp, props.Cv, props.S, props.R])

        def _static_area_derivs(args):
            """Compute static props and Jacobian (13 outputs x 5 inputs)."""
            Tt, Pt, area, W = float(args[0]), float(args[1]), float(args[2]), float(args[3])
            composition = np.array(args[4:])
            FAR = extract_FAR(composition)

            # Check cache first
            if 'static_area' in self._cache:
                cTt, cPt, carea, cW, cFAR, sprops_arr, dTt, dPt, darea, dW, dFAR = self._cache['static_area']
                if (abs(cTt - Tt) < 1e-10 and abs(cPt - Pt) < 1e-10 and
                    abs(carea - area) < 1e-10 and abs(cW - W) < 1e-10 and (cFAR is None or abs(cFAR - FAR) < 1e-10)):
                    return np.concatenate([sprops_arr, dTt, dPt, darea, dW, dFAR])

            # Compute fresh (fallback)
            if supports_FAR:
                props = thermo.static_from_area(Tt, Pt, area, W, FAR=FAR)
            else:
                props = thermo.static_from_area(Tt, Pt, area, W)
            primal = [props.Ts, props.Ps, props.hs, props.rhos,
                      props.MN, props.V, props.Vsonic, props.area,
                      props.gamma, props.Cp, props.Cv, props.S, props.R]

            keys = ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                    'gamma', 'Cp', 'Cv', 'S', 'R']

            try:
                if supports_FAR:
                    thermo.linearize_static_area(Tt, Pt, area, W, FAR)
                    jvp_Tt = thermo.jvp_static_area(1.0, 0.0, 0.0, 0.0, 0.0)
                    jvp_Pt = thermo.jvp_static_area(0.0, 1.0, 0.0, 0.0, 0.0)
                    jvp_area = thermo.jvp_static_area(0.0, 0.0, 1.0, 0.0, 0.0)
                    jvp_W = thermo.jvp_static_area(0.0, 0.0, 0.0, 1.0, 0.0)
                    jvp_FAR = thermo.jvp_static_area(0.0, 0.0, 0.0, 0.0, 1.0)
                else:
                    thermo.linearize_static_area(Tt, Pt, area, W)
                    jvp_Tt = thermo.jvp_static_area(1.0, 0.0, 0.0, 0.0)
                    jvp_Pt = thermo.jvp_static_area(0.0, 1.0, 0.0, 0.0)
                    jvp_area = thermo.jvp_static_area(0.0, 0.0, 1.0, 0.0)
                    jvp_W = thermo.jvp_static_area(0.0, 0.0, 0.0, 1.0)
                    jvp_FAR = {k: 0.0 for k in keys}  # FAR derivative is zero for CEA
                dTt = [jvp_Tt[k] for k in keys]
                dPt = [jvp_Pt[k] for k in keys]
                darea = [jvp_area[k] for k in keys]
                dW = [jvp_W[k] for k in keys]
                dFAR = [jvp_FAR[k] for k in keys]
            except NotImplementedError:
                # Finite difference fallback
                eps = 1e-6
                if supports_FAR:
                    def compute(Tt_, Pt_, area_, W_, FAR_):
                        p = thermo.static_from_area(Tt_, Pt_, area_, W_, FAR=FAR_)
                        return [p.Ts, p.Ps, p.hs, p.rhos, p.MN, p.V, p.Vsonic, p.area,
                                p.gamma, p.Cp, p.Cv, p.S, p.R]
                    base = np.array(primal)
                    dTt = (np.array(compute(Tt+eps, Pt, area, W, FAR)) - base) / eps
                    dPt = (np.array(compute(Tt, Pt+eps, area, W, FAR)) - base) / eps
                    darea = (np.array(compute(Tt, Pt, area+eps, W, FAR)) - base) / eps
                    dW = (np.array(compute(Tt, Pt, area, W+eps, FAR)) - base) / eps
                    dFAR = (np.array(compute(Tt, Pt, area, W, FAR+eps)) - base) / eps
                else:
                    def compute(Tt_, Pt_, area_, W_):
                        p = thermo.static_from_area(Tt_, Pt_, area_, W_)
                        return [p.Ts, p.Ps, p.hs, p.rhos, p.MN, p.V, p.Vsonic, p.area,
                                p.gamma, p.Cp, p.Cv, p.S, p.R]
                    base = np.array(primal)
                    dTt = (np.array(compute(Tt+eps, Pt, area, W)) - base) / eps
                    dPt = (np.array(compute(Tt, Pt+eps, area, W)) - base) / eps
                    darea = (np.array(compute(Tt, Pt, area+eps, W)) - base) / eps
                    dW = (np.array(compute(Tt, Pt, area, W+eps)) - base) / eps
                    dFAR = np.zeros(len(keys))  # FAR derivative is zero for CEA

            # Return: primal(13) + dTt(13) + dPt(13) + darea(13) + dW(13) + dFAR(13) = 78 values
            return np.concatenate([primal, dTt, dPt, darea, dW, dFAR])

        @custom_jvp
        def static_from_area_jax(Tt, Pt, area, W, composition):
            args = jnp.concatenate([jnp.array([Tt, Pt, area, W]), composition])
            return pure_callback(_static_area_impl,
                                 jax.ShapeDtypeStruct((13,), jnp.float64),
                                 args)

        @static_from_area_jax.defjvp
        def static_from_area_jvp(primals, tangents):
            Tt, Pt, area, W, composition = primals
            Tt_dot, Pt_dot, area_dot, W_dot, composition_dot = tangents

            args = jnp.concatenate([jnp.array([Tt, Pt, area, W]), composition])
            result = pure_callback(_static_area_derivs,
                                   jax.ShapeDtypeStruct((78,), jnp.float64),
                                   args)

            primal = result[0:13]
            dprops_dTt = result[13:26]
            dprops_dPt = result[26:39]
            dprops_darea = result[39:52]
            dprops_dW = result[52:65]
            dprops_dFAR = result[65:78]

            # For TABULAR: FAR = composition[0], so FAR_dot = composition_dot[0]
            # For CEA: dprops_dFAR = 0, so this term vanishes regardless
            FAR_dot = composition_dot[0] if supports_FAR else 0.0

            props_dot = (dprops_dTt * Tt_dot + dprops_dPt * Pt_dot +
                        dprops_darea * area_dot + dprops_dW * W_dot +
                        dprops_dFAR * FAR_dot)
            return primal, props_dot

        self.static_from_area = static_from_area_jax
