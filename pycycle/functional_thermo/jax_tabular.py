"""
Pure JAX implementation of tabular thermodynamic interpolation.

This module provides JAX-traceable versions of the tabular thermo operations,
eliminating the need for pure_callback and enabling efficient JIT compilation.
"""

import time
import jax
import jax.numpy as jnp


# =============================================================================
# Thermo Profiling Infrastructure
# =============================================================================

_thermo_profiling_enabled = False

_thermo_stats = {
    # Call counts
    'props_TP_calls': 0,
    'T_from_hP_calls': 0,
    'static_from_MN_calls': 0,
    'static_from_area_calls': 0,
    # Timing (only when profiling enabled)
    'props_TP_time': 0.0,
    'T_from_hP_time': 0.0,
    'static_from_MN_time': 0.0,
    'static_from_area_time': 0.0,
    # Newton iteration tracking (filled by debug runs)
    'T_from_hP_total_iters': 0,
    'static_from_MN_total_iters': 0,
    'static_from_area_total_iters': 0,
}


def enable_thermo_profiling():
    """Enable detailed thermo profiling (adds timing overhead)."""
    global _thermo_profiling_enabled
    _thermo_profiling_enabled = True


def disable_thermo_profiling():
    """Disable thermo profiling."""
    global _thermo_profiling_enabled
    _thermo_profiling_enabled = False


def reset_thermo_stats():
    """Reset all thermo profiling statistics."""
    for key in _thermo_stats:
        _thermo_stats[key] = 0.0 if 'time' in key else 0


def get_thermo_stats():
    """Return a copy of the thermo statistics dictionary."""
    return dict(_thermo_stats)


def print_thermo_stats():
    """Print detailed thermo profiling statistics."""
    stats = _thermo_stats
    print("\n=== JaxTabularThermo Profiling Statistics ===")

    for method in ['props_TP', 'T_from_hP', 'static_from_MN', 'static_from_area']:
        calls = stats[f'{method}_calls']
        time_ms = stats[f'{method}_time'] * 1000
        if calls > 0:
            avg_ms = time_ms / calls
            print(f"  {method}:")
            print(f"    calls: {calls}")
            print(f"    total time: {time_ms:.3f} ms")
            print(f"    avg time: {avg_ms:.3f} ms")

            # Show Newton iterations if available
            iter_key = f'{method}_total_iters'
            if iter_key in stats and stats[iter_key] > 0:
                avg_iters = stats[iter_key] / calls
                print(f"    avg Newton iters: {avg_iters:.1f}")

    print("=============================================\n")


class JaxTrilinearInterp:
    """
    Pure JAX trilinear interpolator for 3D structured grids.

    This replaces the numpy-based MultiOutputTrilinearInterp with JAX operations
    that can be traced and JIT-compiled.

    Parameters
    ----------
    grid : tuple of ndarray
        (FAR_grid, P_grid, T_grid) - 1D arrays of grid points
    values_dict : dict of ndarray
        {'h': h_table, 'S': S_table, ...} - 3D arrays of property values
    """

    def __init__(self, grid, values_dict):
        # Convert grid arrays to JAX arrays
        self.grid = tuple(jnp.array(g) for g in grid)
        self.grid_sizes = tuple(len(g) for g in grid)

        # Store property names and stack values for efficient access
        self.property_names = list(values_dict.keys())

        # Stack all value tables: shape (n_props, nFAR, nP, nT)
        self._stacked_values = jnp.stack(
            [jnp.array(values_dict[name]) for name in self.property_names],
            axis=0
        )

    def _find_cell_idx(self, x, grid):
        """Find cell index using searchsorted equivalent in JAX."""
        # searchsorted returns index where element would be inserted
        idx = jnp.searchsorted(grid, x, side='right') - 1
        # Clamp to valid range [0, len-2] for interpolation
        return jnp.clip(idx, 0, len(grid) - 2)

    def _find_cell_and_coords(self, point):
        """Find cell indices and compute normalized coordinates.

        Returns
        -------
        tuple
            (i_FAR, i_P, i_T, xd, yd, zd, dy, dz) where:
            - i_FAR, i_P, i_T: cell indices
            - xd, yd, zd: normalized coordinates [0, 1] within cell
            - dy, dz: grid spacing (for derivative computation)
        """
        FAR, P, T = point[0], point[1], point[2]

        i_FAR = self._find_cell_idx(FAR, self.grid[0])
        i_P = self._find_cell_idx(P, self.grid[1])
        i_T = self._find_cell_idx(T, self.grid[2])

        x0, x1 = self.grid[0][i_FAR], self.grid[0][i_FAR + 1]
        y0, y1 = self.grid[1][i_P], self.grid[1][i_P + 1]
        z0, z1 = self.grid[2][i_T], self.grid[2][i_T + 1]

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        xd = (FAR - x0) / dx
        yd = (P - y0) / dy
        zd = (T - z0) / dz

        return i_FAR, i_P, i_T, xd, yd, zd, dy, dz

    def _trilinear_core(self, c000, c001, c010, c011, c100, c101, c110, c111, xd, yd, zd):
        """Compute trilinear interpolation from 8 corner values.

        Returns
        -------
        tuple
            (values, c0, c1, c00, c01, c10, c11) where values is the interpolated
            result and the intermediate values are returned for derivative computation.
        """
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        values = c0 * (1 - zd) + c1 * zd

        return values, c0, c1, c00, c01, c10, c11

    def interpolate(self, point):
        """
        Interpolate all properties at the given point.

        Parameters
        ----------
        point : array-like
            (FAR, P, T) coordinates

        Returns
        -------
        dict
            Property values at the interpolation point
        """
        i_FAR, i_P, i_T, xd, yd, zd, _, _ = self._find_cell_and_coords(point)

        # Get corner values for all properties at once
        # _stacked_values shape: (n_props, nFAR, nP, nT)
        c000 = self._stacked_values[:, i_FAR, i_P, i_T]
        c001 = self._stacked_values[:, i_FAR, i_P, i_T + 1]
        c010 = self._stacked_values[:, i_FAR, i_P + 1, i_T]
        c011 = self._stacked_values[:, i_FAR, i_P + 1, i_T + 1]
        c100 = self._stacked_values[:, i_FAR + 1, i_P, i_T]
        c101 = self._stacked_values[:, i_FAR + 1, i_P, i_T + 1]
        c110 = self._stacked_values[:, i_FAR + 1, i_P + 1, i_T]
        c111 = self._stacked_values[:, i_FAR + 1, i_P + 1, i_T + 1]

        values, *_ = self._trilinear_core(c000, c001, c010, c011, c100, c101, c110, c111, xd, yd, zd)

        return {name: values[i] for i, name in enumerate(self.property_names)}

    def interpolate_single(self, point, prop_idx):
        """
        Interpolate a single property at the given point.

        More efficient than interpolate() when only one property is needed.

        Parameters
        ----------
        point : array-like
            (FAR, P, T) coordinates
        prop_idx : int
            Index of the property to interpolate

        Returns
        -------
        float
            Property value at the interpolation point
        """
        i_FAR, i_P, i_T, xd, yd, zd, _, _ = self._find_cell_and_coords(point)

        # Get corner values for single property
        vals = self._stacked_values[prop_idx]
        c000 = vals[i_FAR, i_P, i_T]
        c001 = vals[i_FAR, i_P, i_T + 1]
        c010 = vals[i_FAR, i_P + 1, i_T]
        c011 = vals[i_FAR, i_P + 1, i_T + 1]
        c100 = vals[i_FAR + 1, i_P, i_T]
        c101 = vals[i_FAR + 1, i_P, i_T + 1]
        c110 = vals[i_FAR + 1, i_P + 1, i_T]
        c111 = vals[i_FAR + 1, i_P + 1, i_T + 1]

        value, *_ = self._trilinear_core(c000, c001, c010, c011, c100, c101, c110, c111, xd, yd, zd)

        return value

    def interpolate_with_derivs(self, point):
        """
        Interpolate all properties and compute analytical derivatives w.r.t. P and T.

        Uses the analytical derivatives of trilinear interpolation to avoid
        finite difference approximations.

        Parameters
        ----------
        point : array-like
            (FAR, P, T) coordinates

        Returns
        -------
        values : dict
            Property values at the interpolation point
        dvalues_dP : dict
            Derivatives of properties w.r.t. P (second coordinate)
        dvalues_dT : dict
            Derivatives of properties w.r.t. T (third coordinate)
        """
        i_FAR, i_P, i_T, xd, yd, zd, dy, dz = self._find_cell_and_coords(point)

        # Get corner values for all properties at once
        # _stacked_values shape: (n_props, nFAR, nP, nT)
        c000 = self._stacked_values[:, i_FAR, i_P, i_T]
        c001 = self._stacked_values[:, i_FAR, i_P, i_T + 1]
        c010 = self._stacked_values[:, i_FAR, i_P + 1, i_T]
        c011 = self._stacked_values[:, i_FAR, i_P + 1, i_T + 1]
        c100 = self._stacked_values[:, i_FAR + 1, i_P, i_T]
        c101 = self._stacked_values[:, i_FAR + 1, i_P, i_T + 1]
        c110 = self._stacked_values[:, i_FAR + 1, i_P + 1, i_T]
        c111 = self._stacked_values[:, i_FAR + 1, i_P + 1, i_T + 1]

        values, c0, c1, c00, c01, c10, c11 = self._trilinear_core(
            c000, c001, c010, c011, c100, c101, c110, c111, xd, yd, zd
        )

        # Analytical derivatives
        # d(value)/dT = d(value)/d(zd) * d(zd)/dT = (c1 - c0) / dz
        dvalues_dT = (c1 - c0) / dz

        # d(value)/dP = d(value)/d(yd) * d(yd)/dP
        # d(value)/d(yd) = d(c0)/d(yd) * (1 - zd) + d(c1)/d(yd) * zd
        #                = (c10 - c00) * (1 - zd) + (c11 - c01) * zd
        dc0_dyd = c10 - c00
        dc1_dyd = c11 - c01
        dvalues_dP = (dc0_dyd * (1 - zd) + dc1_dyd * zd) / dy

        return (
            {name: values[i] for i, name in enumerate(self.property_names)},
            {name: dvalues_dP[i] for i, name in enumerate(self.property_names)},
            {name: dvalues_dT[i] for i, name in enumerate(self.property_names)},
        )


class JaxTabularThermo:
    """
    Pure JAX implementation of tabular thermodynamics.

    This class provides JAX-traceable versions of the key thermo operations,
    enabling efficient JIT compilation without pure_callback overhead.

    Parameters
    ----------
    spec : dict
        Tabular data specification containing grid points and property values
    """

    def __init__(self, spec):
        """Initialize with tabular data specification."""
        # Create interpolator for all total properties
        grid = (spec['FAR'], spec['P'], spec['T'])
        values_dict = {
            'h': spec['h'],
            'S': spec['S'],
            'gamma': spec['gamma'],
            'Cp': spec['Cp'],
            'Cv': spec['Cv'],
            'rho': spec['rho'],
            'R': spec['R'],
        }
        self._interp = JaxTrilinearInterp(grid, values_dict)

        # Unit conversion factors (SI to English)
        # Assuming input_units='English' for pyCycle compatibility
        h_to_si = 2326.0  # Btu/lbm -> J/kg
        h_from_si = 1.0 / h_to_si
        P_to_si = 6894.76  # psi -> Pa
        T_to_si_scale = 5.0 / 9.0  # Rankine -> Kelvin scale
        S_from_si = 1.0 / 4186.8  # J/(kg*K) -> Btu/(lbm*R)
        rho_from_si = 0.062428  # kg/m^3 -> lbm/ft^3
        self._unit_conversions = (h_to_si, h_from_si, P_to_si, T_to_si_scale, S_from_si, rho_from_si)

        # Create JIT-compiled versions of the methods
        self._setup_jit_functions()
        self._setup_static_functions()

    def _setup_jit_functions(self):
        """Create JIT-compiled versions of thermo functions."""
        interp = self._interp
        h_to_si, h_from_si, P_to_si, T_to_si_scale, S_from_si, rho_from_si = self._unit_conversions

        @jax.jit
        def _props_TP_jit(T, P, FAR):
            """JIT-compiled props_TP."""
            T_si = T * T_to_si_scale
            P_si = P * P_to_si
            point = jnp.array([FAR, P_si, T_si])
            props_si = interp.interpolate(point)

            h = props_si['h'] * h_from_si
            S = props_si['S'] * S_from_si
            gamma = props_si['gamma']
            Cp = props_si['Cp'] * S_from_si
            Cv = props_si['Cv'] * S_from_si
            rho = props_si['rho'] * rho_from_si
            R = props_si['R'] * S_from_si

            return jnp.array([h, S, gamma, Cp, Cv, rho, R])

        @jax.jit
        def _T_from_hP_jit(h_target, P, FAR):
            """JIT-compiled T_from_hP with while_loop Newton solver."""
            h_target_si = h_target * h_to_si
            P_si = P * P_to_si

            # Initial guess
            T_si_init = jnp.clip(jnp.abs(h_target_si) / 1000.0 + 300.0, 300.0, 2000.0)

            def h_and_deriv(T_si):
                """Get h and dh/dT using analytical derivatives from interpolator."""
                point = jnp.array([FAR, P_si, T_si])
                props, _, dprops_dT = interp.interpolate_with_derivs(point)
                return props['h'], dprops_dT['h']

            def h_only(T_si):
                """Get h value only (more efficient for residual check)."""
                point = jnp.array([FAR, P_si, T_si])
                return interp.interpolate_single(point, 0)  # h is at index 0

            def cond_fn(state):
                T_si, residual, i = state
                return (jnp.abs(residual) > 1e-8) & (i < 20)

            def body_fn(state):
                T_si, _, i = state

                # Get h and analytical dh/dT from interpolator
                h_si, dh_dT = h_and_deriv(T_si)
                residual = h_si - h_target_si
                dh_dT = jnp.where(jnp.abs(dh_dT) < 1e-20, 1e-20, dh_dT)

                # Newton step with bounds
                dx = -residual / dh_dT
                T_si_new = jnp.clip(T_si + dx, 160.0, 2400.0)

                # Compute new residual for convergence check
                residual_new = h_only(T_si_new) - h_target_si
                return (T_si_new, residual_new, i + 1)

            # Initialize state: (T_si, residual, iteration)
            residual_init = h_only(T_si_init) - h_target_si
            init_state = (T_si_init, residual_init, 0)

            # Run Newton iteration
            final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
            T_si = final_state[0]
            n_iters = final_state[2]

            return T_si / T_to_si_scale, n_iters

        self._props_TP_jit = _props_TP_jit
        self._T_from_hP_jit_with_iters = jax.jit(_T_from_hP_jit)

        # Wrapper that discards iteration count for normal use
        @jax.jit
        def _T_from_hP_jit_simple(h_target, P, FAR):
            result, _ = _T_from_hP_jit(h_target, P, FAR)
            return result

        self._T_from_hP_jit = _T_from_hP_jit_simple

    def T_from_hP(self, h_target, P, FAR):
        """
        Solve for temperature given enthalpy and pressure.

        Pure JAX implementation using Newton's method with JAX interpolation.

        Parameters
        ----------
        h_target : float
            Target enthalpy (English units: Btu/lbm)
        P : float
            Pressure (English units: psi)
        FAR : float
            Fuel-to-air ratio

        Returns
        -------
        float
            Temperature (English units: Rankine)
        """
        # Check if we're being traced by JAX (can't do Python profiling during tracing)
        is_tracing = isinstance(h_target, jax.core.Tracer)
        if not is_tracing:
            _thermo_stats['T_from_hP_calls'] += 1
            if _thermo_profiling_enabled:
                t0 = time.perf_counter()
                result, n_iters = self._T_from_hP_jit_with_iters(h_target, P, FAR)
                _thermo_stats['T_from_hP_time'] += time.perf_counter() - t0
                _thermo_stats['T_from_hP_total_iters'] += int(n_iters)
                return result
        return self._T_from_hP_jit(h_target, P, FAR)

    def props_TP(self, T, P, FAR):
        """
        Get all thermodynamic properties at given T, P, FAR.

        Parameters
        ----------
        T : float
            Temperature (English units: Rankine)
        P : float
            Pressure (English units: psi)
        FAR : float
            Fuel-to-air ratio

        Returns
        -------
        array
            [h, S, gamma, Cp, Cv, rho, R] in English units
        """
        is_tracing = isinstance(T, jax.core.Tracer)
        if not is_tracing:
            _thermo_stats['props_TP_calls'] += 1
            if _thermo_profiling_enabled:
                t0 = time.perf_counter()
                result = self._props_TP_jit(T, P, FAR)
                _thermo_stats['props_TP_time'] += time.perf_counter() - t0
                return result
        return self._props_TP_jit(T, P, FAR)

    def static_from_MN(self, Tt, Pt, MN, W, FAR):
        """
        Compute static properties from total conditions and Mach number.

        Pure JAX implementation using 2D Newton solver.

        Parameters
        ----------
        Tt : float
            Total temperature (English units: Rankine)
        Pt : float
            Total pressure (English units: psi)
        MN : float
            Mach number
        W : float
            Mass flow rate (English units: lbm/s)
        FAR : float
            Fuel-to-air ratio

        Returns
        -------
        array
            [Ts, Ps, hs, rhos, MN, V, Vsonic, area, gamma, Cp, Cv, S, R]
            in English units
        """
        is_tracing = isinstance(Tt, jax.core.Tracer)
        if not is_tracing:
            _thermo_stats['static_from_MN_calls'] += 1
            if _thermo_profiling_enabled:
                t0 = time.perf_counter()
                result = self._static_from_MN_jit(Tt, Pt, MN, W, FAR)
                _thermo_stats['static_from_MN_time'] += time.perf_counter() - t0
                return result
        return self._static_from_MN_jit(Tt, Pt, MN, W, FAR)

    def static_from_area(self, Tt, Pt, area, W, FAR):
        """
        Compute static properties from total conditions and flow area.

        Pure JAX implementation using nested Newton solvers.

        Parameters
        ----------
        Tt : float
            Total temperature (English units: Rankine)
        Pt : float
            Total pressure (English units: psi)
        area : float
            Flow area (English units: inch^2)
        W : float
            Mass flow rate (English units: lbm/s)
        FAR : float
            Fuel-to-air ratio

        Returns
        -------
        array
            [Ts, Ps, hs, rhos, MN, V, Vsonic, area, gamma, Cp, Cv, S, R]
            in English units
        """
        is_tracing = isinstance(Tt, jax.core.Tracer)
        if not is_tracing:
            _thermo_stats['static_from_area_calls'] += 1
            if _thermo_profiling_enabled:
                t0 = time.perf_counter()
                result = self._static_from_area_jit(Tt, Pt, area, W, FAR)
                _thermo_stats['static_from_area_time'] += time.perf_counter() - t0
                return result
        return self._static_from_area_jit(Tt, Pt, area, W, FAR)

    def _setup_static_functions(self):
        """Create JIT-compiled versions of static property functions."""
        interp = self._interp
        h_to_si, h_from_si, P_to_si, T_to_si_scale, S_from_si, rho_from_si = self._unit_conversions

        # Additional conversion factors for static properties
        # W: lbm/s -> kg/s
        W_to_si = 0.45359237
        # area: inch^2 -> m^2
        area_to_si = 0.00064516
        area_from_si = 1.0 / area_to_si
        # velocity: m/s -> ft/s
        V_from_si = 3.28084
        # R: J/(kg*K) -> ft*lbf/(lbm*R) for Vsonic calc
        # Actually for Vsonic = sqrt(gamma*R*T), R is in J/(kg*K), T in K, result in m/s
        # So we convert Vsonic from m/s to ft/s at the end

        def props_at_TP_si(T_si, P_si, FAR):
            """Get properties at given T, P in SI units."""
            point = jnp.array([FAR, P_si, T_si])
            props_si = interp.interpolate(point)
            return props_si

        def props_at_TP_si_with_derivs(T_si, P_si, FAR):
            """Get properties and analytical derivatives at given T, P in SI units.

            Returns
            -------
            props : dict
                Property values
            dprops_dP : dict
                Derivatives w.r.t. P (in SI units)
            dprops_dT : dict
                Derivatives w.r.t. T (in SI units)
            """
            point = jnp.array([FAR, P_si, T_si])
            return interp.interpolate_with_derivs(point)

        def static_from_MN_impl(Tt, Pt, MN, W, FAR):
            """Pure JAX static_from_MN implementation."""
            # Convert to SI
            Tt_si = Tt * T_to_si_scale
            Pt_si = Pt * P_to_si
            W_si = W * W_to_si

            # Get total properties
            tot_props = props_at_TP_si(Tt_si, Pt_si, FAR)
            ht_si = tot_props['h']
            S_total = tot_props['S']
            gamma_t = tot_props['gamma']
            R_t = tot_props['R']
            Cp_t = tot_props['Cp']
            Cv_t = tot_props['Cv']

            # Clamp MN to avoid numerical issues (but still run Newton solver)
            MN_clamped = jnp.maximum(MN, 1e-10)
            MN_sq = MN_clamped ** 2

            # Initial guess using ideal gas isentropic relations
            Ps0_si = Pt_si * (1.0 + (gamma_t - 1.0) / 2.0 * MN_sq) ** (-gamma_t / (gamma_t - 1.0))
            Ts0_si = Tt_si * (Ps0_si / Pt_si) ** ((gamma_t - 1.0) / gamma_t)

            # 2D Newton solver for (Ts, Ps) using while_loop
            # Residuals:
            #   R1 = S(Ts, Ps) - S_total  (entropy conservation)
            #   R2 = hs + MN²·γ·R·Ts/2 - ht  (energy conservation)
            #
            # State carries props and derivatives to avoid redundant lookups.
            # State indices:
            #   0: Ts_si, 1: Ps_si, 2: R1, 3: R2,
            #   4: S, 5: h, 6: gamma, 7: R,
            #   8: dS_dT, 9: dS_dP, 10: dh_dT, 11: dh_dP,
            #   12: dgamma_dT, 13: dgamma_dP, 14: dR_dT, 15: dR_dP,
            #   16: iteration

            def compute_state_from_TP(Ts_si, Ps_si, i):
                """Compute full state including props, derivs, and residuals."""
                props_s, dprops_dP, dprops_dT = props_at_TP_si_with_derivs(Ts_si, Ps_si, FAR)
                S_s = props_s['S']
                hs_si = props_s['h']
                gamma_s = props_s['gamma']
                R_s = props_s['R']

                # Residuals
                Vsonic_sq = gamma_s * R_s * Ts_si
                V_sq = MN_sq * Vsonic_sq
                R1 = S_s - S_total
                R2 = hs_si + 0.5 * V_sq - ht_si

                return jnp.array([
                    Ts_si, Ps_si, R1, R2,
                    S_s, hs_si, gamma_s, R_s,
                    dprops_dT['S'], dprops_dP['S'],
                    dprops_dT['h'], dprops_dP['h'],
                    dprops_dT['gamma'], dprops_dP['gamma'],
                    dprops_dT['R'], dprops_dP['R'],
                    i
                ])

            def cond_fn_2d(state):
                R1, R2, i = state[2], state[3], state[16]
                residual_norm = jnp.sqrt(R1**2 + R2**2)
                return (residual_norm > 1e-8) & (i < 20)

            def body_fn_2d(state):
                # Unpack state - props and derivs already computed
                Ts_si, Ps_si = state[0], state[1]
                R1, R2 = state[2], state[3]
                gamma_s, R_s = state[6], state[7]
                dS_dT, dS_dP = state[8], state[9]
                dh_dT, dh_dP = state[10], state[11]
                dgamma_dT, dgamma_dP = state[12], state[13]
                dR_dT, dR_dP = state[14], state[15]
                i = state[16]

                # Analytical Jacobian (no interpolation needed - already have derivs!)
                dR1_dT = dS_dT
                dR1_dP = dS_dP

                dVsq_dT = MN_sq * (dgamma_dT * R_s * Ts_si + gamma_s * dR_dT * Ts_si + gamma_s * R_s)
                dVsq_dP = MN_sq * (dgamma_dP * R_s * Ts_si + gamma_s * dR_dP * Ts_si)

                dR2_dT = dh_dT + 0.5 * dVsq_dT
                dR2_dP = dh_dP + 0.5 * dVsq_dP

                # Solve 2x2 system: J * dx = -R
                det = dR1_dT * dR2_dP - dR1_dP * dR2_dT
                det = jnp.where(jnp.abs(det) < 1e-20, 1e-20, det)

                dTs = (-R1 * dR2_dP + R2 * dR1_dP) / det
                dPs = (-R2 * dR1_dT + R1 * dR2_dT) / det

                # Clamp updates
                Ts_new = jnp.clip(Ts_si + dTs, 160.0, 2400.0)
                Ps_new = jnp.clip(Ps_si + dPs, 100.0, 1e8)

                # Compute new state (ONE interpolation call for next iteration)
                return compute_state_from_TP(Ts_new, Ps_new, i + 1)

            # Initialize state with ONE interpolation call
            init_state = compute_state_from_TP(Ts0_si, Ps0_si, 0.0)

            # Run 2D Newton iteration
            final_state = jax.lax.while_loop(cond_fn_2d, body_fn_2d, init_state)

            # Extract results from Newton solver
            Ts_newton = final_state[0]
            Ps_newton = final_state[1]
            S_newton = final_state[4]
            hs_newton = final_state[5]
            gamma_newton = final_state[6]
            R_newton = final_state[7]

            # Only Cp and Cv need a final lookup (not in state)
            props_final = props_at_TP_si(Ts_newton, Ps_newton, FAR)
            Cp_newton = props_final['Cp']
            Cv_newton = props_final['Cv']

            # Compute derived quantities for Newton solution
            rhos_newton = Ps_newton / (R_newton * Ts_newton)
            Vsonic_newton = jnp.sqrt(gamma_newton * R_newton * Ts_newton)
            V_newton = MN_clamped * Vsonic_newton
            area_newton = W_si / (rhos_newton * V_newton)

            # === Compute darea/dMN via implicit differentiation ===
            # Extract Jacobian entries from final state
            dS_dT_final = final_state[8]
            dS_dP_final = final_state[9]
            dh_dT_final = final_state[10]
            dh_dP_final = final_state[11]
            dgamma_dT_final = final_state[12]
            dgamma_dP_final = final_state[13]
            dR_dT_final = final_state[14]
            dR_dP_final = final_state[15]

            # Recompute dR2_dT and dR2_dP at converged point
            Vsonic_sq_newton = gamma_newton * R_newton * Ts_newton
            dVsq_dT_final = MN_sq * (dgamma_dT_final * R_newton * Ts_newton
                                     + gamma_newton * dR_dT_final * Ts_newton
                                     + gamma_newton * R_newton)
            dVsq_dP_final = MN_sq * (dgamma_dP_final * R_newton * Ts_newton
                                     + gamma_newton * dR_dP_final * Ts_newton)
            dR2_dT_final = dh_dT_final + 0.5 * dVsq_dT_final
            dR2_dP_final = dh_dP_final + 0.5 * dVsq_dP_final

            # Solve 2x2 system for dTs/dMN and dPs/dMN:
            # [dS_dT   dS_dP ] [dTs/dMN]   [     0      ]
            # [dR2_dT  dR2_dP] [dPs/dMN] = [-MN*Vsonic²]
            det_final = dS_dT_final * dR2_dP_final - dS_dP_final * dR2_dT_final
            det_final = jnp.where(jnp.abs(det_final) < 1e-30, 1e-30, det_final)

            rhs_MN = -MN_clamped * Vsonic_sq_newton  # ∂R2/∂MN = MN * Vsonic²
            dTs_dMN = (0.0 * dR2_dP_final - rhs_MN * dS_dP_final) / det_final
            dPs_dMN = (dS_dT_final * rhs_MN - 0.0 * dR2_dT_final) / det_final

            # Chain rule for gamma and R
            dgamma_dMN = dgamma_dT_final * dTs_dMN + dgamma_dP_final * dPs_dMN
            dR_dMN = dR_dT_final * dTs_dMN + dR_dP_final * dPs_dMN

            # Derivatives of rho and Vsonic (in SI units)
            # rho = Ps / (R * Ts)
            drho_dMN_newton = rhos_newton * (dPs_dMN / Ps_newton - dR_dMN / R_newton - dTs_dMN / Ts_newton)

            # Vsonic = sqrt(gamma * R * Ts)
            dVsonic_dMN_newton = 0.5 * Vsonic_newton * (dgamma_dMN / gamma_newton + dR_dMN / R_newton + dTs_dMN / Ts_newton)

            # darea/dMN in SI units: area = W / (rho * MN * Vsonic)
            darea_dMN_newton = area_newton * (-1.0 / MN_clamped - drho_dMN_newton / rhos_newton - dVsonic_dMN_newton / Vsonic_newton)

            # Zero MN case: static = total (cheap to compute)
            rhos_zero = Pt_si / (R_t * Tt_si)
            Vsonic_zero = jnp.sqrt(gamma_t * R_t * Tt_si)

            # Use jnp.where to blend results (avoids tracing both lax.cond branches)
            is_zero_MN = MN < 1e-10

            # For zero MN case, derivative is undefined/infinite, use large value
            darea_dMN_si = jnp.where(is_zero_MN, -1e30, darea_dMN_newton)

            Ts_si = jnp.where(is_zero_MN, Tt_si, Ts_newton)
            Ps_si = jnp.where(is_zero_MN, Pt_si, Ps_newton)
            hs_si = jnp.where(is_zero_MN, ht_si, hs_newton)
            rhos_si = jnp.where(is_zero_MN, rhos_zero, rhos_newton)
            MN_out = jnp.where(is_zero_MN, 0.0, MN)
            V_si = jnp.where(is_zero_MN, 0.0, V_newton)
            Vsonic_si = jnp.where(is_zero_MN, Vsonic_zero, Vsonic_newton)
            area_si = jnp.where(is_zero_MN, jnp.inf, area_newton)
            gamma_out = jnp.where(is_zero_MN, gamma_t, gamma_newton)
            Cp_out = jnp.where(is_zero_MN, Cp_t, Cp_newton)
            Cv_out = jnp.where(is_zero_MN, Cv_t, Cv_newton)
            S_out = jnp.where(is_zero_MN, S_total, S_newton)
            R_out = jnp.where(is_zero_MN, R_t, R_newton)

            result_si = jnp.array([Ts_si, Ps_si, hs_si, rhos_si, MN_out, V_si, Vsonic_si,
                                   area_si, gamma_out, Cp_out, Cv_out, S_out, R_out])

            # Convert back to English units
            Ts = result_si[0] / T_to_si_scale
            Ps = result_si[1] / P_to_si
            hs = result_si[2] * h_from_si
            rhos = result_si[3] * rho_from_si
            MN_out = result_si[4]
            V = result_si[5] * V_from_si
            Vsonic = result_si[6] * V_from_si
            area = result_si[7] * area_from_si
            gamma_out = result_si[8]
            Cp = result_si[9] * S_from_si
            Cv = result_si[10] * S_from_si
            S = result_si[11] * S_from_si
            R = result_si[12] * S_from_si

            # Convert darea_dMN to English units (area_from_si, MN is dimensionless)
            darea_dMN = darea_dMN_si * area_from_si

            return jnp.array([Ts, Ps, hs, rhos, MN_out, V, Vsonic, area,
                              gamma_out, Cp, Cv, S, R, darea_dMN])

        def static_from_area_impl(Tt, Pt, area, W, FAR):
            """
            Pure JAX static_from_area using unified 3D Newton on (Ts, Ps, MN).

            Instead of nested Newton loops (outer on MN, inner 2D on Ts/Ps),
            this solves all three unknowns simultaneously in a single Newton loop.

            Residuals:
                R1 = S(Ts, Ps) - S_total        (entropy conservation)
                R2 = hs + 0.5*MN²*γ*R*Ts - ht   (energy conservation)
                R3 = area_computed - area        (mass continuity)

            Jacobian (3x3):
                [dS_dT      dS_dP      0           ]
                [dR2_dT     dR2_dP     MN*γ*R*Ts   ]
                [dR3_dT     dR3_dP     -area/MN    ]
            """
            # Convert to SI
            Tt_si = Tt * T_to_si_scale
            Pt_si = Pt * P_to_si
            W_si = W * W_to_si
            area_si = area * area_to_si

            # Get total properties
            tot_props = props_at_TP_si(Tt_si, Pt_si, FAR)
            ht_si = tot_props['h']
            S_total = tot_props['S']
            gamma_t = tot_props['gamma']
            R_t = tot_props['R']

            # Initial guesses using ideal gas isentropic relations
            MN0 = 0.5
            MN_sq = MN0 ** 2
            Ps0_si = Pt_si * (1.0 + (gamma_t - 1.0) / 2.0 * MN_sq) ** (-gamma_t / (gamma_t - 1.0))
            Ts0_si = Tt_si * (Ps0_si / Pt_si) ** ((gamma_t - 1.0) / gamma_t)

            # 3D Newton solver for (Ts, Ps, MN) using while_loop
            # State indices:
            #   0: Ts_si, 1: Ps_si, 2: MN
            #   3: R1, 4: R2, 5: R3 (residuals)
            #   6: S, 7: hs, 8: gamma, 9: R (properties)
            #   10: dS_dT, 11: dS_dP, 12: dh_dT, 13: dh_dP
            #   14: dgamma_dT, 15: dgamma_dP, 16: dR_dT, 17: dR_dP
            #   18: area_computed, 19: Vsonic_sq
            #   20: iteration count

            def compute_state_3d(Ts_si, Ps_si, MN, i):
                """Compute full 3D state including props, derivs, and all three residuals."""
                # Get properties and derivatives from interpolator
                props_s, dprops_dP, dprops_dT = props_at_TP_si_with_derivs(Ts_si, Ps_si, FAR)
                S_s = props_s['S']
                hs_si = props_s['h']
                gamma_s = props_s['gamma']
                R_s = props_s['R']

                # Clamp MN for numerical stability
                MN_clamped = jnp.maximum(MN, 1e-10)
                MN_sq = MN_clamped ** 2

                # Derived quantities
                Vsonic_sq = gamma_s * R_s * Ts_si
                V_sq = MN_sq * Vsonic_sq
                rho_s = Ps_si / (R_s * Ts_si)
                V_s = MN_clamped * jnp.sqrt(Vsonic_sq)
                area_computed = W_si / (rho_s * V_s)

                # Residuals
                R1 = S_s - S_total
                R2 = hs_si + 0.5 * V_sq - ht_si
                R3 = area_computed - area_si

                return jnp.array([
                    Ts_si, Ps_si, MN,
                    R1, R2, R3,
                    S_s, hs_si, gamma_s, R_s,
                    dprops_dT['S'], dprops_dP['S'],
                    dprops_dT['h'], dprops_dP['h'],
                    dprops_dT['gamma'], dprops_dP['gamma'],
                    dprops_dT['R'], dprops_dP['R'],
                    area_computed, Vsonic_sq,
                    i
                ])

            def cond_fn_3d(state):
                R1, R2, R3, i = state[3], state[4], state[5], state[20]
                residual_norm = jnp.sqrt(R1**2 + R2**2 + R3**2)
                return (residual_norm > 1e-8) & (i < 30)

            def body_fn_3d(state):
                # Unpack state
                Ts_si, Ps_si, MN = state[0], state[1], state[2]
                R1, R2, R3 = state[3], state[4], state[5]
                gamma_s, R_s = state[8], state[9]
                dS_dT, dS_dP = state[10], state[11]
                dh_dT, dh_dP = state[12], state[13]
                dgamma_dT, dgamma_dP = state[14], state[15]
                dR_dT, dR_dP = state[16], state[17]
                area_computed, Vsonic_sq = state[18], state[19]
                i = state[20]

                MN_clamped = jnp.maximum(MN, 1e-10)
                MN_sq = MN_clamped ** 2

                # === Build 3x3 Jacobian ===
                # Row 1: dR1/d(Ts, Ps, MN) - entropy residual
                J11 = dS_dT  # dR1/dTs
                J12 = dS_dP  # dR1/dPs
                J13 = 0.0    # dR1/dMN (S doesn't depend on MN directly)

                # Row 2: dR2/d(Ts, Ps, MN) - energy residual
                # R2 = hs + 0.5*MN²*γ*R*Ts - ht
                dVsq_dT = MN_sq * (dgamma_dT * R_s * Ts_si + gamma_s * dR_dT * Ts_si + gamma_s * R_s)
                dVsq_dP = MN_sq * (dgamma_dP * R_s * Ts_si + gamma_s * dR_dP * Ts_si)
                J21 = dh_dT + 0.5 * dVsq_dT  # dR2/dTs
                J22 = dh_dP + 0.5 * dVsq_dP  # dR2/dPs
                J23 = MN_clamped * Vsonic_sq  # dR2/dMN = MN * γ * R * Ts

                # Row 3: dR3/d(Ts, Ps, MN) - area residual
                # area = W / (ρ * V) = W / (ρ * MN * Vsonic)
                # Using logarithmic differentiation:
                # d(ln area)/dTs = 0.5*(dR_dT/R + 1/Ts - dgamma_dT/γ)
                # d(ln area)/dPs = -1/Ps + 0.5*(dR_dP/R - dgamma_dP/γ)
                # d(ln area)/dMN = -1/MN
                J31 = 0.5 * area_computed * (dR_dT / R_s + 1.0 / Ts_si - dgamma_dT / gamma_s)
                J32 = area_computed * (-1.0 / Ps_si + 0.5 * dR_dP / R_s - 0.5 * dgamma_dP / gamma_s)
                J33 = -area_computed / MN_clamped  # dR3/dMN

                # === Solve 3x3 system J * dx = -R using block elimination ===
                # Since J13 = 0, we can reduce to 2x2 + back-substitution
                #
                # Row 1: J11*dTs + J12*dPs = -R1
                # Row 2: J21*dTs + J22*dPs + J23*dMN = -R2
                # Row 3: J31*dTs + J32*dPs + J33*dMN = -R3
                #
                # From Row 1: dTs = (-R1 - J12*dPs) / J11
                # Substitute into Rows 2 & 3 to get 2x2 system for (dPs, dMN)

                J11_safe = jnp.where(jnp.abs(J11) < 1e-20, 1e-20, J11)

                # Modified coefficients for 2x2 system
                J22_mod = J22 - J21 * J12 / J11_safe
                J32_mod = J32 - J31 * J12 / J11_safe
                R2_mod = -R2 + J21 * R1 / J11_safe
                R3_mod = -R3 + J31 * R1 / J11_safe

                # Solve 2x2: [J22_mod, J23; J32_mod, J33] * [dPs; dMN] = [R2_mod; R3_mod]
                det2 = J22_mod * J33 - J23 * J32_mod
                det2_safe = jnp.where(jnp.abs(det2) < 1e-20, 1e-20, det2)

                dPs = (R2_mod * J33 - J23 * R3_mod) / det2_safe
                dMN = (J22_mod * R3_mod - R2_mod * J32_mod) / det2_safe

                # Back-substitute to get dTs
                dTs = (-R1 - J12 * dPs) / J11_safe

                # Apply updates with bounds
                Ts_new = jnp.clip(Ts_si + dTs, 160.0, 2400.0)
                Ps_new = jnp.clip(Ps_si + dPs, 100.0, 1e8)
                MN_new = jnp.clip(MN + dMN, 0.01, 0.99)  # Subsonic only

                return compute_state_3d(Ts_new, Ps_new, MN_new, i + 1)

            # Initialize and run Newton iteration
            init_state = compute_state_3d(Ts0_si, Ps0_si, MN0, 0.0)
            final_state = jax.lax.while_loop(cond_fn_3d, body_fn_3d, init_state)

            # Extract converged solution
            Ts_si = final_state[0]
            Ps_si = final_state[1]
            MN_final = final_state[2]
            hs_si = final_state[7]
            gamma_s = final_state[8]
            R_s = final_state[9]
            S_s = final_state[6]

            # Compute remaining properties
            rho_s = Ps_si / (R_s * Ts_si)
            Vsonic = jnp.sqrt(gamma_s * R_s * Ts_si)
            V_s = MN_final * Vsonic
            area_final = W_si / (rho_s * V_s)

            # Get Cp and Cv (not in state)
            props_final = props_at_TP_si(Ts_si, Ps_si, FAR)
            Cp_s = props_final['Cp']
            Cv_s = props_final['Cv']

            # Convert to English units
            Ts = Ts_si / T_to_si_scale
            Ps = Ps_si / P_to_si
            hs = hs_si * h_from_si
            rhos = rho_s * rho_from_si
            V = V_s * V_from_si
            Vsonic_eng = Vsonic * V_from_si
            area_eng = area_final * area_from_si
            gamma_out = gamma_s
            Cp = Cp_s * S_from_si
            Cv = Cv_s * S_from_si
            S = S_s * S_from_si
            R_out = R_s * S_from_si

            return jnp.array([Ts, Ps, hs, rhos, MN_final, V, Vsonic_eng, area_eng,
                              gamma_out, Cp, Cv, S, R_out])

        # JIT compile
        self._static_from_MN_jit = jax.jit(static_from_MN_impl)
        self._static_from_area_jit = jax.jit(static_from_area_impl)
