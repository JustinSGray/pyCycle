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
        W_to_si = 0.45359237      # lbm/s -> kg/s
        area_to_si = 0.00064516   # inch^2 -> m^2
        area_from_si = 1.0 / area_to_si
        V_from_si = 3.28084       # m/s -> ft/s

        # =====================================================================
        # Shared Helper Functions
        # =====================================================================

        def props_at_TP_si(T_si, P_si, FAR):
            """Get properties at given T, P in SI units."""
            point = jnp.array([FAR, P_si, T_si])
            return interp.interpolate(point)

        def props_at_TP_si_with_derivs(T_si, P_si, FAR):
            """Get properties and analytical derivatives at given T, P in SI units."""
            point = jnp.array([FAR, P_si, T_si])
            return interp.interpolate_with_derivs(point)

        def compute_isentropic_initial_guess(Tt_si, Pt_si, gamma_t, MN):
            """Compute initial (Ts, Ps) guess using ideal gas isentropic relations."""
            MN_sq = MN ** 2
            Ps0_si = Pt_si * (1.0 + (gamma_t - 1.0) / 2.0 * MN_sq) ** (-gamma_t / (gamma_t - 1.0))
            Ts0_si = Tt_si * (Ps0_si / Pt_si) ** ((gamma_t - 1.0) / gamma_t)
            return Ts0_si, Ps0_si

        def compute_R1_R2_jacobian(Ts_si, MN_sq, gamma_s, R_s,
                                    dS_dT, dS_dP, dh_dT, dh_dP,
                                    dgamma_dT, dgamma_dP, dR_dT, dR_dP):
            """
            Compute Jacobian entries for R1 (entropy) and R2 (energy) residuals.

            Returns (dR1_dT, dR1_dP, dR2_dT, dR2_dP) - the 2x2 Jacobian for the
            entropy and energy equations w.r.t. (Ts, Ps).
            """
            # R1 = S - S_total
            dR1_dT = dS_dT
            dR1_dP = dS_dP

            # R2 = hs + 0.5*MN²*γ*R*Ts - ht
            # V² = MN² * Vsonic² = MN² * γ * R * Ts
            dVsq_dT = MN_sq * (dgamma_dT * R_s * Ts_si + gamma_s * dR_dT * Ts_si + gamma_s * R_s)
            dVsq_dP = MN_sq * (dgamma_dP * R_s * Ts_si + gamma_s * dR_dP * Ts_si)
            dR2_dT = dh_dT + 0.5 * dVsq_dT
            dR2_dP = dh_dP + 0.5 * dVsq_dP

            return dR1_dT, dR1_dP, dR2_dT, dR2_dP

        def solve_2x2(J11, J12, J21, J22, R1, R2):
            """Solve 2x2 linear system: J * dx = -R, returns (dx1, dx2)."""
            det = J11 * J22 - J12 * J21
            det = jnp.where(jnp.abs(det) < 1e-20, 1e-20, det)
            dx1 = (-R1 * J22 + R2 * J12) / det
            dx2 = (-R2 * J11 + R1 * J21) / det
            return dx1, dx2

        def convert_static_to_english(Ts_si, Ps_si, hs_si, rho_s, MN, V_s, Vsonic_s,
                                       area_s, gamma_s, Cp_s, Cv_s, S_s, R_s):
            """Convert static properties from SI to English units."""
            return jnp.array([
                Ts_si / T_to_si_scale,      # Ts (degR)
                Ps_si / P_to_si,            # Ps (psi)
                hs_si * h_from_si,          # hs (Btu/lbm)
                rho_s * rho_from_si,        # rhos (lbm/ft³)
                MN,                         # MN (dimensionless)
                V_s * V_from_si,            # V (ft/s)
                Vsonic_s * V_from_si,       # Vsonic (ft/s)
                area_s * area_from_si,      # area (in²)
                gamma_s,                    # gamma (dimensionless)
                Cp_s * S_from_si,           # Cp (Btu/(lbm·R))
                Cv_s * S_from_si,           # Cv (Btu/(lbm·R))
                S_s * S_from_si,            # S (Btu/(lbm·R))
                R_s * S_from_si,            # R (Btu/(lbm·R))
            ])

        # =====================================================================
        # static_from_MN: 2D Newton solver for (Ts, Ps) given MN
        # =====================================================================

        def static_from_MN_impl(Tt, Pt, MN, W, FAR):
            """Pure JAX static_from_MN: solve for (Ts, Ps) given MN."""
            # Convert inputs to SI
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

            # Clamp MN
            MN_clamped = jnp.maximum(MN, 1e-10)
            MN_sq = MN_clamped ** 2

            # Initial guess
            Ts0_si, Ps0_si = compute_isentropic_initial_guess(Tt_si, Pt_si, gamma_t, MN_clamped)

            # 2D Newton state: [Ts, Ps, R1, R2, S, hs, gamma, R,
            #                   dS_dT, dS_dP, dh_dT, dh_dP, dgamma_dT, dgamma_dP, dR_dT, dR_dP, iter]
            def compute_state_2d(Ts_si, Ps_si, i):
                props_s, dprops_dP, dprops_dT = props_at_TP_si_with_derivs(Ts_si, Ps_si, FAR)
                S_s, hs_si, gamma_s, R_s = props_s['S'], props_s['h'], props_s['gamma'], props_s['R']

                Vsonic_sq = gamma_s * R_s * Ts_si
                R1 = S_s - S_total
                R2 = hs_si + 0.5 * MN_sq * Vsonic_sq - ht_si

                return jnp.array([
                    Ts_si, Ps_si, R1, R2,
                    S_s, hs_si, gamma_s, R_s,
                    dprops_dT['S'], dprops_dP['S'], dprops_dT['h'], dprops_dP['h'],
                    dprops_dT['gamma'], dprops_dP['gamma'], dprops_dT['R'], dprops_dP['R'],
                    i
                ])

            def cond_fn_2d(state):
                R1, R2, i = state[2], state[3], state[16]
                return (jnp.sqrt(R1**2 + R2**2) > 1e-8) & (i < 20)

            def body_fn_2d(state):
                Ts_si, Ps_si = state[0], state[1]
                R1, R2 = state[2], state[3]
                gamma_s, R_s = state[6], state[7]
                i = state[16]

                dR1_dT, dR1_dP, dR2_dT, dR2_dP = compute_R1_R2_jacobian(
                    Ts_si, MN_sq, gamma_s, R_s,
                    state[8], state[9], state[10], state[11],
                    state[12], state[13], state[14], state[15]
                )

                dTs, dPs = solve_2x2(dR1_dT, dR1_dP, dR2_dT, dR2_dP, R1, R2)

                Ts_new = jnp.clip(Ts_si + dTs, 160.0, 2400.0)
                Ps_new = jnp.clip(Ps_si + dPs, 100.0, 1e8)
                return compute_state_2d(Ts_new, Ps_new, i + 1)

            # Run 2D Newton
            init_state = compute_state_2d(Ts0_si, Ps0_si, 0.0)
            final_state = jax.lax.while_loop(cond_fn_2d, body_fn_2d, init_state)

            # Extract solution
            Ts_newton, Ps_newton = final_state[0], final_state[1]
            S_newton, hs_newton = final_state[4], final_state[5]
            gamma_newton, R_newton = final_state[6], final_state[7]

            # Get Cp, Cv (not in state)
            props_final = props_at_TP_si(Ts_newton, Ps_newton, FAR)
            Cp_newton, Cv_newton = props_final['Cp'], props_final['Cv']

            # Compute derived quantities
            rhos_newton = Ps_newton / (R_newton * Ts_newton)
            Vsonic_newton = jnp.sqrt(gamma_newton * R_newton * Ts_newton)
            V_newton = MN_clamped * Vsonic_newton
            area_newton = W_si / (rhos_newton * V_newton)

            # === Compute darea/dMN via implicit differentiation ===
            Vsonic_sq_newton = gamma_newton * R_newton * Ts_newton
            dR1_dT, dR1_dP, dR2_dT, dR2_dP = compute_R1_R2_jacobian(
                Ts_newton, MN_sq, gamma_newton, R_newton,
                final_state[8], final_state[9], final_state[10], final_state[11],
                final_state[12], final_state[13], final_state[14], final_state[15]
            )

            # Solve for dTs/dMN, dPs/dMN: J * [dTs/dMN; dPs/dMN] = [0; -MN*Vsonic²]
            det = dR1_dT * dR2_dP - dR1_dP * dR2_dT
            det = jnp.where(jnp.abs(det) < 1e-30, 1e-30, det)
            rhs_MN = -MN_clamped * Vsonic_sq_newton
            dTs_dMN = (-rhs_MN * dR1_dP) / det
            dPs_dMN = (dR1_dT * rhs_MN) / det

            # Chain rule for derived quantities
            dgamma_dT, dgamma_dP = final_state[12], final_state[13]
            dR_dT, dR_dP = final_state[14], final_state[15]
            dgamma_dMN = dgamma_dT * dTs_dMN + dgamma_dP * dPs_dMN
            dR_dMN = dR_dT * dTs_dMN + dR_dP * dPs_dMN

            drho_dMN = rhos_newton * (dPs_dMN / Ps_newton - dR_dMN / R_newton - dTs_dMN / Ts_newton)
            dVsonic_dMN = 0.5 * Vsonic_newton * (dgamma_dMN / gamma_newton + dR_dMN / R_newton + dTs_dMN / Ts_newton)
            darea_dMN_newton = area_newton * (-1.0 / MN_clamped - drho_dMN / rhos_newton - dVsonic_dMN / Vsonic_newton)

            # Handle zero MN case
            is_zero_MN = MN < 1e-10
            rhos_zero = Pt_si / (R_t * Tt_si)
            Vsonic_zero = jnp.sqrt(gamma_t * R_t * Tt_si)

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
            darea_dMN_si = jnp.where(is_zero_MN, -1e30, darea_dMN_newton)

            # Convert to English and return
            result = convert_static_to_english(
                Ts_si, Ps_si, hs_si, rhos_si, MN_out, V_si, Vsonic_si,
                area_si, gamma_out, Cp_out, Cv_out, S_out, R_out
            )
            darea_dMN = darea_dMN_si * area_from_si
            return jnp.concatenate([result, jnp.array([darea_dMN])])

        # =====================================================================
        # static_from_area: 3D Newton solver for (Ts, Ps, MN) given area
        # =====================================================================

        def static_from_area_impl(Tt, Pt, area, W, FAR):
            """Pure JAX static_from_area: solve for (Ts, Ps, MN) given area."""
            # Convert inputs to SI
            Tt_si = Tt * T_to_si_scale
            Pt_si = Pt * P_to_si
            W_si = W * W_to_si
            area_si = area * area_to_si

            # Get total properties
            tot_props = props_at_TP_si(Tt_si, Pt_si, FAR)
            ht_si = tot_props['h']
            S_total = tot_props['S']
            gamma_t = tot_props['gamma']

            # Initial guess
            MN0 = 0.5
            Ts0_si, Ps0_si = compute_isentropic_initial_guess(Tt_si, Pt_si, gamma_t, MN0)

            # 3D Newton state: [Ts, Ps, MN, R1, R2, R3, S, hs, gamma, R,
            #                   dS_dT, dS_dP, dh_dT, dh_dP, dgamma_dT, dgamma_dP, dR_dT, dR_dP,
            #                   area_computed, Vsonic_sq, iter]
            def compute_state_3d(Ts_si, Ps_si, MN, i):
                props_s, dprops_dP, dprops_dT = props_at_TP_si_with_derivs(Ts_si, Ps_si, FAR)
                S_s, hs_si, gamma_s, R_s = props_s['S'], props_s['h'], props_s['gamma'], props_s['R']

                MN_clamped = jnp.maximum(MN, 1e-10)
                MN_sq = MN_clamped ** 2
                Vsonic_sq = gamma_s * R_s * Ts_si
                rho_s = Ps_si / (R_s * Ts_si)
                V_s = MN_clamped * jnp.sqrt(Vsonic_sq)
                area_computed = W_si / (rho_s * V_s)

                R1 = S_s - S_total
                R2 = hs_si + 0.5 * MN_sq * Vsonic_sq - ht_si
                R3 = area_computed - area_si

                return jnp.array([
                    Ts_si, Ps_si, MN, R1, R2, R3,
                    S_s, hs_si, gamma_s, R_s,
                    dprops_dT['S'], dprops_dP['S'], dprops_dT['h'], dprops_dP['h'],
                    dprops_dT['gamma'], dprops_dP['gamma'], dprops_dT['R'], dprops_dP['R'],
                    area_computed, Vsonic_sq, i
                ])

            def cond_fn_3d(state):
                R1, R2, R3, i = state[3], state[4], state[5], state[20]
                return (jnp.sqrt(R1**2 + R2**2 + R3**2) > 1e-8) & (i < 30)

            def body_fn_3d(state):
                Ts_si, Ps_si, MN = state[0], state[1], state[2]
                R1, R2, R3 = state[3], state[4], state[5]
                gamma_s, R_s = state[8], state[9]
                area_computed, Vsonic_sq = state[18], state[19]
                i = state[20]

                MN_clamped = jnp.maximum(MN, 1e-10)
                MN_sq = MN_clamped ** 2

                # Jacobian rows 1 & 2 (shared with 2D solver)
                dR1_dT, dR1_dP, dR2_dT, dR2_dP = compute_R1_R2_jacobian(
                    Ts_si, MN_sq, gamma_s, R_s,
                    state[10], state[11], state[12], state[13],
                    state[14], state[15], state[16], state[17]
                )
                dR2_dMN = MN_clamped * Vsonic_sq  # dR2/dMN = MN * γ * R * Ts

                # Jacobian row 3 (area residual)
                dgamma_dT, dgamma_dP = state[14], state[15]
                dR_dT, dR_dP = state[16], state[17]
                dR3_dT = 0.5 * area_computed * (dR_dT / R_s + 1.0 / Ts_si - dgamma_dT / gamma_s)
                dR3_dP = area_computed * (-1.0 / Ps_si + 0.5 * dR_dP / R_s - 0.5 * dgamma_dP / gamma_s)
                dR3_dMN = -area_computed / MN_clamped

                # Solve 3x3 via block elimination (exploiting dR1_dMN = 0)
                J11_safe = jnp.where(jnp.abs(dR1_dT) < 1e-20, 1e-20, dR1_dT)
                J22_mod = dR2_dP - dR2_dT * dR1_dP / J11_safe
                J32_mod = dR3_dP - dR3_dT * dR1_dP / J11_safe
                R2_mod = -R2 + dR2_dT * R1 / J11_safe
                R3_mod = -R3 + dR3_dT * R1 / J11_safe

                det2 = J22_mod * dR3_dMN - dR2_dMN * J32_mod
                det2_safe = jnp.where(jnp.abs(det2) < 1e-20, 1e-20, det2)
                dPs = (R2_mod * dR3_dMN - dR2_dMN * R3_mod) / det2_safe
                dMN = (J22_mod * R3_mod - R2_mod * J32_mod) / det2_safe
                dTs = (-R1 - dR1_dP * dPs) / J11_safe

                Ts_new = jnp.clip(Ts_si + dTs, 160.0, 2400.0)
                Ps_new = jnp.clip(Ps_si + dPs, 100.0, 1e8)
                MN_new = jnp.clip(MN + dMN, 0.01, 0.99)
                return compute_state_3d(Ts_new, Ps_new, MN_new, i + 1)

            # Run 3D Newton
            init_state = compute_state_3d(Ts0_si, Ps0_si, MN0, 0.0)
            final_state = jax.lax.while_loop(cond_fn_3d, body_fn_3d, init_state)

            # Extract solution
            Ts_si, Ps_si, MN_final = final_state[0], final_state[1], final_state[2]
            S_s, hs_si, gamma_s, R_s = final_state[6], final_state[7], final_state[8], final_state[9]

            # Compute derived quantities
            rho_s = Ps_si / (R_s * Ts_si)
            Vsonic = jnp.sqrt(gamma_s * R_s * Ts_si)
            V_s = MN_final * Vsonic
            area_final = W_si / (rho_s * V_s)

            # Get Cp, Cv
            props_final = props_at_TP_si(Ts_si, Ps_si, FAR)
            Cp_s, Cv_s = props_final['Cp'], props_final['Cv']

            # Convert to English and return
            return convert_static_to_english(
                Ts_si, Ps_si, hs_si, rho_s, MN_final, V_s, Vsonic,
                area_final, gamma_s, Cp_s, Cv_s, S_s, R_s
            )

        # JIT compile
        self._static_from_MN_jit = jax.jit(static_from_MN_impl)
        self._static_from_area_jit = jax.jit(static_from_area_impl)
