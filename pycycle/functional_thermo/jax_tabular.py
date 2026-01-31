"""
Pure JAX implementation of tabular thermodynamic interpolation.

This module provides JAX-traceable versions of the tabular thermo operations,
eliminating the need for pure_callback and enabling efficient JIT compilation.
"""

import jax
import jax.numpy as jnp
import numpy as np


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
        self._prop_idx = {name: i for i, name in enumerate(self.property_names)}

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
        FAR, P, T = point[0], point[1], point[2]

        # Find cell indices
        i_FAR = self._find_cell_idx(FAR, self.grid[0])
        i_P = self._find_cell_idx(P, self.grid[1])
        i_T = self._find_cell_idx(T, self.grid[2])

        # Get grid points for this cell
        x0, x1 = self.grid[0][i_FAR], self.grid[0][i_FAR + 1]
        y0, y1 = self.grid[1][i_P], self.grid[1][i_P + 1]
        z0, z1 = self.grid[2][i_T], self.grid[2][i_T + 1]

        # Compute normalized coordinates [0, 1] within cell
        xd = (FAR - x0) / (x1 - x0)
        yd = (P - y0) / (y1 - y0)
        zd = (T - z0) / (z1 - z0)

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

        # Trilinear interpolation
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        values = c0 * (1 - zd) + c1 * zd

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
        FAR, P, T = point[0], point[1], point[2]

        # Find cell indices
        i_FAR = self._find_cell_idx(FAR, self.grid[0])
        i_P = self._find_cell_idx(P, self.grid[1])
        i_T = self._find_cell_idx(T, self.grid[2])

        # Get grid points for this cell
        x0, x1 = self.grid[0][i_FAR], self.grid[0][i_FAR + 1]
        y0, y1 = self.grid[1][i_P], self.grid[1][i_P + 1]
        z0, z1 = self.grid[2][i_T], self.grid[2][i_T + 1]

        # Compute normalized coordinates
        xd = (FAR - x0) / (x1 - x0)
        yd = (P - y0) / (y1 - y0)
        zd = (T - z0) / (z1 - z0)

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

        # Trilinear interpolation
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return c0 * (1 - zd) + c1 * zd


def jax_newton_solve(residual_fn, x0, max_iter=20, tol=1e-10,
                     lower=160.0, upper=2400.0):
    """
    Pure JAX Newton solver using lax.while_loop.

    Parameters
    ----------
    residual_fn : callable
        Function that returns (residual, jacobian) given x
    x0 : float
        Initial guess
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    lower, upper : float
        Bounds for the solution

    Returns
    -------
    x : float
        Solution
    """
    def cond_fn(state):
        x, residual, i = state
        return (jnp.abs(residual) > tol) & (i < max_iter)

    def body_fn(state):
        x, _, i = state
        residual, jacobian = residual_fn(x)
        # Newton step with bounds clamping
        dx = -residual / jacobian
        x_new = jnp.clip(x + dx, lower, upper)
        residual_new, _ = residual_fn(x_new)
        return (x_new, residual_new, i + 1)

    # Initial state
    residual0, _ = residual_fn(x0)
    init_state = (x0, residual0, 0)

    # Run Newton iteration
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return final_state[0]


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

    # Property indices in the stacked values array
    PROP_H = 0
    PROP_S = 1
    PROP_GAMMA = 2
    PROP_CP = 3
    PROP_CV = 4
    PROP_RHO = 5
    PROP_R = 6

    def __init__(self, spec):
        """Initialize with tabular data specification."""
        # Store grid points
        self.FAR_grid = jnp.array(spec['FAR'])
        self.P_grid = jnp.array(spec['P'])
        self.T_grid = jnp.array(spec['T'])

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
        self._h_to_si = 2326.0  # Btu/lbm -> J/kg
        self._h_from_si = 1.0 / 2326.0
        self._P_to_si = 6894.76  # psi -> Pa
        self._T_to_si_offset = 0.0  # Rankine offset
        self._T_to_si_scale = 5.0 / 9.0  # Rankine -> Kelvin scale
        self._S_from_si = 1.0 / 4186.8  # J/(kg*K) -> Btu/(lbm*R)
        self._rho_from_si = 0.062428  # kg/m^3 -> lbm/ft^3

        # Create JIT-compiled versions of the methods
        self._setup_jit_functions()
        self._setup_static_functions()

    def _setup_jit_functions(self):
        """Create JIT-compiled versions of thermo functions."""
        interp = self._interp
        h_to_si = self._h_to_si
        h_from_si = self._h_from_si
        P_to_si = self._P_to_si
        T_to_si_scale = self._T_to_si_scale
        S_from_si = self._S_from_si
        rho_from_si = self._rho_from_si

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

            def h_interp(T_si):
                point = jnp.array([FAR, P_si, T_si])
                return interp.interpolate_single(point, 0)  # PROP_H = 0

            def cond_fn(state):
                T_si, residual, i = state
                return (jnp.abs(residual) > 1e-8) & (i < 20)

            def body_fn(state):
                T_si, _, i = state
                h_si = h_interp(T_si)
                residual = h_si - h_target_si

                # dh/dT via finite difference
                eps = 1.0
                h_plus = h_interp(T_si + eps)
                dh_dT = (h_plus - h_si) / eps
                dh_dT = jnp.where(jnp.abs(dh_dT) < 1e-20, 1e-20, dh_dT)

                # Newton step with bounds
                dx = -residual / dh_dT
                T_si_new = jnp.clip(T_si + dx, 160.0, 2400.0)

                # Compute new residual for convergence check
                residual_new = h_interp(T_si_new) - h_target_si
                return (T_si_new, residual_new, i + 1)

            # Initialize state: (T_si, residual, iteration)
            residual_init = h_interp(T_si_init) - h_target_si
            init_state = (T_si_init, residual_init, 0)

            # Run Newton iteration
            final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
            T_si = final_state[0]

            return T_si / T_to_si_scale

        self._props_TP_jit = _props_TP_jit
        self._T_from_hP_jit = _T_from_hP_jit

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
        return self._static_from_area_jit(Tt, Pt, area, W, FAR)

    def _setup_static_functions(self):
        """Create JIT-compiled versions of static property functions."""
        interp = self._interp
        h_to_si = self._h_to_si
        h_from_si = self._h_from_si
        P_to_si = self._P_to_si
        T_to_si_scale = self._T_to_si_scale
        S_from_si = self._S_from_si
        rho_from_si = self._rho_from_si

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

            # Handle MN ~ 0 case
            def zero_MN_case():
                # Static = Total
                Ts_si = Tt_si
                Ps_si = Pt_si
                hs_si = ht_si
                props_s = props_at_TP_si(Ts_si, Ps_si, FAR)
                R_s = props_s['R']
                gamma_s = props_s['gamma']
                Cp_s = props_s['Cp']
                Cv_s = props_s['Cv']
                rhos_si = Ps_si / (R_s * Ts_si)
                Vsonic_si = jnp.sqrt(gamma_s * R_s * Ts_si)
                return jnp.array([Ts_si, Ps_si, hs_si, rhos_si, 0.0, 0.0, Vsonic_si,
                                  jnp.inf, gamma_s, Cp_s, Cv_s, S_total, R_s])

            def nonzero_MN_case():
                MN_sq = MN ** 2

                # Initial guess using ideal gas isentropic relations
                Ps0_si = Pt_si * (1.0 + (gamma_t - 1.0) / 2.0 * MN_sq) ** (-gamma_t / (gamma_t - 1.0))
                Ts0_si = Tt_si * (Ps0_si / Pt_si) ** ((gamma_t - 1.0) / gamma_t)

                # 2D Newton solver for (Ts, Ps) using while_loop
                # Residuals:
                #   R1 = S(Ts, Ps) - S_total  (entropy conservation)
                #   R2 = hs + MN²·γ·R·Ts/2 - ht  (energy conservation)

                def compute_residuals(Ts_si, Ps_si):
                    """Compute residuals for the 2D Newton system."""
                    props_s = props_at_TP_si(Ts_si, Ps_si, FAR)
                    S_s = props_s['S']
                    hs_si = props_s['h']
                    gamma_s = props_s['gamma']
                    R_s = props_s['R']

                    # Vsonic = sqrt(gamma * R * T)
                    Vsonic_sq = gamma_s * R_s * Ts_si
                    V_sq = MN_sq * Vsonic_sq

                    # Residuals
                    R1 = S_s - S_total
                    R2 = hs_si + 0.5 * V_sq - ht_si
                    return R1, R2

                def cond_fn_2d(state):
                    # state = [Ts_si, Ps_si, R1, R2, i]
                    R1, R2, i = state[2], state[3], state[4]
                    residual_norm = jnp.sqrt(R1**2 + R2**2)
                    return (residual_norm > 1e-8) & (i < 20)

                def body_fn_2d(state):
                    Ts_si, Ps_si = state[0], state[1]
                    i = state[4]

                    props_s = props_at_TP_si(Ts_si, Ps_si, FAR)
                    S_s = props_s['S']
                    hs_si = props_s['h']
                    gamma_s = props_s['gamma']
                    R_s = props_s['R']

                    # Vsonic = sqrt(gamma * R * T)
                    Vsonic_sq = gamma_s * R_s * Ts_si
                    V_sq = MN_sq * Vsonic_sq

                    # Residuals
                    R1 = S_s - S_total
                    R2 = hs_si + 0.5 * V_sq - ht_si

                    # Approximate Jacobian via finite differences
                    eps_T = 1.0  # 1K step
                    eps_P = 100.0  # 100Pa step

                    props_T = props_at_TP_si(Ts_si + eps_T, Ps_si, FAR)
                    props_P = props_at_TP_si(Ts_si, Ps_si + eps_P, FAR)

                    # dR1/dTs, dR1/dPs (entropy derivatives)
                    dR1_dT = (props_T['S'] - S_s) / eps_T
                    dR1_dP = (props_P['S'] - S_s) / eps_P

                    # dR2/dTs, dR2/dPs (energy derivatives)
                    dhs_dT = (props_T['h'] - hs_si) / eps_T
                    dhs_dP = (props_P['h'] - hs_si) / eps_P
                    dgamma_dT = (props_T['gamma'] - gamma_s) / eps_T
                    dgamma_dP = (props_P['gamma'] - gamma_s) / eps_P
                    dR_dT = (props_T['R'] - R_s) / eps_T
                    dR_dP = (props_P['R'] - R_s) / eps_P

                    dVsq_dT = MN_sq * (dgamma_dT * R_s * Ts_si + gamma_s * dR_dT * Ts_si + gamma_s * R_s)
                    dVsq_dP = MN_sq * (dgamma_dP * R_s * Ts_si + gamma_s * dR_dP * Ts_si)

                    dR2_dT = dhs_dT + 0.5 * dVsq_dT
                    dR2_dP = dhs_dP + 0.5 * dVsq_dP

                    # Solve 2x2 system: J * dx = -R
                    det = dR1_dT * dR2_dP - dR1_dP * dR2_dT
                    det = jnp.where(jnp.abs(det) < 1e-20, 1e-20, det)

                    dTs = (-R1 * dR2_dP + R2 * dR1_dP) / det
                    dPs = (-R2 * dR1_dT + R1 * dR2_dT) / det

                    # Clamp updates
                    Ts_new = jnp.clip(Ts_si + dTs, 160.0, 2400.0)
                    Ps_new = jnp.clip(Ps_si + dPs, 100.0, 1e8)

                    # Compute new residuals
                    R1_new, R2_new = compute_residuals(Ts_new, Ps_new)

                    return jnp.array([Ts_new, Ps_new, R1_new, R2_new, i + 1])

                # Initialize state: [Ts_si, Ps_si, R1, R2, iteration]
                R1_init, R2_init = compute_residuals(Ts0_si, Ps0_si)
                init_state = jnp.array([Ts0_si, Ps0_si, R1_init, R2_init, 0.0])

                # Run 2D Newton iteration
                final_state = jax.lax.while_loop(cond_fn_2d, body_fn_2d, init_state)
                Ts_si, Ps_si = final_state[0], final_state[1]

                # Get final properties
                props_final = props_at_TP_si(Ts_si, Ps_si, FAR)
                hs_si = props_final['h']
                gamma_s = props_final['gamma']
                R_s = props_final['R']
                Cp_s = props_final['Cp']
                Cv_s = props_final['Cv']
                S_s = props_final['S']

                # Compute derived quantities
                rhos_si = Ps_si / (R_s * Ts_si)
                Vsonic_si = jnp.sqrt(gamma_s * R_s * Ts_si)
                V_si = MN * Vsonic_si
                area_si = W_si / (rhos_si * V_si)

                return jnp.array([Ts_si, Ps_si, hs_si, rhos_si, MN, V_si, Vsonic_si,
                                  area_si, gamma_s, Cp_s, Cv_s, S_s, R_s])

            # Use lax.cond to handle both cases
            result_si = jax.lax.cond(MN < 1e-10, zero_MN_case, nonzero_MN_case)

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

            return jnp.array([Ts, Ps, hs, rhos, MN_out, V, Vsonic, area,
                              gamma_out, Cp, Cv, S, R])

        def static_from_area_impl(Tt, Pt, area, W, FAR):
            """Pure JAX static_from_area implementation using while_loop Newton."""
            # Convert to SI
            Tt_si = Tt * T_to_si_scale
            Pt_si = Pt * P_to_si
            W_si = W * W_to_si
            area_si = area * area_to_si

            # Get total properties for initial guess
            tot_props = props_at_TP_si(Tt_si, Pt_si, FAR)
            gamma_t = tot_props['gamma']

            # Initial MN guess using ideal gas relations
            # Start with subsonic guess
            MN0 = 0.5

            # Newton iteration on MN to match area using while_loop
            def area_residual(MN_val):
                # Compute static props at this MN
                static_result = static_from_MN_impl(Tt, Pt, MN_val, W, FAR)
                area_computed = static_result[7]  # area is at index 7
                return area_computed - area

            def cond_fn_MN(state):
                # state = [MN, residual, i]
                MN_val, residual, i = state[0], state[1], state[2]
                return (jnp.abs(residual) > 1e-8) & (i < 20)

            def body_fn_MN(state):
                MN_val, _, i = state[0], state[1], state[2]

                res = area_residual(MN_val)
                # Finite difference for derivative
                eps = 1e-6
                res_plus = area_residual(MN_val + eps)
                dres_dMN = (res_plus - res) / eps
                dres_dMN = jnp.where(jnp.abs(dres_dMN) < 1e-20, 1e-20, dres_dMN)
                dMN = -res / dres_dMN
                MN_new = jnp.clip(MN_val + dMN, 0.01, 0.99)  # Subsonic only

                # Compute new residual
                res_new = area_residual(MN_new)
                return jnp.array([MN_new, res_new, i + 1])

            # Initialize state: [MN, residual, iteration]
            res_init = area_residual(MN0)
            init_state = jnp.array([MN0, res_init, 0.0])

            # Run Newton iteration
            final_state = jax.lax.while_loop(cond_fn_MN, body_fn_MN, init_state)
            MN = final_state[0]

            # Final computation with solved MN
            return static_from_MN_impl(Tt, Pt, MN, W, FAR)

        # JIT compile
        self._static_from_MN_jit = jax.jit(static_from_MN_impl)
        self._static_from_area_jit = jax.jit(static_from_area_impl)


# Factory function to create JIT-compiled thermo functions
def create_jax_thermo_functions(spec):
    """
    Create JIT-compiled pure JAX thermo functions from a table specification.

    Parameters
    ----------
    spec : dict
        Tabular data specification

    Returns
    -------
    dict
        Dictionary of JIT-compiled functions:
        - 'T_from_hP': (h, P, FAR) -> T
        - 'props_TP': (T, P, FAR) -> [h, S, gamma, Cp, Cv, rho, R]
    """
    thermo = JaxTabularThermo(spec)

    # Create JIT-compiled versions
    T_from_hP_jit = jax.jit(thermo.T_from_hP)
    props_TP_jit = jax.jit(thermo.props_TP)

    return {
        'T_from_hP': T_from_hP_jit,
        'props_TP': props_TP_jit,
        '_thermo': thermo,  # Keep reference for debugging
    }
