"""
Pure JAX implementation of tabular thermodynamic interpolation.

This module provides JAX-traceable versions of the tabular thermo operations,
eliminating the need for pure_callback and enabling efficient JIT compilation.
"""

import jax
import jax.numpy as jnp


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

            def h_interp(T_si):
                point = jnp.array([FAR, P_si, T_si])
                return interp.interpolate_single(point, 0)  # h is at index 0

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

            # Zero MN case: static = total (cheap to compute)
            rhos_zero = Pt_si / (R_t * Tt_si)
            Vsonic_zero = jnp.sqrt(gamma_t * R_t * Tt_si)

            # Use jnp.where to blend results (avoids tracing both lax.cond branches)
            is_zero_MN = MN < 1e-10
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

                # Use forward-mode AD (jvp) for derivative - works through while_loop
                # jax.grad uses reverse-mode which doesn't work with while_loop
                res, dres_dMN = jax.jvp(area_residual, (MN_val,), (1.0,))
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
