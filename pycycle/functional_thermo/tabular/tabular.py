"""
Tabular thermodynamic property calculations using interpolation.

This module provides utility classes for tabular interpolation:
- NewtonSolver: Generic Newton solver for N-D root-finding problems
- MultiOutputTrilinearInterp: Efficient trilinear interpolator for multiple outputs
- TOTAL_PROPS, STATIC_PROPS: Property name constants
"""

import numpy as np


# =============================================================================
# Generic Newton Solver
# =============================================================================

class NewtonSolver:
    """
    Generic Newton solver for N-D root-finding problems.

    Supports analytical Jacobians, bounds clamping, Armijo-Goldstein line search,
    and multiple convergence criteria. Works for any dimension (1D, 2D, N-D) with
    a unified implementation.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of Newton iterations. Default is 20.
    tol : float, optional
        Convergence tolerance. Default is 1e-10.
    convergence_mode : str, optional
        How to check convergence:
        - 'relative': |R| < tol * |ref| or |R| < abs_tol (default)
        - 'absolute': |R| < tol
        - 'component': max(|R_i|) < tol
    abs_tol : float, optional
        Absolute tolerance floor for 'relative' mode. Default is 1e-6.
    stagnation_tol : float, optional
        Minimum step size before declaring stagnation. Default is 1e-12.
    bounds : tuple, optional
        (lower, upper) bounds for the solution. Can be scalars or arrays.
    linesearch : bool, optional
        Enable Armijo-Goldstein line search. Default is False.
    ls_c : float, optional
        Slope parameter for sufficient decrease condition. Controls how much
        decrease is required. Larger values require more decrease. Default is 0.1.
    ls_rho : float, optional
        Contraction factor for backtracking. Each failed iteration multiplies
        the step size by this factor. Default is 0.5.
    ls_maxiter : int, optional
        Maximum line search iterations before accepting the step. Default is 5.

    Examples
    --------
    1D problem: find T such that h(T) = h_target

    >>> solver = NewtonSolver(max_iter=20, tol=1e-10, bounds=(200.0, 3000.0))
    >>> def residual_and_jac(T):
    ...     h, dh_dT = compute_h_and_derivative(T)
    ...     return h - h_target, dh_dT
    >>> T, converged, n_iter = solver.solve(residual_and_jac, T_guess)

    2D problem: find (Ts, Ps) such that [entropy_error, energy_error] = 0

    >>> solver = NewtonSolver(convergence_mode='component')
    >>> def residual_and_jac(x):
    ...     Ts, Ps = x
    ...     R = np.array([entropy_error(Ts, Ps), energy_error(Ts, Ps)])
    ...     J = np.array([[dR1_dTs, dR1_dPs], [dR2_dTs, dR2_dPs]])
    ...     return R, J
    >>> x, converged, n_iter = solver.solve(residual_and_jac, [Ts_guess, Ps_guess])

    With line search for improved robustness:

    >>> solver = NewtonSolver(linesearch=True, ls_c=0.1, ls_rho=0.5, ls_maxiter=5)
    >>> x, converged, n_iter = solver.solve(residual_and_jac, x0)
    """

    def __init__(self, max_iter=20, tol=1e-10, convergence_mode='relative',
                 abs_tol=1e-6, stagnation_tol=1e-12, bounds=None,
                 linesearch=False, ls_c=0.1, ls_rho=0.5, ls_maxiter=5):
        self.max_iter = max_iter
        self.tol = tol
        self.convergence_mode = convergence_mode
        self.abs_tol = abs_tol
        self.stagnation_tol = stagnation_tol
        self.bounds = bounds
        # Line search parameters
        self.linesearch = linesearch
        self.ls_c = ls_c
        self.ls_rho = ls_rho
        self.ls_maxiter = ls_maxiter

    def solve(self, residual_and_jac_fn, x0, ref_value=None):
        """
        Solve R(x) = 0 using Newton's method.

        Parameters
        ----------
        residual_and_jac_fn : callable
            Function that takes x and returns (residual, jacobian).
            - For 1D: x is scalar, residual is scalar, jacobian is scalar
            - For N-D: x is (N,) array, residual is (N,) array, jacobian is (N,N) array
        x0 : float or array-like
            Initial guess. Determines the problem dimension.
        ref_value : float or array-like, optional
            Reference value for relative convergence checking.
            If None, uses abs_tol as the floor.

        Returns
        -------
        x : float or ndarray
            Solution (same type/shape as x0).
        converged : bool
            True if the solver converged within tolerance.
        n_iter : int
            Number of iterations performed.
        """
        # Normalize to arrays for unified handling
        is_scalar = np.isscalar(x0)
        x = np.atleast_1d(np.asarray(x0, dtype=float))
        n = x.size

        # Set up bounds
        if self.bounds is not None:
            lower = np.atleast_1d(np.asarray(self.bounds[0], dtype=float))
            upper = np.atleast_1d(np.asarray(self.bounds[1], dtype=float))
            # Broadcast scalar bounds to array size
            if lower.size == 1:
                lower = np.full(n, lower[0])
            if upper.size == 1:
                upper = np.full(n, upper[0])
        else:
            lower = np.full(n, -np.inf)
            upper = np.full(n, np.inf)

        # Normalize ref_value
        if ref_value is not None:
            ref_value = np.atleast_1d(np.asarray(ref_value, dtype=float))

        # Newton iteration
        for n_iter in range(1, self.max_iter + 1):
            # Get residual and Jacobian
            if is_scalar:
                residual, jacobian = residual_and_jac_fn(x[0])
                residual = np.atleast_1d(residual)
                jacobian = np.atleast_2d(jacobian)
            else:
                residual, jacobian = residual_and_jac_fn(x)

            # Check convergence
            if self._check_convergence(residual, ref_value):
                return (x[0] if is_scalar else x), True, n_iter

            # Current residual norm (for line search)
            phi0 = np.linalg.norm(residual)

            # Solve linear system: J @ dx = -R
            try:
                dx = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                return (x[0] if is_scalar else x), False, n_iter

            # Line search (if enabled)
            if self.linesearch and phi0 > 0:
                alpha = self._armijo_linesearch(residual_and_jac_fn, x, dx, phi0,
                                                is_scalar, lower, upper)
            else:
                alpha = 1.0

            # Newton update with step size
            x_new = x + alpha * dx

            # Apply bounds
            x_new = np.maximum(lower, np.minimum(upper, x_new))

            # Check for stagnation
            if np.max(np.abs(x_new - x)) < self.stagnation_tol:
                return (x_new[0] if is_scalar else x_new), True, n_iter

            x = x_new

        return (x[0] if is_scalar else x), False, self.max_iter

    def _armijo_linesearch(self, residual_and_jac_fn, x, dx, phi0, is_scalar, lower, upper):
        """
        Armijo-Goldstein backtracking line search.

        Finds a step size alpha such that the sufficient decrease condition is satisfied:
            phi(alpha) <= phi(0) + c * alpha * dphi/dalpha

        For Newton's method, the directional derivative dphi/dalpha at alpha=0 equals -phi(0),
        since a full Newton step would drive the linearized residuals to zero.
        This simplifies the condition to:
            phi(alpha) <= phi(0) * (1 - c * alpha)

        Parameters
        ----------
        residual_and_jac_fn : callable
            Function that returns (residual, jacobian).
        x : ndarray
            Current solution estimate.
        dx : ndarray
            Newton step direction.
        phi0 : float
            Current residual norm ||R(x)||.
        is_scalar : bool
            Whether the original problem was scalar.
        lower : ndarray
            Lower bounds.
        upper : ndarray
            Upper bounds.

        Returns
        -------
        alpha : float
            Accepted step size (between 0 and 1).
        """
        c = self.ls_c
        rho = self.ls_rho
        alpha = 1.0

        # Directional derivative for Newton's method: dphi/dalpha = -phi0
        # Armijo condition: phi(alpha) <= phi0 + c * alpha * (-phi0)
        #                   phi(alpha) <= phi0 * (1 - c * alpha)

        for _ in range(self.ls_maxiter):
            # Trial point
            x_trial = x + alpha * dx
            x_trial = np.maximum(lower, np.minimum(upper, x_trial))

            # Evaluate residual at trial point
            if is_scalar:
                residual_trial, _ = residual_and_jac_fn(x_trial[0])
                residual_trial = np.atleast_1d(residual_trial)
            else:
                residual_trial, _ = residual_and_jac_fn(x_trial)

            phi = np.linalg.norm(residual_trial)

            # Check Armijo condition
            if phi <= phi0 * (1.0 - c * alpha):
                return alpha

            # Backtrack
            alpha *= rho

        # Return whatever alpha we ended up with
        return alpha

    def _check_convergence(self, residual, ref_value):
        """Check convergence based on configured mode."""
        if self.convergence_mode == 'absolute':
            return np.linalg.norm(residual) < self.tol
        elif self.convergence_mode == 'relative':
            if ref_value is not None:
                ref = np.linalg.norm(ref_value)
            else:
                ref = 1.0  # Fall back to abs_tol check
            return np.linalg.norm(residual) < self.tol * ref or np.linalg.norm(residual) < self.abs_tol
        else:  # component
            return np.max(np.abs(residual)) < self.tol


# Property name constants to avoid repetition
TOTAL_PROPS = ('h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R')
STATIC_PROPS = ('Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area', 'gamma', 'Cp', 'Cv', 'S', 'R')


class MultiOutputTrilinearInterp:
    """
    Trilinear interpolator optimized for multiple outputs sharing the same grid.

    Separates cell search from interpolation to allow bulk evaluation of
    multiple properties at the same (FAR, P, T) point. This provides significant
    speedup over calling separate InterpND instances for each property.

    Parameters
    ----------
    grid : tuple of ndarray
        (FAR_grid, P_grid, T_grid) - same for all properties
    values_dict : dict of ndarray
        {'h': h_table, 'S': S_table, ...} - one 3D array per property
    """

    def __init__(self, grid, values_dict):
        self.grid = grid
        self.values = values_dict
        self.property_names = list(values_dict.keys())
        self._n_props = len(self.property_names)
        self._prop_idx = {name: i for i, name in enumerate(self.property_names)}

        # Pre-stack all value tables for efficient bulk access
        # Shape: (n_props, nFAR, nP, nT)
        self._stacked_values = np.stack(
            [values_dict[name] for name in self.property_names], axis=0
        )

        # Cache for bracketing results (last cell indices)
        self._last_idx = [0, 0, 0]

        # Cache for coefficient reuse when point hasn't changed
        self._cache_key = None
        self._cache_coeffs = None
        self._cache_grid_terms = None

    def bracket(self, x):
        """
        Find cell indices for the given point.

        Parameters
        ----------
        x : ndarray or tuple
            Point (FAR, P, T) to locate

        Returns
        -------
        idx : tuple of int
            Cell indices (i_FAR, i_P, i_T)
        """
        # Cell search using searchsorted
        i_FAR = np.searchsorted(self.grid[0], x[0], side='left') - 1
        i_P = np.searchsorted(self.grid[1], x[1], side='left') - 1
        i_T = np.searchsorted(self.grid[2], x[2], side='left') - 1

        # Clamp for extrapolation
        nFAR, nP, nT = len(self.grid[0]), len(self.grid[1]), len(self.grid[2])
        i_FAR = max(0, min(i_FAR, nFAR - 2))
        i_P = max(0, min(i_P, nP - 2))
        i_T = max(0, min(i_T, nT - 2))

        self._last_idx = [i_FAR, i_P, i_T]
        return (i_FAR, i_P, i_T)

    def _compute_grid_terms(self, idx):
        """Pre-compute grid spacing terms for coefficient calculation."""
        i_FAR, i_P, i_T = idx

        x0 = self.grid[0][i_FAR]
        x1 = self.grid[0][i_FAR + 1]
        y0 = self.grid[1][i_P]
        y1 = self.grid[1][i_P + 1]
        z0 = self.grid[2][i_T]
        z1 = self.grid[2][i_T + 1]

        rec_vol = 1.0 / ((x0 - x1) * (y0 - y1) * (z0 - z1))

        return {
            'x0': x0, 'x1': x1,
            'y0': y0, 'y1': y1,
            'z0': z0, 'z1': z1,
            'rec_vol': rec_vol,
        }

    def _compute_coeffs_all(self, idx, grid_terms, dtype=np.float64):
        """
        Compute interpolation coefficients for ALL properties at once.

        Returns
        -------
        ndarray
            Shape (8, n_props) - 8 coefficients for each property
        """
        i_FAR, i_P, i_T = idx
        x0, x1 = grid_terms['x0'], grid_terms['x1']
        y0, y1 = grid_terms['y0'], grid_terms['y1']
        z0, z1 = grid_terms['z0'], grid_terms['z1']
        rec_vol = grid_terms['rec_vol']

        # Extract ALL 8 corners for ALL properties at once
        # Each corner has shape (n_props,)
        c000 = self._stacked_values[:, i_FAR, i_P, i_T]
        c100 = self._stacked_values[:, i_FAR + 1, i_P, i_T]
        c010 = self._stacked_values[:, i_FAR, i_P + 1, i_T]
        c001 = self._stacked_values[:, i_FAR, i_P, i_T + 1]
        c110 = self._stacked_values[:, i_FAR + 1, i_P + 1, i_T]
        c011 = self._stacked_values[:, i_FAR, i_P + 1, i_T + 1]
        c101 = self._stacked_values[:, i_FAR + 1, i_P, i_T + 1]
        c111 = self._stacked_values[:, i_FAR + 1, i_P + 1, i_T + 1]

        # Compute coefficients for ALL properties at once
        # a has shape (8, n_props)
        a = np.empty((8, self._n_props), dtype=dtype)

        a[0] = (-c000 * x1 * y1 * z1 + c001 * x1 * y1 * z0 +
                c010 * x1 * y0 * z1 - c011 * x1 * y0 * z0 +
                c100 * x0 * y1 * z1 - c101 * x0 * y1 * z0 -
                c110 * x0 * y0 * z1 + c111 * x0 * y0 * z0) * rec_vol

        a[1] = (c000 * y1 * z1 - c001 * y1 * z0 - c010 * y0 * z1 +
                c011 * y0 * z0 - c100 * y1 * z1 + c101 * y1 * z0 +
                c110 * y0 * z1 - c111 * y0 * z0) * rec_vol

        a[2] = (c000 * x1 * z1 - c001 * x1 * z0 - c010 * x1 * z1 +
                c011 * x1 * z0 - c100 * x0 * z1 + c101 * x0 * z0 +
                c110 * x0 * z1 - c111 * x0 * z0) * rec_vol

        a[3] = (c000 * x1 * y1 - c001 * x1 * y1 - c010 * x1 * y0 +
                c011 * x1 * y0 - c100 * x0 * y1 + c101 * x0 * y1 +
                c110 * x0 * y0 - c111 * x0 * y0) * rec_vol

        a[4] = (-c000 * z1 + c001 * z0 + c010 * z1 - c011 * z0 +
                c100 * z1 - c101 * z0 - c110 * z1 + c111 * z0) * rec_vol

        a[5] = (-c000 * y1 + c001 * y1 + c010 * y0 - c011 * y0 +
                c100 * y1 - c101 * y1 - c110 * y0 + c111 * y0) * rec_vol

        a[6] = (-c000 * x1 + c001 * x1 + c010 * x1 - c011 * x1 +
                c100 * x0 - c101 * x0 - c110 * x0 + c111 * x0) * rec_vol

        a[7] = (c000 - c001 - c010 + c011 - c100 + c101 + c110 - c111) * rec_vol

        return a

    def interpolate_all(self, x, compute_derivative=True):
        """
        Interpolate ALL properties at once.

        Parameters
        ----------
        x : ndarray or tuple
            Point (FAR, P, T)
        compute_derivative : bool
            If True, also compute gradients

        Returns
        -------
        values : dict
            {property_name: interpolated_value}
        gradients : dict or None
            {property_name: ndarray([dv/dFAR, dv/dP, dv/dT])} if compute_derivative else None
        """
        # Create cache key from the point
        cache_key = (x[0], x[1], x[2])

        # Check if we can reuse cached coefficients
        if self._cache_key == cache_key and self._cache_coeffs is not None:
            a = self._cache_coeffs
            grid_terms = self._cache_grid_terms
            idx = self._last_idx
        else:
            # Do cell search and compute coefficients
            idx = self.bracket(x)
            grid_terms = self._compute_grid_terms(idx)
            a = self._compute_coeffs_all(idx, grid_terms)

            # Cache for reuse
            self._cache_key = cache_key
            self._cache_coeffs = a
            self._cache_grid_terms = grid_terms

        xv, yv, zv = x  # actual coordinates (FAR, P, T)

        # Evaluate polynomial for ALL properties at once
        # val has shape (n_props,)
        val = (a[0] + (a[1] + (a[4] + a[7] * zv) * yv) * xv +
               a[2] * yv + (a[3] + a[5] * xv + a[6] * yv) * zv)

        # Package values as dict
        values = {name: val[i] for i, name in enumerate(self.property_names)}

        if compute_derivative:
            # Compute gradients for ALL properties at once
            # Each gradient component has shape (n_props,)
            d_x = a[1] + yv * a[4] + zv * (a[5] + yv * a[7])  # d/dFAR
            d_y = a[2] + xv * a[4] + zv * (a[6] + xv * a[7])  # d/dP
            d_z = a[3] + xv * a[5] + yv * (a[6] + xv * a[7])  # d/dT

            gradients = {name: np.array([d_x[i], d_y[i], d_z[i]])
                         for i, name in enumerate(self.property_names)}
            return values, gradients
        else:
            return values, None

    def interpolate_subset(self, x, properties, compute_derivative=True):
        """
        Interpolate only specified properties.

        This is optimized to compute all coefficients once (since they share
        the same cell), then extract only the requested properties.

        Parameters
        ----------
        x : ndarray or tuple
            Point (FAR, P, T)
        properties : list of str
            Which properties to interpolate
        compute_derivative : bool
            If True, also compute gradients

        Returns
        -------
        values : dict
        gradients : dict or None
        """
        # Create cache key from the point
        cache_key = (x[0], x[1], x[2])

        # Check if we can reuse cached coefficients
        if self._cache_key == cache_key and self._cache_coeffs is not None:
            a = self._cache_coeffs
        else:
            # Do cell search and compute coefficients
            idx = self.bracket(x)
            grid_terms = self._compute_grid_terms(idx)
            a = self._compute_coeffs_all(idx, grid_terms)

            # Cache for reuse
            self._cache_key = cache_key
            self._cache_coeffs = a
            self._cache_grid_terms = grid_terms

        xv, yv, zv = x  # actual coordinates (FAR, P, T)

        # Get property indices
        prop_indices = [self._prop_idx[p] for p in properties]

        # Evaluate polynomial for selected properties
        values = {}
        for prop, i in zip(properties, prop_indices):
            ai = a[:, i]  # coefficients for this property
            values[prop] = (ai[0] + (ai[1] + (ai[4] + ai[7] * zv) * yv) * xv +
                           ai[2] * yv + (ai[3] + ai[5] * xv + ai[6] * yv) * zv)

        if compute_derivative:
            gradients = {}
            for prop, i in zip(properties, prop_indices):
                ai = a[:, i]
                d_x = ai[1] + yv * ai[4] + zv * (ai[5] + yv * ai[7])  # d/dFAR
                d_y = ai[2] + xv * ai[4] + zv * (ai[6] + xv * ai[7])  # d/dP
                d_z = ai[3] + xv * ai[5] + yv * (ai[6] + xv * ai[7])  # d/dT
                gradients[prop] = np.array([d_x, d_y, d_z])
            return values, gradients
        else:
            return values, None

    def invalidate_cache(self):
        """Clear the coefficient cache (call when point will change)."""
        self._cache_key = None
        self._cache_coeffs = None
        self._cache_grid_terms = None

