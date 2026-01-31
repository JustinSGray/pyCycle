"""
Tabular thermodynamic property calculations using interpolation.
"""

import time
from contextlib import contextmanager

import numpy as np
from scipy.optimize import brentq

from ..base import ThermoInterface, TotalProps, StaticProps


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

# Unit conversion factor mapping for jvp/vjp
_PROP_UNIT_FACTORS = {
    'gamma': 'dimensionless',
    'h': 'h',
    'S': 'S', 'Cp': 'S', 'Cv': 'S', 'R': 'S',
    'rho': 'rho',
}


@contextmanager
def _profile_section(profile_dict, key):
    """Context manager for timing code sections."""
    t0 = time.perf_counter()
    yield
    profile_dict[key] += time.perf_counter() - t0


class TabularThermo(ThermoInterface):
    """
    Thermodynamic property calculations using tabular interpolation.

    This implementation uses pre-computed lookup tables for air/fuel mixtures.
    Properties are interpolated using OpenMDAO's InterpND for structured grids.

    Parameters
    ----------
    FAR : float, optional
        Fuel-to-air ratio. Default is 0.0 (pure air).
    spec : dict, optional
        Tabular data specification containing grid points and property values.
        If None, uses the default AIR_JETA_TAB_SPEC.
    input_units : str, optional
        Unit system for inputs/outputs:
        - 'SI': Use SI units (default)
        - 'English': Use pyCycle English units
    """

    # Class-level registry of all instances for stats collection
    _instances = []

    # Class-level constant arrays (never change, shared across instances)
    _FAR_DERIVS = np.array([0.0, 0.0, 0.0, 0.0, 1.0])   # d/d[Tt, Pt, MN, W, FAR] for FAR
    _MN_DERIVS = np.array([0.0, 0.0, 1.0, 0.0, 0.0])    # d/d[Tt, Pt, MN, W, FAR] for MN

    def __init__(self, FAR=0.0, spec=None, input_units='SI', use_bulk_interp=True):
        # Register this instance
        TabularThermo._instances.append(self)
        from openmdao.components.interp_util.interp import InterpND
        from pycycle.constants import AIR_JETA_TAB_SPEC

        super().__init__(composition={'FAR': FAR}, input_units=input_units)

        # Use default spec if not provided
        if spec is None:
            spec = AIR_JETA_TAB_SPEC
        self.spec = spec

        self.FAR = FAR
        self._use_bulk_interp = use_bulk_interp

        # Create interpolators for each property
        # Grid points: (FAR, P, T) - tables are in SI units
        points = (spec['FAR'], spec['P'], spec['T'])

        # HYBRID APPROACH:
        # - Bulk interpolator for props_TP and linearize (7 props at same point)
        # - Separate InterpND for Newton loops (1-4 props at changing points)
        # This gives optimal performance for both use cases

        # Bulk multi-output interpolator for full property lookups
        values_dict = {prop: spec[prop] for prop in TOTAL_PROPS}
        self._bulk_interp = MultiOutputTrilinearInterp(points, values_dict)

        # Separate InterpND instances for Newton loop lookups
        # These are more efficient when points change frequently
        self._interps = {}
        for prop in TOTAL_PROPS:
            self._interps[prop] = InterpND(
                method='3D-slinear',
                points=points,
                values=spec[prop],
                extrapolate=True
            )

        # Caches for solver initial guesses (store previous converged solutions)
        # These allow warm-starting Newton solvers from the last solution
        self._cache_T_from_hP = None  # T (SI)
        self._cache_T_from_SP = None  # T (SI)
        self._cache_static_MN = None  # (Ts, Ps) in SI

        # Cache for h value/gradient from T_from_hP to avoid redundant lookup in linearize
        # Format: (T_si, P_si, FAR, h_si, grad_h) where grad_h is (dh/dFAR, dh/dP, dh/dT)
        self._cache_h_for_linearize = None

        # Flags to control whether to apply empirical guess or use cached value
        self._needs_guess_T_from_hP = True
        self._needs_guess_T_from_SP = True
        self._needs_guess_static_MN = True

        # Solver statistics for logging
        self._solver_stats = {
            'T_from_hP': {'calls': 0, 'guesses': 0, 'retries': 0},
            'T_from_SP': {'calls': 0, 'guesses': 0, 'retries': 0},
            'static_MN': {'calls': 0, 'guesses': 0, 'retries': 0},
        }

    def _get_FAR(self, FAR):
        """Return FAR if provided, else instance default."""
        return FAR if FAR is not None else self.FAR

    def print_solver_stats(self):
        """Print solver caching statistics."""
        print("\n=== TabularThermo Solver Statistics ===")
        for solver, stats in self._solver_stats.items():
            calls = stats['calls']
            guesses = stats['guesses']
            retries = stats['retries']
            cache_hits = calls - guesses if calls > 0 else 0
            hit_rate = 100.0 * cache_hits / calls if calls > 0 else 0.0
            print(f"  {solver}:")
            print(f"    calls: {calls}, guesses: {guesses}, retries: {retries}")
            print(f"    cache hit rate: {hit_rate:.1f}%")
        print("========================================\n")

    def reset_solver_stats(self):
        """Reset solver statistics counters."""
        for solver in self._solver_stats:
            self._solver_stats[solver] = {'calls': 0, 'guesses': 0, 'retries': 0}

    @classmethod
    def print_all_solver_stats(cls):
        """Print aggregated solver statistics across all TabularThermo instances."""
        totals = {
            'T_from_hP': {'calls': 0, 'guesses': 0, 'retries': 0},
            'T_from_SP': {'calls': 0, 'guesses': 0, 'retries': 0},
            'static_MN': {'calls': 0, 'guesses': 0, 'retries': 0},
        }
        for instance in cls._instances:
            for solver in totals:
                for key in ('calls', 'guesses', 'retries'):
                    totals[solver][key] += instance._solver_stats[solver][key]

        print("\n=== TabularThermo Solver Statistics (All Instances) ===")
        print(f"  Number of TabularThermo instances: {len(cls._instances)}")
        for solver, stats in totals.items():
            calls = stats['calls']
            guesses = stats['guesses']
            retries = stats['retries']
            cache_hits = calls - guesses if calls > 0 else 0
            hit_rate = 100.0 * cache_hits / calls if calls > 0 else 0.0
            print(f"  {solver}:")
            print(f"    calls: {calls}, guesses: {guesses}, retries: {retries}")
            print(f"    cache hit rate: {hit_rate:.1f}%")
        print("========================================================\n")

    @classmethod
    def reset_all_solver_stats(cls):
        """Reset solver statistics for all TabularThermo instances."""
        for instance in cls._instances:
            instance.reset_solver_stats()

    @classmethod
    def clear_instances(cls):
        """Clear the instance registry (call between test runs if needed)."""
        cls._instances = []

    def _lookup_si(self, prop, T_si, P_si, FAR=None):
        """Internal lookup function in SI units.
        Uses separate InterpND (optimized for single property, changing points).
        """
        FAR = self._get_FAR(FAR)
        x = np.array([FAR, P_si, T_si])
        return self._interps[prop].interpolate(x)[0]

    def _lookup_si_with_grad(self, prop, T_si, P_si, FAR=None):
        """Internal lookup function in SI units, returning value and gradient.
        Uses separate InterpND (optimized for single property, changing points).
        """
        FAR = self._get_FAR(FAR)
        x = np.array([FAR, P_si, T_si])
        val, grad = self._interps[prop].interpolate(x, compute_derivative=True)
        return val[0], grad[0]

    def _lookup_all_si(self, T_si, P_si, FAR=None, compute_derivative=True):
        """Lookup all properties at once in SI units.
        Uses bulk interpolator (optimized for many properties at same point).
        """
        FAR = self._get_FAR(FAR)
        x = np.array([FAR, P_si, T_si])
        return self._bulk_interp.interpolate_all(x, compute_derivative=compute_derivative)

    def _lookup_subset_si_bulk(self, T_si, P_si, FAR, properties, compute_derivative=True):
        """Lookup a subset of properties using bulk interpolator.
        Use when point is stable (same point multiple times).
        """
        x = np.array([FAR, P_si, T_si])
        return self._bulk_interp.interpolate_subset(x, properties, compute_derivative=compute_derivative)

    def _lookup_subset_si(self, T_si, P_si, FAR, properties, compute_derivative=True):
        """Lookup a subset of properties using separate InterpND instances.
        Use for Newton loops where point changes each iteration.
        """
        x = np.array([FAR, P_si, T_si])
        values = {}
        gradients = {} if compute_derivative else None
        for prop in properties:
            if compute_derivative:
                val, grad = self._interps[prop].interpolate(x, compute_derivative=True)
                values[prop] = val[0]
                gradients[prop] = grad[0]
            else:
                values[prop] = self._interps[prop].interpolate(x)[0]
        return values, gradients

    # =========================================================================
    # Linearization and JAX-compatible derivatives
    # =========================================================================

    def linearize(self, T, P, FAR=None, props=None):
        """
        Compute property values and cache gradients at the given state.

        This combines the forward pass (property lookup) with linearization
        (gradient computation) to avoid duplicate table accesses.

        Parameters
        ----------
        T : float
            Temperature (in input units)
        P : float
            Pressure (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.
        props : TotalProps, optional
            Pre-computed property values from forward pass. If provided,
            skips the table lookups and only computes gradients.

        Returns
        -------
        TotalProps
            Named tuple with (h, S, gamma, Cp, Cv, rho, R) in input units
        """
        FAR = self._get_FAR(FAR)

        # Convert inputs to SI for lookup
        T_si = self._convert_T_to_si(T)
        P_si = P * self._P_to_si

        # Store the linearization point
        self._lin_T = T
        self._lin_P = P
        self._lin_FAR = FAR
        self._lin_T_si = T_si
        self._lin_P_si = P_si

        # Check if we have cached h from a recent T_from_hP call at this point
        h_cached = False
        cached_h = None
        cached_h_grad = None
        if self._cache_h_for_linearize is not None:
            cache_T, cache_P, cache_FAR, cache_h, cache_grad = self._cache_h_for_linearize
            if (abs(T_si - cache_T) < 1e-10 and
                abs(P_si - cache_P) < 1e-10 and
                abs(FAR - cache_FAR) < 1e-10):
                # Reuse cached h value and gradient from T_from_hP
                cached_h = cache_h
                cached_h_grad = cache_grad
                h_cached = True
            # Clear the cache after use (it's only valid for the immediate next call)
            self._cache_h_for_linearize = None

        # Bulk lookup: get all values and gradients in one call
        props_si, gradients_si = self._lookup_all_si(T_si, P_si, FAR, compute_derivative=True)

        # If we had cached h, use that instead
        if h_cached:
            props_si['h'] = cached_h
            gradients_si['h'] = cached_h_grad

        self._gradients_si = gradients_si

        # If props provided from forward pass, use those; otherwise use computed
        if props is not None:
            return props
        else:
            return self._convert_total_props_from_si(TotalProps(
                h=props_si['h'], S=props_si['S'], gamma=props_si['gamma'],
                Cp=props_si['Cp'], Cv=props_si['Cv'], rho=props_si['rho'], R=props_si['R']
            ))

    def _get_output_unit_factor(self, prop):
        """Get unit conversion factor for a property."""
        unit_type = _PROP_UNIT_FACTORS.get(prop, 'dimensionless')
        if unit_type == 'dimensionless':
            return 1.0
        elif unit_type == 'h':
            return self._h_from_si
        elif unit_type == 'S':
            return self._S_from_si
        elif unit_type == 'rho':
            return self._rho_from_si
        return 1.0

    def jvp(self, T_dot, P_dot, FAR_dot=0.0):
        """
        Compute Jacobian-vector product (forward-mode autodiff).

        Parameters
        ----------
        T_dot : float
            Tangent vector for temperature
        P_dot : float
            Tangent vector for pressure
        FAR_dot : float, optional
            Tangent vector for fuel-to-air ratio. Default is 0.0.

        Returns
        -------
        dict
            Dictionary of property tangents: {'h': h_dot, 'S': S_dot, ...}
        """
        if not hasattr(self, '_gradients_si'):
            raise RuntimeError("Must call linearize() before jvp()")

        result = {}
        for prop in TOTAL_PROPS:
            # gradients are (dProp/dFAR, dProp/dP, dProp/dT) in SI
            grad = self._gradients_si[prop]
            out_factor = self._get_output_unit_factor(prop)

            # Convert gradient to input units
            # grad is (dProp/dFAR, dProp/dP, dProp/dT) in SI
            dprop_dFAR = grad[0] * out_factor  # FAR is dimensionless
            dprop_dP = grad[1] * self._P_to_si * out_factor
            dprop_dT = grad[2] * self._T_to_si * out_factor

            result[prop] = dprop_dT * T_dot + dprop_dP * P_dot + dprop_dFAR * FAR_dot

        return result

    def vjp(self, h_bar=0.0, S_bar=0.0, gamma_bar=0.0, Cp_bar=0.0,
            Cv_bar=0.0, rho_bar=0.0, R_bar=0.0):
        """
        Compute vector-Jacobian product (reverse-mode autodiff).

        Parameters
        ----------
        h_bar : float
            Cotangent (gradient) for enthalpy
        S_bar : float
            Cotangent for entropy
        gamma_bar : float
            Cotangent for gamma
        Cp_bar : float
            Cotangent for Cp
        Cv_bar : float
            Cotangent for Cv
        rho_bar : float
            Cotangent for rho
        R_bar : float
            Cotangent for R

        Returns
        -------
        tuple
            (T_bar, P_bar, FAR_bar) - gradients with respect to T, P, and FAR
        """
        if not hasattr(self, '_gradients_si'):
            raise RuntimeError("Must call linearize() before vjp()")

        cotangents = {
            'h': h_bar, 'S': S_bar, 'gamma': gamma_bar,
            'Cp': Cp_bar, 'Cv': Cv_bar, 'rho': rho_bar, 'R': R_bar
        }

        T_bar = 0.0
        P_bar = 0.0
        FAR_bar = 0.0

        for prop, cotan in cotangents.items():
            if cotan != 0.0:
                grad = self._gradients_si[prop]
                out_factor = self._get_output_unit_factor(prop)

                # grad is (dProp/dFAR, dProp/dP, dProp/dT) in SI
                dprop_dFAR = grad[0] * out_factor  # FAR is dimensionless
                dprop_dP = grad[1] * self._P_to_si * out_factor
                dprop_dT = grad[2] * self._T_to_si * out_factor

                T_bar = T_bar + dprop_dT * cotan
                P_bar = P_bar + dprop_dP * cotan
                FAR_bar = FAR_bar + dprop_dFAR * cotan

        return T_bar, P_bar, FAR_bar

    # =========================================================================
    # Total property calculations
    # =========================================================================

    def props_TP(self, T, P, FAR=None):
        """Compute all thermodynamic properties from T and P.

        Parameters
        ----------
        T : float
            Temperature (in input units)
        P : float
            Pressure (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.

        Returns
        -------
        TotalProps
            Named tuple with (h, S, gamma, Cp, Cv, rho, R) in input units
        """
        FAR = self._get_FAR(FAR)

        # Convert inputs to SI
        T_si = self._convert_T_to_si(T)
        P_si = P * self._P_to_si

        # Bulk lookup in SI (single cell search, all properties at once)
        values, _ = self._lookup_all_si(T_si, P_si, FAR, compute_derivative=False)

        props_si = TotalProps(
            h=values['h'],
            S=values['S'],
            gamma=values['gamma'],
            Cp=values['Cp'],
            Cv=values['Cv'],
            rho=values['rho'],
            R=values['R']
        )

        # Convert outputs from SI
        return self._convert_total_props_from_si(props_si)

    def h(self, T, P, FAR=None):
        """Compute enthalpy from T and P."""
        return self.props_TP(T, P, FAR).h

    def S(self, T, P, FAR=None):
        """Compute entropy from T and P."""
        return self.props_TP(T, P, FAR).S

    def gamma(self, T, P, FAR=None):
        """Compute ratio of specific heats from T and P."""
        return self.props_TP(T, P, FAR).gamma

    def Cp(self, T, P, FAR=None):
        """Compute specific heat at constant pressure from T and P."""
        return self.props_TP(T, P, FAR).Cp

    def Cv(self, T, P, FAR=None):
        """Compute specific heat at constant volume from T and P."""
        return self.props_TP(T, P, FAR).Cv

    def rho(self, T, P, FAR=None):
        """Compute density from T and P."""
        return self.props_TP(T, P, FAR).rho

    def R(self, T, P, FAR=None):
        """Compute specific gas constant from T and P."""
        return self.props_TP(T, P, FAR).R

    # =========================================================================
    # Inverse calculations (solve for T)
    # =========================================================================

    def T_from_hP(self, h_target, P, FAR=None):
        """Solve for temperature given enthalpy and pressure.

        Uses Newton's method with analytical derivatives for fast convergence.

        Parameters
        ----------
        h_target : float
            Target enthalpy (in input units)
        P : float
            Pressure (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.

        Returns
        -------
        float
            Temperature (in input units) such that h(T, P, FAR) = h_target
        """
        FAR = self._get_FAR(FAR)

        # Convert inputs to SI
        h_si = h_target * self._h_to_si
        P_si = P * self._P_to_si

        T_si = self._T_from_hP_si(h_si, P_si, FAR)

        # Convert output from SI
        return self._convert_T_from_si(T_si)

    # Shared Newton solver instances (class-level to avoid repeated instantiation)
    _newton_solver_1d = NewtonSolver(
        max_iter=20, tol=1e-10, convergence_mode='relative',
        abs_tol=1e-6, bounds=(160.0, 2400.0)
    )

    def _T_from_hP_si(self, h_target_si, P_si, FAR, _retry=False):
        """Solve for temperature given enthalpy and pressure (SI units).

        Uses Newton's method with analytical derivatives. Uses cached solution
        from previous solve as initial guess when available.

        Parameters
        ----------
        h_target_si : float
            Target enthalpy in SI (J/kg)
        P_si : float
            Pressure in SI (Pa)
        FAR : float
            Fuel-to-air ratio
        _retry : bool
            Internal flag to prevent infinite recursion on retry

        Returns
        -------
        float
            Temperature in SI (K)
        """
        # Track solver stats
        if not _retry:
            self._solver_stats['T_from_hP']['calls'] += 1

        # Apply initial guess if needed, otherwise use cached value
        if self._needs_guess_T_from_hP:
            # Empirical initial guess: h ≈ Cp * T, so T ≈ h / Cp
            T0 = max(300.0, min(2000.0, abs(h_target_si) / 1000.0 + 300.0))
            self._needs_guess_T_from_hP = False
            self._solver_stats['T_from_hP']['guesses'] += 1
        else:
            T0 = self._cache_T_from_hP

        # Closure to capture state for residual/Jacobian computation
        # Store last computed values for caching after solve
        cache = {}

        def residual_and_jac(T):
            values, gradients = self._lookup_subset_si(T, P_si, FAR, ['h'], compute_derivative=True)
            h = values['h']
            grad_h = gradients['h']  # (dh/dFAR, dh/dP, dh/dT)
            # Store for post-solve caching
            cache['h'] = h
            cache['grad_h'] = grad_h
            cache['T'] = T
            return h - h_target_si, grad_h[2]

        # Solve using generic Newton solver
        T, converged, _ = self._newton_solver_1d.solve(residual_and_jac, T0, ref_value=h_target_si)

        # If not converged and haven't retried, reset guess flag and retry once
        if not converged and not _retry:
            self._needs_guess_T_from_hP = True
            self._solver_stats['T_from_hP']['retries'] += 1
            return self._T_from_hP_si(h_target_si, P_si, FAR, _retry=True)

        # Cache the converged solution
        self._cache_T_from_hP = T

        # Cache h value and gradient for reuse in linearize()
        # This avoids redundant lookup when linearize is called right after T_from_hP
        if cache:
            self._cache_h_for_linearize = (
                cache['T'], P_si, FAR, cache['h'], cache['grad_h'].copy()
            )

        return T

    def T_from_SP(self, S_target, P, FAR=None):
        """Solve for temperature given entropy and pressure.

        Uses Newton's method with analytical derivatives for fast convergence.

        Parameters
        ----------
        S_target : float
            Target entropy (in input units)
        P : float
            Pressure (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.

        Returns
        -------
        float
            Temperature (in input units) such that S(T, P, FAR) = S_target
        """
        FAR = self._get_FAR(FAR)

        # Convert inputs to SI
        S_si = S_target * self._S_to_si
        P_si = P * self._P_to_si

        T_si = self._T_from_SP_si(S_si, P_si, FAR)

        return self._convert_T_from_si(T_si)

    # =========================================================================
    # Static property calculations
    # =========================================================================

    def _ideal_gas_Ps_guess(self, Tt, Pt, MN, gamma):
        """
        Compute initial guess for static pressure using ideal gas isentropic relations.

        Parameters
        ----------
        Tt : float
            Total temperature
        Pt : float
            Total pressure
        MN : float
            Mach number
        gamma : float
            Ratio of specific heats (at total conditions)

        Returns
        -------
        Ps : float
            Initial guess for static pressure
        """
        gm1 = gamma - 1.0
        temp_ratio = 1.0 / (1.0 + gm1 / 2.0 * MN**2)
        return Pt * temp_ratio ** (gamma / gm1)

    def _static_from_MN_si(self, Tt_si, Pt_si, MN, W_si, FAR=None, _retry=False):
        """Compute static properties in SI units.

        Uses the same physics as the original pyCycle:
        1. Energy conservation: ht = hs + V²/2 where V = MN * Vsonic
        2. Entropy conservation: S(Ts, Ps) = S_total (isentropic process)

        Uses a 2D Newton solver on (Ts, Ps) with analytical Jacobians for
        fast quadratic convergence. Uses cached solution from previous solve
        as initial guess when available.

        Parameters
        ----------
        Tt_si : float
            Total temperature in SI (K)
        Pt_si : float
            Total pressure in SI (Pa)
        MN : float
            Mach number
        W_si : float
            Mass flow rate in SI (kg/s)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.
        _retry : bool
            Internal flag to prevent infinite recursion on retry
        """
        FAR = self._get_FAR(FAR)

        # Get total properties using bulk lookup
        tot_values, _ = self._lookup_subset_si(Tt_si, Pt_si, FAR, ['h', 'S', 'gamma'],
                                                compute_derivative=False)
        ht = tot_values['h']
        S_total = tot_values['S']
        gamma_t = tot_values['gamma']

        # Handle zero Mach number case (no flow, static = total)
        if MN < 1e-10:
            hs = ht
            Ts = Tt_si
            Ps = Pt_si
            gam_s = gamma_t
            # Get remaining properties
            extra_values, _ = self._lookup_subset_si(Ts, Ps, FAR, ['R', 'Cp', 'Cv'],
                                                      compute_derivative=False)
            R_s = extra_values['R']
            Cp_s = extra_values['Cp']
            Cv_s = extra_values['Cv']
            rhos = Ps / (R_s * Ts)
            return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                              MN=MN, V=0.0, Vsonic=np.sqrt(gam_s * R_s * Ts), area=np.inf,
                              gamma=gam_s, Cp=Cp_s, Cv=Cv_s, S=S_total, R=R_s)

        MN_sq = MN ** 2

        # Track solver stats
        if not _retry:
            self._solver_stats['static_MN']['calls'] += 1

        # Apply initial guess if needed, otherwise use cached values
        if self._needs_guess_static_MN:
            # Initial guesses using ideal gas isentropic relations
            Ps0 = self._ideal_gas_Ps_guess(Tt_si, Pt_si, MN, gamma_t)
            # Ts from isentropic relation: Ts/Tt = (Ps/Pt)^((gamma-1)/gamma)
            Ts0 = Tt_si * (Ps0 / Pt_si) ** ((gamma_t - 1.0) / gamma_t)
            self._solver_stats['static_MN']['guesses'] += 1
            self._needs_guess_static_MN = False
        else:
            Ts0, Ps0 = self._cache_static_MN

        # Cache for storing properties from last iteration (for post-processing)
        cache = {}

        # Closure for 2D residual and Jacobian
        # Residuals (normalized):
        #   R1 = (S(Ts, Ps) - S_total) / S_total  (entropy conservation)
        #   R2 = (hs + MN²·γ·R·Ts/2 - ht) / ht   (energy conservation)
        def residual_and_jac(x):
            Ts, Ps = x[0], x[1]

            # Bulk lookup: Get all 4 properties needed for Newton in ONE call
            values, gradients = self._lookup_subset_si(Ts, Ps, FAR, ['S', 'h', 'gamma', 'R'],
                                                       compute_derivative=True)

            S_s = values['S']
            hs = values['h']
            gamma_s = values['gamma']
            R_s = values['R']

            # Store for post-processing
            cache['hs'] = hs
            cache['gamma_s'] = gamma_s
            cache['R_s'] = R_s

            grad_S = gradients['S']      # (dS/dFAR, dS/dP, dS/dT)
            grad_h = gradients['h']      # (dh/dFAR, dh/dP, dh/dT)
            grad_gamma = gradients['gamma']
            grad_R = gradients['R']

            # Compute residuals (normalized)
            R1 = (S_s - S_total) / S_total
            kinetic = MN_sq * gamma_s * R_s * Ts / 2.0
            ht_calc = hs + kinetic
            R2 = (ht_calc - ht) / ht

            # Jacobian of residuals w.r.t. (Ts, Ps)
            dR1_dTs = grad_S[2] / S_total
            dR1_dPs = grad_S[1] / S_total

            dkinetic_dTs = MN_sq / 2.0 * (
                grad_gamma[2] * R_s * Ts + gamma_s * grad_R[2] * Ts + gamma_s * R_s
            )
            dkinetic_dPs = MN_sq / 2.0 * (
                grad_gamma[1] * R_s * Ts + gamma_s * grad_R[1] * Ts
            )

            dR2_dTs = (grad_h[2] + dkinetic_dTs) / ht
            dR2_dPs = (grad_h[1] + dkinetic_dPs) / ht

            residual = np.array([R1, R2])
            jacobian = np.array([[dR1_dTs, dR1_dPs],
                                 [dR2_dTs, dR2_dPs]])

            return residual, jacobian

        # Create 2D solver with problem-specific bounds
        # Ps must be less than Pt (static < total for subsonic flow)
        solver_2d = NewtonSolver(
            max_iter=20, tol=1e-10, convergence_mode='component',
            bounds=([160.0, Pt_si * 0.001], [2400.0, Pt_si * 0.9999])
        )

        # Solve
        x0 = np.array([Ts0, Ps0])
        x, converged, _ = solver_2d.solve(residual_and_jac, x0)
        Ts, Ps = x[0], x[1]

        # If not converged and haven't retried, reset guess flag and retry once
        if not converged and not _retry:
            self._needs_guess_static_MN = True
            self._solver_stats['static_MN']['retries'] += 1
            return self._static_from_MN_si(Tt_si, Pt_si, MN, W_si, FAR, _retry=True)

        # Cache the converged solution
        self._cache_static_MN = (Ts, Ps)

        # Retrieve properties from last Newton iteration
        hs = cache['hs']
        gamma_s = cache['gamma_s']
        R_s = cache['R_s']

        # Only look up Cp and Cv which weren't needed for the Newton solve
        extra_values, _ = self._lookup_subset_si(Ts, Ps, FAR, ['Cp', 'Cv'], compute_derivative=False)
        Cp_s = extra_values['Cp']
        Cv_s = extra_values['Cv']

        # Speed of sound and velocity (use static properties from Newton loop)
        Vsonic = np.sqrt(gamma_s * R_s * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps / (R_s * Ts)

        # Area from continuity
        area = W_si / (rhos * V) if V > 0 else np.inf

        return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                          MN=MN, V=V, Vsonic=Vsonic, area=area,
                          gamma=gamma_s, Cp=Cp_s, Cv=Cv_s, S=S_total, R=R_s)

    def _T_from_SP_si(self, S_target_si, P_si, FAR, _retry=False):
        """Solve for temperature given entropy and pressure (SI units).

        Uses Newton's method with analytical derivatives. Uses cached solution
        from previous solve as initial guess when available.

        Parameters
        ----------
        S_target_si : float
            Target entropy in SI (J/kg/K)
        P_si : float
            Pressure in SI (Pa)
        FAR : float
            Fuel-to-air ratio
        _retry : bool
            Internal flag to prevent infinite recursion on retry

        Returns
        -------
        float
            Temperature in SI (K)
        """
        # Track solver stats
        if not _retry:
            self._solver_stats['T_from_SP']['calls'] += 1

        # Apply initial guess if needed, otherwise use cached value
        if self._needs_guess_T_from_SP:
            # Empirical initial guess: mid-range temperature
            T0 = 800.0
            self._needs_guess_T_from_SP = False
            self._solver_stats['T_from_SP']['guesses'] += 1
        else:
            T0 = self._cache_T_from_SP

        # Closure for residual/Jacobian computation
        def residual_and_jac(T):
            values, gradients = self._lookup_subset_si(T, P_si, FAR, ['S'], compute_derivative=True)
            S = values['S']
            dS_dT = gradients['S'][2]  # (dS/dFAR, dS/dP, dS/dT)
            return S - S_target_si, dS_dT

        # Solve using generic Newton solver (reuse the 1D solver instance)
        T, converged, _ = self._newton_solver_1d.solve(residual_and_jac, T0, ref_value=S_target_si)

        # If not converged and haven't retried, reset guess flag and retry once
        if not converged and not _retry:
            self._needs_guess_T_from_SP = True
            self._solver_stats['T_from_SP']['retries'] += 1
            return self._T_from_SP_si(S_target_si, P_si, FAR, _retry=True)

        # Cache the converged solution
        self._cache_T_from_SP = T

        return T

    def static_from_MN(self, Tt, Pt, MN, W, FAR=None):
        """Compute static properties from total conditions and Mach number.

        Parameters
        ----------
        Tt : float
            Total temperature (in input units)
        Pt : float
            Total pressure (in input units)
        MN : float
            Mach number
        W : float
            Mass flow rate (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.

        Returns
        -------
        StaticProps
            Named tuple with static properties in input units
        """
        FAR = self._get_FAR(FAR)

        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        W_si = W * self._W_to_si

        props_si = self._static_from_MN_si(Tt_si, Pt_si, MN, W_si, FAR)

        # Convert outputs from SI
        return self._convert_static_props_from_si(props_si)

    # Cache for Newton solver initial guess
    _last_MN_solution = {}

    def static_from_area(self, Tt, Pt, area, W, MN_guess=0.5, subsonic=True, FAR=None):
        """Compute static properties from total conditions and flow area.

        Uses Newton's method with analytical derivatives for fast convergence,
        with brentq fallback for robustness.

        Parameters
        ----------
        Tt : float
            Total temperature (in input units)
        Pt : float
            Total pressure (in input units)
        area : float
            Flow area (in input units)
        W : float
            Mass flow rate (in input units)
        MN_guess : float, optional
            Initial guess for Mach number
        subsonic : bool, optional
            If True, find subsonic solution; if False, find supersonic
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.

        Returns
        -------
        StaticProps
            Named tuple with static properties in input units
        """
        FAR = self._get_FAR(FAR)

        # Try Newton's method first (much faster when it converges)
        # Uses analytical derivatives from linearize_static_MN
        MN, props, converged = self._newton_solve_MN(
            Tt, Pt, area, W, FAR, MN_guess, subsonic
        )

        if converged:
            return props

        # Fall back to brentq for robustness
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        area_si = area * self._area_to_si
        W_si = W * self._W_to_si

        def area_residual(MN):
            props = self._static_from_MN_si(Tt_si, Pt_si, MN, W_si, FAR)
            return float(props.area) - float(area_si)

        if subsonic:
            MN_min, MN_max = 0.01, 0.999
        else:
            MN_min, MN_max = 1.001, 5.0

        MN = brentq(area_residual, MN_min, MN_max, xtol=1e-10)
        props_si = self._static_from_MN_si(Tt_si, Pt_si, MN, W_si, FAR)

        return self._convert_static_props_from_si(props_si)

    def _newton_solve_MN(self, Tt, Pt, area, W, FAR, MN_guess, subsonic,
                         max_iter=10, tol=1e-10):
        """
        Solve for MN using Newton's method with analytical derivatives.

        Uses linearize_static_MN to compute d(area)/d(MN) analytically,
        avoiding the need for finite differences.

        Parameters
        ----------
        Tt : float
            Total temperature (in input units)
        Pt : float
            Total pressure (in input units)
        area : float
            Target flow area (in input units)
        W : float
            Mass flow rate (in input units)
        FAR : float
            Fuel-to-air ratio
        MN_guess : float
            Initial guess for Mach number
        subsonic : bool
            If True, find subsonic solution; if False, find supersonic
        max_iter : int, optional
            Maximum Newton iterations
        tol : float, optional
            Convergence tolerance (relative)

        Returns
        -------
        MN : float
            Converged Mach number
        props : StaticProps
            Static properties at converged MN (in input units)
        converged : bool
            True if Newton converged
        """
        # Use cached solution as initial guess if available
        # Cache key uses input values (user units)
        cache_key = (round(Tt, 2), round(Pt, 0), round(W, 4), round(FAR, 4), subsonic)
        if cache_key in self._last_MN_solution:
            MN = self._last_MN_solution[cache_key]
        else:
            MN = MN_guess if MN_guess > 0.01 else (0.4 if subsonic else 1.5)

        # Bounds for subsonic/supersonic
        MN_min = 0.01 if subsonic else 1.001
        MN_max = 0.999 if subsonic else 5.0

        props = None
        for _ in range(max_iter):
            # Compute static properties at current MN
            props = self.static_from_MN(Tt, Pt, MN, W, FAR)
            area_computed = props.area

            # Get analytical derivative d(area)/d(MN) via linearize_static_MN
            # Pass sprops to avoid recomputing static properties
            self.linearize_static_MN(Tt, Pt, MN, W, FAR, sprops=props)
            darea_dMN = self._jacobian_static_MN['area'][2]

            residual = area_computed - area

            # Check convergence
            if abs(residual) < tol * area:
                self._last_MN_solution[cache_key] = MN
                return MN, props, True

            # Newton update
            if abs(darea_dMN) < 1e-20:
                return MN, props, False  # Derivative too small

            MN_new = MN - residual / darea_dMN

            # Clamp to bounds
            MN_new = max(MN_min, min(MN_max, MN_new))

            # Check for stagnation
            if abs(MN_new - MN) < 1e-14:
                self._last_MN_solution[cache_key] = MN_new
                return MN_new, props, True

            MN = MN_new

        return MN, props, False  # Did not converge

    # =========================================================================
    # Analytical derivatives for static properties
    # =========================================================================

    # Profiling stats for linearize_static_MN
    _profile_static_MN = {
        'calls': 0,
        'total_time': 0.0,
        'lookup_total': 0.0,
        'grad_total': 0.0,
        'grad_static': 0.0,
        'isentropic': 0.0,
        'jacobian': 0.0,
    }

    @classmethod
    def print_profile_static_MN(cls):
        p = cls._profile_static_MN
        print("\n=== linearize_static_MN Profile ===")
        print(f"  calls: {p['calls']}")
        print(f"  total_time: {p['total_time']*1000:.3f} ms")
        if p['calls'] > 0:
            print(f"  avg_time: {p['total_time']/p['calls']*1000:.3f} ms")
        print(f"  breakdown:")
        print(f"    lookup_total: {p['lookup_total']*1000:.3f} ms ({100*p['lookup_total']/max(p['total_time'],1e-9):.1f}%)")
        print(f"    grad_total: {p['grad_total']*1000:.3f} ms ({100*p['grad_total']/max(p['total_time'],1e-9):.1f}%)")
        print(f"    grad_static: {p['grad_static']*1000:.3f} ms ({100*p['grad_static']/max(p['total_time'],1e-9):.1f}%)")
        print(f"    isentropic: {p['isentropic']*1000:.3f} ms ({100*p['isentropic']/max(p['total_time'],1e-9):.1f}%)")
        print(f"    jacobian: {p['jacobian']*1000:.3f} ms ({100*p['jacobian']/max(p['total_time'],1e-9):.1f}%)")
        print("====================================\n")

    @classmethod
    def reset_profile_static_MN(cls):
        for k in cls._profile_static_MN:
            cls._profile_static_MN[k] = 0.0 if k != 'calls' else 0

    def linearize_static_MN(self, Tt, Pt, MN, W, FAR=None, sprops=None):
        """
        Compute static properties and cache analytical gradients for static_from_MN.

        Uses the implicit function theorem to compute derivatives through the
        energy and entropy constraints:
        - F(Ts, Ps) = S(Ts, Ps) - S_total = 0 (entropy constraint)
        - G(Ts, Ps) = hs + MN²*gamma*R*Ts/2 - ht = 0 (energy constraint)

        The Jacobian of (Ts, Ps) w.r.t. inputs is computed analytically using
        the interpolator's gradient() method.

        Parameters
        ----------
        Tt : float
            Total temperature (in input units)
        Pt : float
            Total pressure (in input units)
        MN : float
            Mach number
        W : float
            Mass flow rate (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.
        sprops : StaticProps, optional
            Pre-computed static properties from forward pass. If provided,
            uses these for the primal values.

        Returns
        -------
        StaticProps
            Named tuple with static properties in input units
        """
        t_start = time.perf_counter()
        p = self._profile_static_MN
        FAR = self._get_FAR(FAR)

        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        W_si = W * self._W_to_si

        # Compute primal static properties if not provided
        if sprops is not None:
            props = sprops
            Ts_si = self._convert_T_to_si(props.Ts)
            Ps_si = props.Ps * self._P_to_si
        else:
            props = self.static_from_MN(Tt, Pt, MN, W, FAR)
            Ts_si = self._convert_T_to_si(props.Ts)
            Ps_si = props.Ps * self._P_to_si

        # Get total properties and their gradients (bulk call - h and S at total conditions)
        tot_values, tot_grads = self._lookup_subset_si(Tt_si, Pt_si, FAR, ['h', 'S'],
                                                        compute_derivative=True)
        ht_si = tot_values['h']
        S_total = tot_values['S']
        grad_ht_tot = tot_grads['h']  # (dh/dFAR, dh/dP, dh/dT)
        grad_S_tot = tot_grads['S']   # (dS/dFAR, dS/dP, dS/dT)

        # Get static properties and their gradients at (Ts, Ps, FAR) (bulk call - all needed props)
        stat_values, stat_grads = self._lookup_subset_si(
            Ts_si, Ps_si, FAR, ['h', 'S', 'gamma', 'R', 'Cp', 'Cv'],
            compute_derivative=True
        )

        hs_si = stat_values['h']
        gamma_s = stat_values['gamma']
        R_s = stat_values['R']

        grad_hs = stat_grads['h']      # (dhs/dFAR, dhs/dPs, dhs/dTs)
        grad_Ss = stat_grads['S']      # (dSs/dFAR, dSs/dPs, dSs/dTs)
        grad_gams = stat_grads['gamma']
        grad_Rs = stat_grads['R']
        grad_Cps = stat_grads['Cp']
        grad_Cvs = stat_grads['Cv']

        MN_sq = MN ** 2

        # Implicit constraints:
        # F = S(Ts, Ps, FAR) - S_total(Tt, Pt, FAR) = 0
        # G = hs(Ts, Ps, FAR) + MN²*gamma_s*R_s*Ts/2 - ht(Tt, Pt, FAR) = 0
        #
        # Jacobian of constraints w.r.t. (Ts, Ps):
        # dF/dTs = dSs/dTs
        # dF/dPs = dSs/dPs
        # dG/dTs = dhs/dTs + MN²/2 * (dgamma/dTs * R_s * Ts + gamma_s * dR/dTs * Ts + gamma_s * R_s)
        # dG/dPs = dhs/dPs + MN²/2 * (dgamma/dPs * R_s * Ts + gamma_s * dR/dPs * Ts)

        dF_dTs = grad_Ss[2]  # dSs/dTs
        dF_dPs = grad_Ss[1]  # dSs/dPs

        kinetic_term = MN_sq * gamma_s * R_s * Ts_si / 2.0
        dG_dTs = grad_hs[2] + MN_sq / 2.0 * (
            grad_gams[2] * R_s * Ts_si + gamma_s * grad_Rs[2] * Ts_si + gamma_s * R_s
        )
        dG_dPs = grad_hs[1] + MN_sq / 2.0 * (
            grad_gams[1] * R_s * Ts_si + gamma_s * grad_Rs[1] * Ts_si
        )

        # Invert the constraint Jacobian using Cramer's rule
        det = dF_dTs * dG_dPs - dF_dPs * dG_dTs
        if abs(det) > 1e-20:
            inv_det = 1.0 / det
            J_inv_00 = dG_dPs * inv_det
            J_inv_01 = -dF_dPs * inv_det
            J_inv_10 = -dG_dTs * inv_det
            J_inv_11 = dF_dTs * inv_det
        else:
            J_inv_00 = J_inv_01 = J_inv_10 = J_inv_11 = 0.0

        # Compute dTs, dPs w.r.t. each input using implicit function theorem:
        # [dTs/dx, dPs/dx]^T = -J_inv @ [dF/dx, dG/dx]^T

        # Derivatives w.r.t. Tt (in input units):
        dF_dTt = -grad_S_tot[2] * self._T_to_si
        dG_dTt = -grad_ht_tot[2] * self._T_to_si
        dTs_dTt_si = -(J_inv_00 * dF_dTt + J_inv_01 * dG_dTt)
        dPs_dTt_si = -(J_inv_10 * dF_dTt + J_inv_11 * dG_dTt)

        # Derivatives w.r.t. Pt (in input units):
        dF_dPt = -grad_S_tot[1] * self._P_to_si
        dG_dPt = -grad_ht_tot[1] * self._P_to_si
        dTs_dPt_si = -(J_inv_00 * dF_dPt + J_inv_01 * dG_dPt)
        dPs_dPt_si = -(J_inv_10 * dF_dPt + J_inv_11 * dG_dPt)

        # Derivatives w.r.t. MN:
        dG_dMN = MN * gamma_s * R_s * Ts_si
        dTs_dMN_si = -(J_inv_01 * dG_dMN)
        dPs_dMN_si = -(J_inv_11 * dG_dMN)

        # Derivatives w.r.t. W: Neither F nor G depends on W directly
        dTs_dW_si, dPs_dW_si = 0.0, 0.0

        # Derivatives w.r.t. FAR:
        dF_dFAR = grad_Ss[0] - grad_S_tot[0]
        dG_dFAR = (grad_hs[0] +
                   MN_sq / 2.0 * (grad_gams[0] * R_s * Ts_si + gamma_s * grad_Rs[0] * Ts_si) -
                   grad_ht_tot[0])
        dTs_dFAR_si = -(J_inv_00 * dF_dFAR + J_inv_01 * dG_dFAR)
        dPs_dFAR_si = -(J_inv_10 * dF_dFAR + J_inv_11 * dG_dFAR)

        # Build derivative arrays for Ts and Ps in input units
        # [d/dTt, d/dPt, d/dMN, d/dW, d/dFAR]
        dTs_d = np.array([dTs_dTt_si, dTs_dPt_si, dTs_dMN_si, dTs_dW_si, dTs_dFAR_si]) * self._T_from_si
        dPs_d = np.array([dPs_dTt_si, dPs_dPt_si, dPs_dMN_si, dPs_dW_si, dPs_dFAR_si]) * self._P_from_si

        # SI versions for chain rule
        dTs_d_si = np.array([dTs_dTt_si, dTs_dPt_si, dTs_dMN_si, dTs_dW_si, dTs_dFAR_si])
        dPs_d_si = np.array([dPs_dTt_si, dPs_dPt_si, dPs_dMN_si, dPs_dW_si, dPs_dFAR_si])

        # Chain rule for property derivatives
        # Property(Ts, Ps, FAR) -> dProp/dx = dProp/dTs * dTs/dx + dProp/dPs * dPs/dx + dProp/dFAR * dFAR/dx
        FAR_derivs = self._FAR_DERIVS

        def chain_rule(grad):
            """grad is (dProp/dFAR, dProp/dPs, dProp/dTs) in SI"""
            return grad[2] * dTs_d_si + grad[1] * dPs_d_si + grad[0] * FAR_derivs

        # Tabular property derivatives (in SI, then convert)
        dhs_d = chain_rule(grad_hs) * self._h_from_si
        dSs_d = chain_rule(grad_Ss) * self._S_from_si
        dgams_d = chain_rule(grad_gams)
        dCps_d = chain_rule(grad_Cps) * self._S_from_si
        dCvs_d = chain_rule(grad_Cvs) * self._S_from_si
        dRs_d_si = chain_rule(grad_Rs)
        dRs_d = dRs_d_si * self._S_from_si

        # Vsonic = sqrt(gamma_s * R_s * Ts)
        Vsonic_si = np.sqrt(gamma_s * R_s * Ts_si)
        if Vsonic_si > 0:
            dVsonic_d_si = (R_s * Ts_si * dgams_d + gamma_s * Ts_si * dRs_d_si +
                            gamma_s * R_s * dTs_d_si) / (2 * Vsonic_si)
        else:
            dVsonic_d_si = np.zeros(5)
        dVsonic_d = dVsonic_d_si * self._V_from_si

        # V = MN * Vsonic
        V_si = MN * Vsonic_si
        dV_d_si = self._MN_DERIVS * Vsonic_si + MN * dVsonic_d_si
        dV_d = dV_d_si * self._V_from_si

        # rhos = Ps / (R_s * Ts)
        rhos_si = Ps_si / (R_s * Ts_si)
        drhos_d_si = (dPs_d_si / (R_s * Ts_si) -
                      Ps_si * dRs_d_si / (R_s ** 2 * Ts_si) -
                      Ps_si * dTs_d_si / (R_s * Ts_si ** 2))
        drhos_d = drhos_d_si * self._rho_from_si

        # area = W / (rhos * V)
        area_si = W_si / (rhos_si * V_si) if V_si > 0 else np.inf
        dW_d_si = np.array([0.0, 0.0, 0.0, self._W_to_si, 0.0])
        if V_si > 0 and rhos_si > 0:
            darea_d_si = (dW_d_si / (rhos_si * V_si) -
                          W_si * drhos_d_si / (rhos_si ** 2 * V_si) -
                          W_si * dV_d_si / (rhos_si * V_si ** 2))
        else:
            darea_d_si = np.zeros(5)
        darea_d = darea_d_si * self._area_from_si

        # Store Jacobian
        self._jacobian_static_MN = {
            'Ts': dTs_d,
            'Ps': dPs_d,
            'hs': dhs_d,
            'rhos': drhos_d,
            'MN': self._MN_DERIVS,
            'V': dV_d,
            'Vsonic': dVsonic_d,
            'area': darea_d,
            'gamma': dgams_d,
            'Cp': dCps_d,
            'Cv': dCvs_d,
            'S': dSs_d,
            'R': dRs_d,
        }

        p['calls'] += 1
        p['total_time'] += time.perf_counter() - t_start

        return props

    def get_jacobian_static_MN(self):
        """
        Return the full Jacobian for static_from_MN.

        Returns
        -------
        dict
            Dictionary mapping property names to arrays of 5 partial derivatives
            [d/dTt, d/dPt, d/dMN, d/dW, d/dFAR]
        """
        if not hasattr(self, '_jacobian_static_MN'):
            raise RuntimeError("Must call linearize_static_MN() before get_jacobian_static_MN()")
        return self._jacobian_static_MN

    def jvp_static_MN(self, Tt_dot, Pt_dot, MN_dot, W_dot, FAR_dot=0.0):
        """
        Compute JVP for static_from_MN using cached Jacobian.

        Parameters
        ----------
        Tt_dot : float
            Tangent for total temperature
        Pt_dot : float
            Tangent for total pressure
        MN_dot : float
            Tangent for Mach number
        W_dot : float
            Tangent for mass flow rate
        FAR_dot : float, optional
            Tangent for fuel-to-air ratio. Default is 0.0.

        Returns
        -------
        dict
            Dictionary of property tangents
        """
        if not hasattr(self, '_jacobian_static_MN'):
            raise RuntimeError("Must call linearize_static_MN() before jvp_static_MN()")

        jac = self._jacobian_static_MN
        tangent = np.array([Tt_dot, Pt_dot, MN_dot, W_dot, FAR_dot])

        # Simple matrix-vector product using pre-computed Jacobian
        result = {}
        for prop in STATIC_PROPS:
            result[prop] = float(np.dot(jac[prop], tangent))

        return result

    def linearize_static_area(self, Tt, Pt, area, W, FAR=None):
        """
        Compute static properties and cache gradients for static_from_area.

        This combines the forward pass and linearization into one call,
        avoiding duplicate lookups when both are needed.

        This uses implicit differentiation since MN is solved via Newton's method.
        The implicit constraint is: area_computed(MN) = area_target
        Using implicit function theorem: dMN/dx = -[d(area)/dMN]^(-1) * d(area)/dx

        Parameters
        ----------
        Tt : float
            Total temperature (in input units)
        Pt : float
            Total pressure (in input units)
        area : float
            Flow area (in input units)
        W : float
            Mass flow rate (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.

        Returns
        -------
        StaticProps
            Named tuple with static properties in input units
        """
        FAR = self._get_FAR(FAR)

        # First compute the solution to get MN
        props = self.static_from_area(Tt, Pt, area, W, FAR=FAR)
        MN = props.MN

        # Now linearize static_from_MN at this MN (this computes full Jacobian)
        self.linearize_static_MN(Tt, Pt, MN, W, FAR)

        # Get full Jacobian from static_from_MN (computed in linearize_static_MN)
        jac_MN = self.get_jacobian_static_MN()

        # Extract area derivatives: [d/dTt, d/dPt, d/dMN, d/dW, d/dFAR]
        darea_dTt = jac_MN['area'][0]
        darea_dPt = jac_MN['area'][1]
        darea_dMN = jac_MN['area'][2]
        darea_dW = jac_MN['area'][3]
        darea_dFAR = jac_MN['area'][4]

        # Implicit function theorem: dMN/dx = -darea_dx / darea_dMN
        # For area_target: constraint is area_computed - area_target = 0
        # So dMN/d(area_target) = +1 / darea_dMN (positive!)
        if abs(darea_dMN) > 1e-20:
            dMN_dTt = -darea_dTt / darea_dMN
            dMN_dPt = -darea_dPt / darea_dMN
            dMN_darea = 1.0 / darea_dMN
            dMN_dW = -darea_dW / darea_dMN
            dMN_dFAR = -darea_dFAR / darea_dMN
        else:
            dMN_dTt = dMN_dPt = dMN_darea = dMN_dW = dMN_dFAR = 0.0

        # Store for backward compatibility with jvp_static_area
        self._static_area_cache = {
            'MN': MN,
            'FAR': FAR,
            'dMN_dTt': dMN_dTt,
            'dMN_dPt': dMN_dPt,
            'dMN_darea': dMN_darea,
            'dMN_dW': dMN_dW,
            'dMN_dFAR': dMN_dFAR,
        }

        # Compute full Jacobian for static_from_area directly
        # Total derivative = partial via MN path
        # d(output)/d(input) = d(output)/dMN * dMN/d(input) for [Tt, Pt, area, W, FAR]
        # where MN implicitly depends on [Tt, Pt, area, W, FAR]

        self._jacobian_static_area = {}
        for prop in STATIC_PROPS:
            # jac_MN[prop] = [d/dTt, d/dPt, d/dMN, d/dW, d/dFAR] from static_from_MN
            # For static_from_area inputs are [Tt, Pt, area, W, FAR]
            # Total deriv = direct (through Tt, Pt, W, FAR) + indirect (through MN)
            jac_mn = jac_MN[prop]
            # Direct effects: jac_mn[0] for Tt, jac_mn[1] for Pt, jac_mn[3] for W, jac_mn[4] for FAR
            # MN effect: jac_mn[2] * dMN/d*
            self._jacobian_static_area[prop] = np.array([
                jac_mn[0] + jac_mn[2] * dMN_dTt,    # d/dTt
                jac_mn[1] + jac_mn[2] * dMN_dPt,    # d/dPt
                jac_mn[2] * dMN_darea,              # d/darea (only through MN)
                jac_mn[3] + jac_mn[2] * dMN_dW,     # d/dW
                jac_mn[4] + jac_mn[2] * dMN_dFAR,   # d/dFAR
            ])

        return props

    def get_jacobian_static_area(self):
        """
        Return the full Jacobian for static_from_area.

        Returns
        -------
        dict
            Dictionary mapping property names to arrays of 5 partial derivatives
            [d/dTt, d/dPt, d/darea, d/dW, d/dFAR]
        """
        if not hasattr(self, '_jacobian_static_area'):
            raise RuntimeError("Must call linearize_static_area() before get_jacobian_static_area()")
        return self._jacobian_static_area

    def jvp_static_area(self, Tt_dot, Pt_dot, area_dot, W_dot, FAR_dot=0.0):
        """
        Compute JVP for static_from_area using cached Jacobian.

        Parameters
        ----------
        Tt_dot : float
            Tangent for total temperature
        Pt_dot : float
            Tangent for total pressure
        area_dot : float
            Tangent for flow area
        W_dot : float
            Tangent for mass flow rate
        FAR_dot : float, optional
            Tangent for fuel-to-air ratio. Default is 0.0.

        Returns
        -------
        dict
            Dictionary of property tangents
        """
        if not hasattr(self, '_jacobian_static_area'):
            raise RuntimeError("Must call linearize_static_area() before jvp_static_area()")

        jac = self._jacobian_static_area
        tangent = np.array([Tt_dot, Pt_dot, area_dot, W_dot, FAR_dot])

        # Simple matrix-vector product using pre-computed Jacobian
        result = {}
        for prop in STATIC_PROPS:
            result[prop] = float(np.dot(jac[prop], tangent))

        return result

    def static_from_Ps(self, Tt, Pt, Ps, W, FAR=None):
        """Compute static properties from total conditions and static pressure.

        Uses the same physics as the original pyCycle (PsCalc):
        1. Entropy conservation: S(Ts, Ps) = S_total (find Ts)
        2. Energy conservation: V = sqrt(2 * (ht - hs)) [in SI units: J/kg -> m/s]
        3. MN = V / Vsonic

        Parameters
        ----------
        Tt : float
            Total temperature (in input units)
        Pt : float
            Total pressure (in input units)
        Ps : float
            Static pressure (in input units)
        W : float
            Mass flow rate (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.

        Returns
        -------
        StaticProps
            Named tuple with static properties in input units
        """
        FAR = self._get_FAR(FAR)

        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        Ps_si = Ps * self._P_to_si
        W_si = W * self._W_to_si

        # Get total properties in SI (J/kg for enthalpy) using bulk lookup
        tot_values, _ = self._lookup_subset_si(Tt_si, Pt_si, FAR, ['h', 'S'],
                                                compute_derivative=False)
        ht_si = tot_values['h']
        S_total = tot_values['S']

        # Find Ts from entropy constraint: S(Ts, Ps) = S_total
        Ts_si = self._T_from_SP_si(S_total, Ps_si, FAR)

        # Full static properties at (Ts, Ps) in SI using bulk lookup
        stat_values, _ = self._lookup_subset_si(Ts_si, Ps_si, FAR,
                                                 ['h', 'gamma', 'Cp', 'Cv', 'R'],
                                                 compute_derivative=False)
        hs_si = stat_values['h']
        gam_s = stat_values['gamma']
        Cp_s = stat_values['Cp']
        Cv_s = stat_values['Cv']
        R_s = stat_values['R']

        # Speed of sound in SI (m/s)
        Vsonic_si = np.sqrt(gam_s * R_s * Ts_si)

        # Velocity from energy conservation: ht = hs + V²/2
        # V = sqrt(2 * (ht - hs)) in SI units (J/kg -> m/s)
        # Handle case where ht < hs (inverted, as in original PsCalc)
        if ht_si >= hs_si:
            V_si = np.sqrt(2.0 * (ht_si - hs_si))
        else:
            V_si = np.sqrt(2.0 * (hs_si - ht_si))

        # Mach number
        MN = V_si / Vsonic_si

        # Density from ideal gas law (SI)
        rhos_si = Ps_si / (R_s * Ts_si)

        # Area from continuity (SI)
        area_si = W_si / (rhos_si * V_si) if V_si > 0 else np.inf

        props_si = StaticProps(Ts=Ts_si, Ps=Ps_si, hs=hs_si, rhos=rhos_si,
                              MN=MN, V=V_si, Vsonic=Vsonic_si, area=area_si,
                              gamma=gam_s, Cp=Cp_s, Cv=Cv_s, S=S_total, R=R_s)

        return self._convert_static_props_from_si(props_si)
