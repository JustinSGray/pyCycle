"""
Tabular thermodynamic property calculations using interpolation.
"""

import time
from contextlib import contextmanager

import numpy as np
from scipy.optimize import brentq

from ..base import ThermoInterface, TotalProps, StaticProps


# Property name constants to avoid repetition
TOTAL_PROPS = ('h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R')
STATIC_PROPS = ('Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area', 'gamma', 'Cp', 'Cv', 'S', 'R')

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

    def __init__(self, FAR=0.0, spec=None, input_units='SI'):
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

        # Create interpolators for each property
        # Grid points: (FAR, P, T) - tables are in SI units
        points = (spec['FAR'], spec['P'], spec['T'])

        # Use 3D-slinear for 3D grids - it's ~2.3x faster than generic slinear
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
        """Internal lookup function in SI units."""
        FAR = self._get_FAR(FAR)
        x = np.array([FAR, P_si, T_si])
        return self._interps[prop].interpolate(x)[0]

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

        x = np.array([FAR, P_si, T_si])

        # Store the linearization point
        self._lin_T = T
        self._lin_P = P
        self._lin_FAR = FAR
        self._lin_T_si = T_si
        self._lin_P_si = P_si

        # Compute values and gradients together using compute_derivative=True
        # This avoids redundant cell lookups
        self._gradients_si = {}
        props_si = {}
        for prop in TOTAL_PROPS:
            val, grad = self._interps[prop].interpolate(x, compute_derivative=True)
            props_si[prop] = val[0]
            self._gradients_si[prop] = grad[0]

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

        # Lookup in SI
        props_si = TotalProps(
            h=self._lookup_si('h', T_si, P_si, FAR),
            S=self._lookup_si('S', T_si, P_si, FAR),
            gamma=self._lookup_si('gamma', T_si, P_si, FAR),
            Cp=self._lookup_si('Cp', T_si, P_si, FAR),
            Cv=self._lookup_si('Cv', T_si, P_si, FAR),
            rho=self._lookup_si('rho', T_si, P_si, FAR),
            R=self._lookup_si('R', T_si, P_si, FAR)
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
        max_iter = 20
        tol = 1e-10

        # Track solver stats
        if not _retry:
            self._solver_stats['T_from_hP']['calls'] += 1

        # Apply initial guess if needed, otherwise use cached value
        if self._needs_guess_T_from_hP:
            # Empirical initial guess: h ≈ Cp * T, so T ≈ h / Cp
            T = max(300.0, min(2000.0, abs(h_target_si) / 1000.0 + 300.0))
            self._needs_guess_T_from_hP = False
            self._solver_stats['T_from_hP']['guesses'] += 1
        else:
            T = self._cache_T_from_hP

        converged = False
        for _ in range(max_iter):
            x = np.array([FAR, P_si, T])

            # Get h and dh/dT in single call (avoids redundant cell lookup)
            h_arr, grad_h_2d = self._interps['h'].interpolate(x, compute_derivative=True)
            h = h_arr[0]
            dh_dT = grad_h_2d[0, 2]  # (dh/dFAR, dh/dP, dh/dT)

            # Residual and derivative
            residual = h - h_target_si

            # Check convergence (relative tolerance)
            if abs(residual) < tol * abs(h_target_si) or abs(residual) < 1e-6:
                converged = True
                break

            # Newton update
            if abs(dh_dT) < 1e-30:
                break  # Derivative too small

            dT = -residual / dh_dT

            # Update with bounds enforcement
            T_new = T + dT
            T_new = max(160.0, min(2400.0, T_new))

            # Check for stagnation
            if abs(T_new - T) < 1e-12:
                T = T_new
                break

            T = T_new

        # If not converged and haven't retried, reset guess flag and retry once
        if not converged and not _retry:
            self._needs_guess_T_from_hP = True
            self._solver_stats['T_from_hP']['retries'] += 1
            return self._T_from_hP_si(h_target_si, P_si, FAR, _retry=True)

        # Cache the converged solution
        self._cache_T_from_hP = T

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

        # Get total properties
        ht = self._lookup_si('h', Tt_si, Pt_si, FAR)
        S_total = self._lookup_si('S', Tt_si, Pt_si, FAR)
        gamma_t = self._lookup_si('gamma', Tt_si, Pt_si, FAR)

        # Handle zero Mach number case (no flow, static = total)
        if MN < 1e-10:
            hs = ht
            Ts = Tt_si
            Ps = Pt_si
            gam_s = gamma_t
            R_s = self._lookup_si('R', Ts, Ps, FAR)
            Cp_s = self._lookup_si('Cp', Ts, Ps, FAR)
            Cv_s = self._lookup_si('Cv', Ts, Ps, FAR)
            rhos = Ps / (R_s * Ts)
            return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                              MN=MN, V=0.0, Vsonic=np.sqrt(gam_s * R_s * Ts), area=np.inf,
                              gamma=gam_s, Cp=Cp_s, Cv=Cv_s, S=S_total, R=R_s)

        MN_sq = MN ** 2
        max_iter = 20
        tol = 1e-10

        # Track solver stats
        if not _retry:
            self._solver_stats['static_MN']['calls'] += 1

        # Apply initial guess if needed, otherwise use cached values
        if self._needs_guess_static_MN:
            # Initial guesses using ideal gas isentropic relations
            Ps = self._ideal_gas_Ps_guess(Tt_si, Pt_si, MN, gamma_t)
            # Ts from isentropic relation: Ts/Tt = (Ps/Pt)^((gamma-1)/gamma)
            Ts = Tt_si * (Ps / Pt_si) ** ((gamma_t - 1.0) / gamma_t)
            self._solver_stats['static_MN']['guesses'] += 1
            self._needs_guess_static_MN = False
        else:
            Ts, Ps = self._cache_static_MN

        # 2D Newton solver for coupled (Ts, Ps) system
        # Residuals:
        #   R1 = S(Ts, Ps) - S_total = 0  (entropy conservation)
        #   R2 = hs + MN²·γ·R·Ts/2 - ht = 0  (energy conservation)

        converged = False
        for _ in range(max_iter):
            # Evaluate properties and gradients at current (Ts, Ps)
            x = np.array([FAR, Ps, Ts])

            # Get values AND gradients in single calls (avoids redundant cell lookups)
            # Each call returns (value_array, gradient_2d_array)
            S_arr, grad_S_2d = self._interps['S'].interpolate(x, compute_derivative=True)
            h_arr, grad_h_2d = self._interps['h'].interpolate(x, compute_derivative=True)
            gam_arr, grad_gam_2d = self._interps['gamma'].interpolate(x, compute_derivative=True)
            R_arr, grad_R_2d = self._interps['R'].interpolate(x, compute_derivative=True)

            S_s = S_arr[0]
            hs = h_arr[0]
            gamma_s = gam_arr[0]
            R_s = R_arr[0]

            grad_S = grad_S_2d[0]
            grad_h = grad_h_2d[0]
            grad_gamma = grad_gam_2d[0]
            grad_R = grad_R_2d[0]

            # Compute residuals
            # R1: entropy conservation (normalized)
            R1 = (S_s - S_total) / S_total

            # R2: energy conservation (normalized)
            # ht = hs + MN² * gamma_s * R_s * Ts / 2
            kinetic = MN_sq * gamma_s * R_s * Ts / 2.0
            ht_calc = hs + kinetic
            R2 = (ht_calc - ht) / ht

            # Check convergence
            if abs(R1) < tol and abs(R2) < tol:
                converged = True
                break

            # Jacobian of residuals w.r.t. (Ts, Ps)
            # dR1/dTs = (dS/dTs) / S_total
            # dR1/dPs = (dS/dPs) / S_total
            dR1_dTs = grad_S[2] / S_total
            dR1_dPs = grad_S[1] / S_total

            # dR2/dTs = (dhs/dTs + MN²/2 * (dγ/dTs·R·Ts + γ·dR/dTs·Ts + γ·R)) / ht
            # dR2/dPs = (dhs/dPs + MN²/2 * (dγ/dPs·R·Ts + γ·dR/dPs·Ts)) / ht
            dkinetic_dTs = MN_sq / 2.0 * (
                grad_gamma[2] * R_s * Ts + gamma_s * grad_R[2] * Ts + gamma_s * R_s
            )
            dkinetic_dPs = MN_sq / 2.0 * (
                grad_gamma[1] * R_s * Ts + gamma_s * grad_R[1] * Ts
            )

            dR2_dTs = (grad_h[2] + dkinetic_dTs) / ht
            dR2_dPs = (grad_h[1] + dkinetic_dPs) / ht

            # Solve 2x2 linear system: J @ [dTs, dPs]^T = -[R1, R2]^T
            det = dR1_dTs * dR2_dPs - dR1_dPs * dR2_dTs
            if abs(det) < 1e-30:
                # Jacobian is singular
                break

            # Cramer's rule
            dTs = (-R1 * dR2_dPs + R2 * dR1_dPs) / det
            dPs = (-R2 * dR1_dTs + R1 * dR2_dTs) / det

            # Apply Newton update with damping to stay in valid range
            alpha = 1.0
            Ts_new = Ts + alpha * dTs
            Ps_new = Ps + alpha * dPs

            # Clamp to valid ranges
            Ts_new = max(160.0, min(2400.0, Ts_new))
            Ps_new = max(Pt_si * 0.001, min(Pt_si * 0.9999, Ps_new))

            # Check for stagnation
            if abs(Ts_new - Ts) < 1e-12 and abs(Ps_new - Ps) < 1e-12:
                Ts, Ps = Ts_new, Ps_new
                break

            Ts, Ps = Ts_new, Ps_new

        # If not converged and haven't retried, reset guess flag and retry once
        if not converged and not _retry:
            self._needs_guess_static_MN = True
            self._solver_stats['static_MN']['retries'] += 1
            return self._static_from_MN_si(Tt_si, Pt_si, MN, W_si, FAR, _retry=True)

        # Cache the converged solution
        self._cache_static_MN = (Ts, Ps)

        # Reuse hs, gamma_s, R_s from last Newton iteration (already computed above)
        # Only look up Cp and Cv which weren't needed for the Newton solve
        Cp_s = self._lookup_si('Cp', Ts, Ps, FAR)
        Cv_s = self._lookup_si('Cv', Ts, Ps, FAR)

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
        max_iter = 20
        tol = 1e-10

        # Track solver stats
        if not _retry:
            self._solver_stats['T_from_SP']['calls'] += 1

        # Apply initial guess if needed, otherwise use cached value
        if self._needs_guess_T_from_SP:
            # Empirical initial guess: mid-range temperature
            T = 800.0
            self._needs_guess_T_from_SP = False
            self._solver_stats['T_from_SP']['guesses'] += 1
        else:
            T = self._cache_T_from_SP

        converged = False
        for _ in range(max_iter):
            x = np.array([FAR, P_si, T])

            # Get S and dS/dT in single call (avoids redundant cell lookup)
            S_arr, grad_S_2d = self._interps['S'].interpolate(x, compute_derivative=True)
            S = S_arr[0]
            dS_dT = grad_S_2d[0, 2]  # (dS/dFAR, dS/dP, dS/dT)

            # Residual
            residual = S - S_target_si

            # Check convergence (relative tolerance)
            if abs(residual) < tol * abs(S_target_si) or abs(residual) < 1e-6:
                converged = True
                break

            # Newton update
            if abs(dS_dT) < 1e-30:
                break  # Derivative too small

            dT = -residual / dS_dT

            # Update with bounds enforcement
            T_new = T + dT
            T_new = max(160.0, min(2400.0, T_new))

            # Check for stagnation
            if abs(T_new - T) < 1e-12:
                T = T_new
                break

            T = T_new

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

        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        area_si = area * self._area_to_si
        W_si = W * self._W_to_si

        # Try Newton's method first (much faster when it converges)
        MN, props_si, converged = self._newton_solve_MN(
            Tt_si, Pt_si, area_si, W_si, FAR, MN_guess, subsonic
        )

        if converged:
            return self._convert_static_props_from_si(props_si)

        # Fall back to brentq for robustness
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

    def _newton_solve_MN(self, Tt_si, Pt_si, area_si, W_si, FAR, MN_guess, subsonic,
                         max_iter=10, tol=1e-10):
        """
        Solve for MN using Newton's method with analytical derivative.

        Returns
        -------
        MN : float
            Converged Mach number
        props_si : StaticProps
            Static properties at converged MN
        converged : bool
            True if Newton converged
        """
        # Use cached solution as initial guess if available
        cache_key = (round(Tt_si, 2), round(Pt_si, 0), round(W_si, 4), round(FAR, 4), subsonic)
        if cache_key in self._last_MN_solution:
            MN = self._last_MN_solution[cache_key]
        else:
            MN = MN_guess if MN_guess > 0.01 else (0.4 if subsonic else 1.5)

        # Bounds for subsonic/supersonic
        MN_min = 0.01 if subsonic else 1.001
        MN_max = 0.999 if subsonic else 5.0

        props_si = None
        for _ in range(max_iter):
            # Compute area and its derivative w.r.t. MN
            area_computed, darea_dMN, props_si = self._area_and_derivative_si(
                Tt_si, Pt_si, MN, W_si, FAR
            )

            residual = area_computed - area_si

            # Check convergence
            if abs(residual) < tol * area_si:
                self._last_MN_solution[cache_key] = MN
                return MN, props_si, True

            # Newton update
            if abs(darea_dMN) < 1e-20:
                return MN, props_si, False  # Derivative too small

            MN_new = MN - residual / darea_dMN

            # Clamp to bounds
            MN_new = max(MN_min, min(MN_max, MN_new))

            # Check for stagnation
            if abs(MN_new - MN) < 1e-14:
                self._last_MN_solution[cache_key] = MN_new
                return MN_new, props_si, True

            MN = MN_new

        return MN, props_si, False  # Did not converge

    def _area_and_derivative_si(self, Tt_si, Pt_si, MN, W_si, FAR):
        """
        Compute area and d(area)/d(MN) at given conditions.

        Uses the new physics (_static_from_MN_si with energy/entropy constraints)
        and computes d(area)/d(MN) numerically via finite difference.

        Returns area, darea_dMN, and full static props (to avoid recomputation).
        """
        # Compute static properties at current MN
        props_si = self._static_from_MN_si(Tt_si, Pt_si, MN, W_si, FAR)
        area = props_si.area

        # Compute d(area)/d(MN) via finite difference
        eps = 1e-6
        MN_pert = MN + eps
        props_pert = self._static_from_MN_si(Tt_si, Pt_si, MN_pert, W_si, FAR)
        darea_dMN = (props_pert.area - area) / eps

        return area, darea_dMN, props_si

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

        # Get total properties and their gradients (combined call)
        x_tot = np.array([FAR, Pt_si, Tt_si])
        ht_arr, grad_ht_2d = self._interps['h'].interpolate(x_tot, compute_derivative=True)
        St_arr, grad_St_2d = self._interps['S'].interpolate(x_tot, compute_derivative=True)
        ht_si = ht_arr[0]
        S_total = St_arr[0]
        grad_ht_tot = grad_ht_2d[0]  # (dh/dFAR, dh/dP, dh/dT)
        grad_S_tot = grad_St_2d[0]   # (dS/dFAR, dS/dP, dS/dT)

        # Get static properties and their gradients at (Ts, Ps, FAR) (combined calls)
        x_stat = np.array([FAR, Ps_si, Ts_si])
        hs_arr, grad_hs_2d = self._interps['h'].interpolate(x_stat, compute_derivative=True)
        Ss_arr, grad_Ss_2d = self._interps['S'].interpolate(x_stat, compute_derivative=True)
        gams_arr, grad_gams_2d = self._interps['gamma'].interpolate(x_stat, compute_derivative=True)
        Rs_arr, grad_Rs_2d = self._interps['R'].interpolate(x_stat, compute_derivative=True)
        Cps_arr, grad_Cps_2d = self._interps['Cp'].interpolate(x_stat, compute_derivative=True)
        Cvs_arr, grad_Cvs_2d = self._interps['Cv'].interpolate(x_stat, compute_derivative=True)

        hs_si = hs_arr[0]
        gamma_s = gams_arr[0]
        R_s = Rs_arr[0]

        grad_hs = grad_hs_2d[0]      # (dhs/dFAR, dhs/dPs, dhs/dTs)
        grad_Ss = grad_Ss_2d[0]      # (dSs/dFAR, dSs/dPs, dSs/dTs)
        grad_gams = grad_gams_2d[0]
        grad_Rs = grad_Rs_2d[0]
        grad_Cps = grad_Cps_2d[0]
        grad_Cvs = grad_Cvs_2d[0]

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

        # Get total properties in SI (J/kg for enthalpy)
        ht_si = self._lookup_si('h', Tt_si, Pt_si, FAR)
        S_total = self._lookup_si('S', Tt_si, Pt_si, FAR)

        # Find Ts from entropy constraint: S(Ts, Ps) = S_total
        Ts_si = self._T_from_SP_si(S_total, Ps_si, FAR)

        # Full static properties at (Ts, Ps) in SI
        hs_si = self._lookup_si('h', Ts_si, Ps_si, FAR)
        gam_s = self._lookup_si('gamma', Ts_si, Ps_si, FAR)
        Cp_s = self._lookup_si('Cp', Ts_si, Ps_si, FAR)
        Cv_s = self._lookup_si('Cv', Ts_si, Ps_si, FAR)
        R_s = self._lookup_si('R', Ts_si, Ps_si, FAR)

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
