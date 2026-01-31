"""
Tabular thermodynamic property calculations using interpolation.
"""

import numpy as np

from ..base import ThermoInterface, TotalProps, StaticProps


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

    def __init__(self, FAR=0.0, spec=None, input_units='SI'):
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
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self._interps[prop] = InterpND(
                method='3D-slinear',
                points=points,
                values=spec[prop],
                extrapolate=True
            )

    def _lookup_si(self, prop, T_si, P_si, FAR=None):
        """Internal lookup function in SI units."""
        if FAR is None:
            FAR = self.FAR
        x = np.array([FAR, float(P_si), float(T_si)])
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
        if FAR is None:
            FAR = self.FAR

        # Convert inputs to SI for lookup
        T_si = self._convert_T_to_si(T)
        P_si = P * self._P_to_si

        x = np.array([FAR, float(P_si), float(T_si)])

        # Store the linearization point
        self._lin_T = T
        self._lin_P = P
        self._lin_FAR = FAR
        self._lin_T_si = T_si
        self._lin_P_si = P_si

        # Compute gradients for each property (in SI units)
        # gradients are (dProp/dFAR, dProp/dP, dProp/dT)
        self._gradients_si = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self._gradients_si[prop] = self._interps[prop].gradient(x)

        # If props provided from forward pass, use those; otherwise lookup
        if props is not None:
            return props
        else:
            props_si = {}
            for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
                props_si[prop] = self._interps[prop].interpolate(x)[0]
            return self._convert_total_props_from_si(TotalProps(
                h=props_si['h'], S=props_si['S'], gamma=props_si['gamma'],
                Cp=props_si['Cp'], Cv=props_si['Cv'], rho=props_si['rho'], R=props_si['R']
            ))

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
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            # gradients are (dProp/dFAR, dProp/dP, dProp/dT) in SI
            grad = self._gradients_si[prop]

            if prop == 'gamma':
                out_factor = 1.0
            elif prop in ('S', 'Cp', 'Cv', 'R'):
                out_factor = self._S_from_si
            elif prop == 'h':
                out_factor = self._h_from_si
            elif prop == 'rho':
                out_factor = self._rho_from_si

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

                if prop == 'gamma':
                    out_factor = 1.0
                elif prop in ('S', 'Cp', 'Cv', 'R'):
                    out_factor = self._S_from_si
                elif prop == 'h':
                    out_factor = self._h_from_si
                elif prop == 'rho':
                    out_factor = self._rho_from_si

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
        if FAR is None:
            FAR = self.FAR

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
        from scipy.optimize import brentq

        if FAR is None:
            FAR = self.FAR

        # Convert inputs to SI
        h_si = h_target * self._h_to_si
        P_si = P * self._P_to_si

        def residual(T_si):
            return float(self._lookup_si('h', T_si, P_si, FAR)) - float(h_si)

        T_min, T_max = 150.0, 2500.0
        T_si = brentq(residual, T_min, T_max, xtol=1e-10)

        # Convert output from SI
        return self._convert_T_from_si(T_si)

    def T_from_SP(self, S_target, P, FAR=None):
        """Solve for temperature given entropy and pressure.

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
        from scipy.optimize import brentq

        if FAR is None:
            FAR = self.FAR

        # Convert inputs to SI
        S_si = S_target * self._S_to_si
        P_si = P * self._P_to_si

        def residual(T_si):
            return float(self._lookup_si('S', T_si, P_si, FAR)) - float(S_si)

        T_min, T_max = 150.0, 2500.0
        T_si = brentq(residual, T_min, T_max, xtol=1e-10)

        return self._convert_T_from_si(T_si)

    # =========================================================================
    # Static property calculations
    # =========================================================================

    def _static_from_MN_si(self, Tt_si, Pt_si, MN, W_si, FAR=None):
        """Compute static properties in SI units.

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
        """
        if FAR is None:
            FAR = self.FAR

        # Get gamma and R at total conditions (in SI)
        gam = self._lookup_si('gamma', Tt_si, Pt_si, FAR)
        R_gas = self._lookup_si('R', Tt_si, Pt_si, FAR)

        # Isentropic relations
        temp_ratio = 1.0 / (1.0 + (gam - 1.0) / 2.0 * MN**2)
        Ts = Tt_si * temp_ratio
        Ps = Pt_si * temp_ratio**(gam / (gam - 1.0))

        # Full static properties at static T and P
        hs = self._lookup_si('h', Ts, Ps, FAR)
        S_s = self._lookup_si('S', Ts, Ps, FAR)
        gam_s = self._lookup_si('gamma', Ts, Ps, FAR)
        Cp_s = self._lookup_si('Cp', Ts, Ps, FAR)
        Cv_s = self._lookup_si('Cv', Ts, Ps, FAR)
        R_s = self._lookup_si('R', Ts, Ps, FAR)

        # Speed of sound and velocity (use static properties)
        Vsonic = np.sqrt(gam_s * R_s * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps / (R_s * Ts)

        # Area from continuity
        area = W_si / (rhos * V) if V > 0 else np.inf

        return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                          MN=MN, V=V, Vsonic=Vsonic, area=area,
                          gamma=gam_s, Cp=Cp_s, Cv=Cv_s, S=S_s, R=R_s)

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
        if FAR is None:
            FAR = self.FAR

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
        if FAR is None:
            FAR = self.FAR

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
        from scipy.optimize import brentq

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

        Returns area, darea_dMN, and full static props (to avoid recomputation).
        """
        # Get gamma at total conditions
        gam = self._lookup_si('gamma', Tt_si, Pt_si, FAR)

        # Isentropic relations
        MN2 = MN ** 2
        gm1 = gam - 1.0
        denom = 1.0 + gm1 / 2.0 * MN2
        temp_ratio = 1.0 / denom
        exp = gam / gm1

        Ts = Tt_si * temp_ratio
        Ps = Pt_si * temp_ratio ** exp

        # Static properties
        gam_s = self._lookup_si('gamma', Ts, Ps, FAR)
        R_s = self._lookup_si('R', Ts, Ps, FAR)

        # Flow properties
        Vsonic = np.sqrt(gam_s * R_s * Ts)
        V = MN * Vsonic
        rhos = Ps / (R_s * Ts)
        area = W_si / (rhos * V) if V > 0 else np.inf

        # Compute d(area)/d(MN) analytically
        # area = W / (rhos * V) = W * R_s * Ts / (Ps * V)
        # V = MN * Vsonic
        # Need: d(area)/d(MN) = d(area)/d(Ts)*d(Ts)/d(MN) + d(area)/d(Ps)*d(Ps)/d(MN) + d(area)/d(V)*d(V)/d(MN)

        # Derivatives of temp_ratio w.r.t. MN
        dtr_dMN = -gm1 * MN / (denom ** 2)

        # d(Ts)/d(MN) = Tt * d(temp_ratio)/d(MN)
        dTs_dMN = Tt_si * dtr_dMN

        # d(Ps)/d(MN) = Pt * exp * temp_ratio^(exp-1) * d(temp_ratio)/d(MN)
        dPs_dMN = Pt_si * exp * temp_ratio ** (exp - 1) * dtr_dMN

        # d(Vsonic)/d(MN) ≈ 0.5 * Vsonic / Ts * dTs_dMN (ignoring gamma_s, R_s dependence on Ts, Ps)
        # More accurate: Vsonic = sqrt(gam_s * R_s * Ts)
        # d(Vsonic)/d(MN) = 0.5/Vsonic * (gam_s * R_s * dTs_dMN) = 0.5 * Vsonic / Ts * dTs_dMN
        dVsonic_dMN = 0.5 * Vsonic / Ts * dTs_dMN if Ts > 0 else 0.0

        # d(V)/d(MN) = Vsonic + MN * d(Vsonic)/d(MN)
        dV_dMN = Vsonic + MN * dVsonic_dMN

        # d(rhos)/d(MN) = d(Ps/R_s/Ts)/d(MN) ≈ (dPs_dMN - Ps/Ts*dTs_dMN) / (R_s * Ts)
        # Ignoring R_s dependence on static conditions
        drhos_dMN = (dPs_dMN / (R_s * Ts) - Ps * dTs_dMN / (R_s * Ts ** 2))

        # d(area)/d(MN) = -W / (rhos * V)^2 * (drhos_dMN * V + rhos * dV_dMN)
        #               = -area / (rhos * V) * (drhos_dMN * V + rhos * dV_dMN)
        if V > 0 and rhos > 0:
            darea_dMN = -W_si / (rhos ** 2 * V ** 2) * (drhos_dMN * V + rhos * dV_dMN)
        else:
            darea_dMN = 0.0

        # Build full props for return (avoid recomputing later)
        hs = self._lookup_si('h', Ts, Ps, FAR)
        S_s = self._lookup_si('S', Ts, Ps, FAR)
        Cp_s = self._lookup_si('Cp', Ts, Ps, FAR)
        Cv_s = self._lookup_si('Cv', Ts, Ps, FAR)

        props_si = StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                               MN=MN, V=V, Vsonic=Vsonic, area=area,
                               gamma=gam_s, Cp=Cp_s, Cv=Cv_s, S=S_s, R=R_s)

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
        Compute static properties and cache gradients for static_from_MN.

        This combines the forward pass and linearization into one call,
        avoiding duplicate lookups when both are needed.

        The static_from_MN calculation is explicit (no solver), so we can
        differentiate directly using the chain rule through:
        1. Isentropic relations: Ts(Tt, MN, gamma), Ps(Pt, MN, gamma)
        2. Property lookups at (Ts, Ps, FAR)
        3. Flow relations: V, Vsonic, rhos, area

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
            skips the static property value lookups (only computes gradients).

        Returns
        -------
        StaticProps
            Named tuple with static properties in input units
        """
        import time
        t_start = time.perf_counter()
        p = self._profile_static_MN

        if FAR is None:
            FAR = self.FAR

        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        W_si = W * self._W_to_si

        # Get gamma at total conditions
        t0 = time.perf_counter()
        gam = self._lookup_si('gamma', Tt_si, Pt_si, FAR)
        R_tot = self._lookup_si('R', Tt_si, Pt_si, FAR)
        p['lookup_total'] += time.perf_counter() - t0

        # Linearize at total conditions for dgamma/dTt, dgamma/dPt, dgamma/dFAR
        t0 = time.perf_counter()
        x_tot = np.array([FAR, float(Pt_si), float(Tt_si)])
        grad_gam_tot = self._interps['gamma'].gradient(x_tot)  # (dg/dFAR, dg/dP, dg/dT)
        grad_R_tot = self._interps['R'].gradient(x_tot)
        p['grad_total'] += time.perf_counter() - t0

        dgam_dTt = grad_gam_tot[2] * self._T_to_si  # Convert to input units
        dgam_dPt = grad_gam_tot[1] * self._P_to_si
        dgam_dFAR = grad_gam_tot[0]  # FAR is dimensionless

        # Isentropic relations
        t0 = time.perf_counter()
        MN2 = MN ** 2
        gm1 = gam - 1.0
        gm1_half = gm1 / 2.0
        denom = 1.0 + gm1_half * MN2
        temp_ratio = 1.0 / denom

        Ts_si = Tt_si * temp_ratio
        exp = gam / gm1
        Ps_si = Pt_si * temp_ratio ** exp

        # Derivatives of temp_ratio w.r.t. inputs
        # temp_ratio = 1 / (1 + (gam-1)/2 * MN^2)
        # d(temp_ratio)/dMN = -(gam-1) * MN / denom^2
        # d(temp_ratio)/dgam = -MN^2 / (2 * denom^2)
        dtr_dMN = -gm1 * MN / (denom ** 2)
        dtr_dgam = -MN2 / (2.0 * denom ** 2)

        # Derivatives of Ts w.r.t. inputs (Ts = Tt * temp_ratio)
        # All derivatives should be in input units (e.g., degR/degR, degR/psi, degR/MN)
        dTs_dTt = temp_ratio + Tt_si * dtr_dgam * dgam_dTt / self._T_to_si
        dTs_dPt = Tt_si * dtr_dgam * dgam_dPt / self._P_to_si
        dTs_dMN = Tt_si * dtr_dMN * self._T_from_si  # Convert T_si to T_input
        dTs_dFAR = Tt_si * dtr_dgam * dgam_dFAR * self._T_from_si  # Through gamma dependency

        # Derivatives of Ps w.r.t. inputs
        # Ps = Pt * temp_ratio^exp, exp = gam/(gam-1)
        # d(exp)/dgam = -1/(gam-1)^2
        dexp_dgam = -1.0 / (gm1 ** 2)
        ln_tr = np.log(temp_ratio) if temp_ratio > 0 else 0.0

        # d(Ps)/dPt = temp_ratio^exp + Pt * exp * temp_ratio^(exp-1) * dtr/dgam * dgam/dPt
        #           + Pt * temp_ratio^exp * ln(temp_ratio) * dexp/dgam * dgam/dPt
        dPs_dPt_base = temp_ratio ** exp
        dPs_dPt = dPs_dPt_base + Pt_si * (
            exp * temp_ratio ** (exp - 1) * dtr_dgam * dgam_dPt +
            temp_ratio ** exp * ln_tr * dexp_dgam * dgam_dPt
        ) / self._P_to_si

        dPs_dTt = Pt_si * (
            exp * temp_ratio ** (exp - 1) * dtr_dgam * dgam_dTt +
            temp_ratio ** exp * ln_tr * dexp_dgam * dgam_dTt
        ) * self._P_from_si  # Convert P_si to P_input (was incorrectly / _T_to_si)

        dPs_dMN = Pt_si * exp * temp_ratio ** (exp - 1) * dtr_dMN * self._P_from_si  # Convert P_si to P_input

        # d(Ps)/dFAR through gamma dependency
        dPs_dFAR = Pt_si * (
            exp * temp_ratio ** (exp - 1) * dtr_dgam * dgam_dFAR +
            temp_ratio ** exp * ln_tr * dexp_dgam * dgam_dFAR
        ) * self._P_from_si
        p['isentropic'] += time.perf_counter() - t0

        # Get static properties and their gradients at (Ts, Ps, FAR)
        t0 = time.perf_counter()
        x_stat = np.array([FAR, float(Ps_si), float(Ts_si)])
        grad_hs = self._interps['h'].gradient(x_stat)
        grad_Ss = self._interps['S'].gradient(x_stat)
        grad_gams = self._interps['gamma'].gradient(x_stat)
        grad_Cps = self._interps['Cp'].gradient(x_stat)
        grad_Cvs = self._interps['Cv'].gradient(x_stat)
        grad_Rs = self._interps['R'].gradient(x_stat)
        grad_rhos = self._interps['rho'].gradient(x_stat)
        p['grad_static'] += time.perf_counter() - t0

        # Static property values - use sprops if provided, else look up
        t0 = time.perf_counter()
        if sprops is not None:
            # Use pre-computed values from forward pass (convert to SI)
            hs_si = sprops.hs * self._h_to_si
            Ss_si = sprops.S * self._S_to_si
            gam_s = sprops.gamma  # dimensionless
            Cp_s = sprops.Cp * self._S_to_si
            Cv_s = sprops.Cv * self._S_to_si
            R_s = sprops.R * self._S_to_si
        else:
            # Look up from tables
            hs_si = self._lookup_si('h', Ts_si, Ps_si, FAR)
            Ss_si = self._lookup_si('S', Ts_si, Ps_si, FAR)
            gam_s = self._lookup_si('gamma', Ts_si, Ps_si, FAR)
            Cp_s = self._lookup_si('Cp', Ts_si, Ps_si, FAR)
            Cv_s = self._lookup_si('Cv', Ts_si, Ps_si, FAR)
            R_s = self._lookup_si('R', Ts_si, Ps_si, FAR)
        p['lookup_total'] += time.perf_counter() - t0

        # Flow calculations
        Vsonic_si = np.sqrt(gam_s * R_s * Ts_si)
        V_si = MN * Vsonic_si
        rhos_si = Ps_si / (R_s * Ts_si)
        area_si = W_si / (rhos_si * V_si) if V_si > 0 else np.inf

        # Store cached values needed for JVP (kept for backward compatibility)
        self._static_MN_cache = {
            # Input values (SI)
            'Tt_si': Tt_si, 'Pt_si': Pt_si, 'MN': MN, 'W_si': W_si, 'FAR': FAR,
            # Intermediate values
            'gam': gam, 'temp_ratio': temp_ratio, 'exp': exp,
            'Ts_si': Ts_si, 'Ps_si': Ps_si,
            'gam_s': gam_s, 'R_s': R_s,
            'Vsonic_si': Vsonic_si, 'V_si': V_si, 'rhos_si': rhos_si, 'area_si': area_si,
            # Gradients of Ts, Ps w.r.t. inputs (in input units)
            'dTs_dTt': dTs_dTt, 'dTs_dPt': dTs_dPt, 'dTs_dMN': dTs_dMN, 'dTs_dFAR': dTs_dFAR,
            'dPs_dTt': dPs_dTt, 'dPs_dPt': dPs_dPt, 'dPs_dMN': dPs_dMN, 'dPs_dFAR': dPs_dFAR,
            # Gradients of static properties w.r.t. (FAR, Ps, Ts) in SI
            'grad_hs': grad_hs, 'grad_Ss': grad_Ss, 'grad_gams': grad_gams,
            'grad_Cps': grad_Cps, 'grad_Cvs': grad_Cvs, 'grad_Rs': grad_Rs,
            'grad_rhos': grad_rhos,
        }

        # Compute full Jacobian matrix directly (13 outputs x 5 inputs)
        # This avoids calling jvp_static_MN 5 times with basis vectors
        t0 = time.perf_counter()
        self._compute_full_jacobian_static_MN()
        p['jacobian'] += time.perf_counter() - t0

        p['calls'] += 1
        p['total_time'] += time.perf_counter() - t_start

        # Return static properties converted to input units
        props_si = StaticProps(
            Ts=Ts_si, Ps=Ps_si, hs=hs_si, rhos=rhos_si,
            MN=MN, V=V_si, Vsonic=Vsonic_si, area=area_si,
            gamma=gam_s, Cp=Cp_s, Cv=Cv_s, S=Ss_si, R=R_s
        )
        return self._convert_static_props_from_si(props_si)

    def _compute_full_jacobian_static_MN(self):
        """Compute and cache the full Jacobian for static_from_MN."""
        c = self._static_MN_cache

        # Build intermediate Jacobian: d[Ts_si, Ps_si]/d[Tt, Pt, MN, W, FAR]
        # Shape: (2, 5) - but we compute in input units then convert
        dTs_d = np.array([c['dTs_dTt'], c['dTs_dPt'], c['dTs_dMN'], 0.0, c['dTs_dFAR']]) * self._T_to_si
        dPs_d = np.array([c['dPs_dTt'], c['dPs_dPt'], c['dPs_dMN'], 0.0, c['dPs_dFAR']]) * self._P_to_si

        # Property gradients: grad[i] = (dProp/dFAR, dProp/dPs, dProp/dTs) in SI
        # Chain rule: dProp/dX = grad[2]*dTs/dX + grad[1]*dPs/dX + grad[0]*dFAR/dX
        def chain_rule_vec(grad):
            # Returns array of 5 derivatives w.r.t. [Tt, Pt, MN, W, FAR]
            FAR_derivs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # d(FAR)/d[Tt,Pt,MN,W,FAR]
            return grad[2] * dTs_d + grad[1] * dPs_d + grad[0] * FAR_derivs

        # Compute derivatives for tabular properties
        dhs_d = chain_rule_vec(c['grad_hs']) * self._h_from_si
        dSs_d = chain_rule_vec(c['grad_Ss']) * self._S_from_si
        dgams_d = chain_rule_vec(c['grad_gams'])
        dCps_d = chain_rule_vec(c['grad_Cps']) * self._S_from_si
        dCvs_d = chain_rule_vec(c['grad_Cvs']) * self._S_from_si
        dRs_d = chain_rule_vec(c['grad_Rs']) * self._S_from_si

        # Flow variable derivatives (more complex chain rules)
        gam_s, R_s, Ts_si = c['gam_s'], c['R_s'], c['Ts_si']
        Vsonic = c['Vsonic_si']
        Ps_si, rhos_si = c['Ps_si'], c['rhos_si']
        W_si, V_si, area_si = c['W_si'], c['V_si'], c['area_si']
        MN = c['MN']

        # Vsonic = sqrt(gam_s * R_s * Ts)
        # dVsonic = (R_s*Ts*dgam_s + gam_s*Ts*dR_s + gam_s*R_s*dTs) / (2*Vsonic)
        if Vsonic > 0:
            dgams_d_si = chain_rule_vec(c['grad_gams'])  # dimensionless
            dRs_d_si = chain_rule_vec(c['grad_Rs'])  # SI
            dVsonic_d_si = (R_s * Ts_si * dgams_d_si + gam_s * Ts_si * dRs_d_si +
                           gam_s * R_s * dTs_d) / (2 * Vsonic)
        else:
            dVsonic_d_si = np.zeros(5)
        dVsonic_d = dVsonic_d_si * self._V_from_si

        # V = MN * Vsonic
        # dV/d[Tt,Pt,MN,W,FAR] = dMN/d* * Vsonic + MN * dVsonic/d*
        dMN_d = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # MN is direct input
        dV_d_si = dMN_d * Vsonic + MN * dVsonic_d_si
        dV_d = dV_d_si * self._V_from_si

        # rhos = Ps / (R_s * Ts)
        # drhos = dPs/(R*T) - Ps*dR/(R^2*T) - Ps*dT/(R*T^2)
        dRs_d_si = chain_rule_vec(c['grad_Rs'])
        drhos_d_si = (dPs_d / (R_s * Ts_si) -
                      Ps_si * dRs_d_si / (R_s ** 2 * Ts_si) -
                      Ps_si * dTs_d / (R_s * Ts_si ** 2))
        drhos_d = drhos_d_si * self._rho_from_si

        # area = W / (rhos * V)
        # darea = dW/(rhos*V) - W*drhos/(rhos^2*V) - W*dV/(rhos*V^2)
        dW_d_si = np.array([0.0, 0.0, 0.0, self._W_to_si, 0.0])  # W is direct input
        if V_si > 0 and rhos_si > 0:
            darea_d_si = (dW_d_si / (rhos_si * V_si) -
                          W_si * drhos_d_si / (rhos_si ** 2 * V_si) -
                          W_si * dV_d_si / (rhos_si * V_si ** 2))
        else:
            darea_d_si = np.zeros(5)
        darea_d = darea_d_si * self._area_from_si

        # Ts, Ps derivatives in output units
        dTs_d_out = dTs_d * self._T_from_si
        dPs_d_out = dPs_d * self._P_from_si

        # Store full Jacobian as dict of arrays (each array is 5 derivatives)
        self._jacobian_static_MN = {
            'Ts': dTs_d_out,
            'Ps': dPs_d_out,
            'hs': dhs_d,
            'rhos': drhos_d,
            'MN': dMN_d,
            'V': dV_d,
            'Vsonic': dVsonic_d,
            'area': darea_d,
            'gamma': dgams_d,
            'Cp': dCps_d,
            'Cv': dCvs_d,
            'S': dSs_d,
            'R': dRs_d,
        }

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

    def jacobian_static_MN(self):
        """
        Return the full Jacobian matrix for static_from_MN.

        Returns
        -------
        dict
            Dictionary mapping property names to arrays of 5 partial derivatives
            [d/dTt, d/dPt, d/dMN, d/dW, d/dFAR]
        """
        if not hasattr(self, '_static_MN_cache'):
            raise RuntimeError("Must call linearize_static_MN() before jacobian_static_MN()")

        # Compute all five JVPs efficiently
        jvp_Tt = self.jvp_static_MN(1.0, 0.0, 0.0, 0.0, 0.0)
        jvp_Pt = self.jvp_static_MN(0.0, 1.0, 0.0, 0.0, 0.0)
        jvp_MN = self.jvp_static_MN(0.0, 0.0, 1.0, 0.0, 0.0)
        jvp_W = self.jvp_static_MN(0.0, 0.0, 0.0, 1.0, 0.0)
        jvp_FAR = self.jvp_static_MN(0.0, 0.0, 0.0, 0.0, 1.0)

        result = {}
        for prop in ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                     'gamma', 'Cp', 'Cv', 'S', 'R']:
            result[prop] = np.array([jvp_Tt[prop], jvp_Pt[prop], jvp_MN[prop],
                                     jvp_W[prop], jvp_FAR[prop]])

        return result

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
        for prop in ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area', 'gamma', 'Cp', 'Cv', 'S', 'R']:
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
        if FAR is None:
            FAR = self.FAR

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
        dMN_d = np.array([dMN_dTt, dMN_dPt, dMN_darea, dMN_dW, dMN_dFAR])

        self._jacobian_static_area = {}
        for prop in ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area', 'gamma', 'Cp', 'Cv', 'S', 'R']:
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
        for prop in ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area', 'gamma', 'Cp', 'Cv', 'S', 'R']:
            result[prop] = float(np.dot(jac[prop], tangent))

        return result

    def static_from_Ps(self, Tt, Pt, Ps, W, FAR=None):
        """Compute static properties from total conditions and static pressure.

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
        if FAR is None:
            FAR = self.FAR

        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        Ps_si = Ps * self._P_to_si
        W_si = W * self._W_to_si

        # Get gamma at total conditions for isentropic relation
        gam = self._lookup_si('gamma', Tt_si, Pt_si, FAR)

        # From isentropic relation
        pressure_ratio = Ps_si / Pt_si
        Ts = Tt_si * pressure_ratio**((gam - 1.0) / gam)

        # Full static properties at static T and P
        hs = self._lookup_si('h', Ts, Ps_si, FAR)
        S_s = self._lookup_si('S', Ts, Ps_si, FAR)
        gam_s = self._lookup_si('gamma', Ts, Ps_si, FAR)
        Cp_s = self._lookup_si('Cp', Ts, Ps_si, FAR)
        Cv_s = self._lookup_si('Cv', Ts, Ps_si, FAR)
        R_s = self._lookup_si('R', Ts, Ps_si, FAR)

        # Compute Mach number from temperature ratio
        temp_ratio = Ts / Tt_si
        MN_sq = 2.0 / (gam - 1.0) * (1.0 / temp_ratio - 1.0)
        MN = np.sqrt(max(0.0, MN_sq))

        # Speed of sound and velocity (use static properties)
        Vsonic = np.sqrt(gam_s * R_s * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps_si / (R_s * Ts)

        # Area from continuity
        area = W_si / (rhos * V) if V > 0 else np.inf

        props_si = StaticProps(Ts=Ts, Ps=Ps_si, hs=hs, rhos=rhos,
                              MN=MN, V=V, Vsonic=Vsonic, area=area,
                              gamma=gam_s, Cp=Cp_s, Cv=Cv_s, S=S_s, R=R_s)

        return self._convert_static_props_from_si(props_si)
