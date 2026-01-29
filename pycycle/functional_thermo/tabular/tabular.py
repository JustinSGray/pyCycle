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

        self._interps = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self._interps[prop] = InterpND(
                method='slinear',
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

    def linearize(self, T, P, FAR=None):
        """
        Compute and cache property gradients at the given state.

        Parameters
        ----------
        T : float
            Temperature (in input units)
        P : float
            Pressure (in input units)
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.
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

        # Compute and cache gradients for each property (in SI units)
        # gradients are (dProp/dFAR, dProp/dP, dProp/dT)
        self._gradients_si = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self._gradients_si[prop] = self._interps[prop].gradient(x)

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

    def static_from_area(self, Tt, Pt, area, W, MN_guess=0.5, subsonic=True, FAR=None):
        """Compute static properties from total conditions and flow area.

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
            Initial guess for Mach number (not used, kept for API compatibility)
        subsonic : bool, optional
            If True, find subsonic solution; if False, find supersonic
        FAR : float, optional
            Fuel-to-air ratio. If None, uses the instance's FAR.

        Returns
        -------
        StaticProps
            Named tuple with static properties in input units
        """
        from scipy.optimize import brentq

        if FAR is None:
            FAR = self.FAR

        # Convert inputs to SI
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

    # =========================================================================
    # Analytical derivatives for static properties
    # =========================================================================

    def linearize_static_MN(self, Tt, Pt, MN, W, FAR=None):
        """
        Compute and cache gradients for static_from_MN at the given state.

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
        """
        if FAR is None:
            FAR = self.FAR

        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        W_si = W * self._W_to_si

        # Get gamma at total conditions
        gam = self._lookup_si('gamma', Tt_si, Pt_si, FAR)
        R_tot = self._lookup_si('R', Tt_si, Pt_si, FAR)

        # Linearize at total conditions for dgamma/dTt, dgamma/dPt, dgamma/dFAR
        x_tot = np.array([FAR, float(Pt_si), float(Tt_si)])
        grad_gam_tot = self._interps['gamma'].gradient(x_tot)  # (dg/dFAR, dg/dP, dg/dT)
        grad_R_tot = self._interps['R'].gradient(x_tot)

        dgam_dTt = grad_gam_tot[2] * self._T_to_si  # Convert to input units
        dgam_dPt = grad_gam_tot[1] * self._P_to_si
        dgam_dFAR = grad_gam_tot[0]  # FAR is dimensionless

        # Isentropic relations
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

        # Get static properties and their gradients at (Ts, Ps, FAR)
        x_stat = np.array([FAR, float(Ps_si), float(Ts_si)])
        grad_hs = self._interps['h'].gradient(x_stat)
        grad_Ss = self._interps['S'].gradient(x_stat)
        grad_gams = self._interps['gamma'].gradient(x_stat)
        grad_Cps = self._interps['Cp'].gradient(x_stat)
        grad_Cvs = self._interps['Cv'].gradient(x_stat)
        grad_Rs = self._interps['R'].gradient(x_stat)
        grad_rhos = self._interps['rho'].gradient(x_stat)

        # Static property values
        hs_si = self._lookup_si('h', Ts_si, Ps_si, FAR)
        gam_s = self._lookup_si('gamma', Ts_si, Ps_si, FAR)
        R_s = self._lookup_si('R', Ts_si, Ps_si, FAR)

        # Flow calculations
        Vsonic_si = np.sqrt(gam_s * R_s * Ts_si)
        V_si = MN * Vsonic_si
        rhos_si = Ps_si / (R_s * Ts_si)
        area_si = W_si / (rhos_si * V_si) if V_si > 0 else np.inf

        # Store all the cached values needed for JVP
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
        Compute JVP for static_from_MN using cached linearization.

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
        if not hasattr(self, '_static_MN_cache'):
            raise RuntimeError("Must call linearize_static_MN() before jvp_static_MN()")

        c = self._static_MN_cache

        # Derivatives of Ts, Ps w.r.t. inputs (including FAR)
        Ts_dot_si = (c['dTs_dTt'] * Tt_dot + c['dTs_dPt'] * Pt_dot +
                     c['dTs_dMN'] * MN_dot + c['dTs_dFAR'] * FAR_dot) * self._T_to_si
        Ps_dot_si = (c['dPs_dTt'] * Tt_dot + c['dPs_dPt'] * Pt_dot +
                     c['dPs_dMN'] * MN_dot + c['dPs_dFAR'] * FAR_dot) * self._P_to_si

        # Chain rule for static properties: dprop = dprop/dTs * Ts_dot + dprop/dPs * Ps_dot + dprop/dFAR * FAR_dot
        def chain_rule(grad):
            # grad is (dProp/dFAR, dProp/dPs, dProp/dTs) in SI
            # Include direct FAR dependency plus indirect through Ts and Ps
            return grad[2] * Ts_dot_si + grad[1] * Ps_dot_si + grad[0] * FAR_dot

        hs_dot_si = chain_rule(c['grad_hs'])
        Ss_dot_si = chain_rule(c['grad_Ss'])
        gams_dot = chain_rule(c['grad_gams'])
        Cps_dot_si = chain_rule(c['grad_Cps'])
        Cvs_dot_si = chain_rule(c['grad_Cvs'])
        Rs_dot_si = chain_rule(c['grad_Rs'])

        # Vsonic = sqrt(gam_s * R_s * Ts)
        # d(Vsonic) = 1/(2*Vsonic) * (R_s*Ts * dgam_s + gam_s*Ts * dR_s + gam_s*R_s * dTs)
        Vsonic = c['Vsonic_si']
        gam_s, R_s, Ts_si = c['gam_s'], c['R_s'], c['Ts_si']
        if Vsonic > 0:
            Vsonic_dot_si = (R_s * Ts_si * gams_dot + gam_s * Ts_si * Rs_dot_si +
                            gam_s * R_s * Ts_dot_si) / (2 * Vsonic)
        else:
            Vsonic_dot_si = 0.0

        # V = MN * Vsonic
        V_dot_si = MN_dot * Vsonic + c['MN'] * Vsonic_dot_si

        # rhos = Ps / (R_s * Ts) = Ps * R_s^(-1) * Ts^(-1)
        # d(rhos) = dPs/(R*T) - Ps*dR/(R^2*T) - Ps*dT/(R*T^2)
        Ps_si, rhos_si = c['Ps_si'], c['rhos_si']
        rhos_dot_si = (Ps_dot_si / (R_s * Ts_si) -
                       Ps_si * Rs_dot_si / (R_s ** 2 * Ts_si) -
                       Ps_si * Ts_dot_si / (R_s * Ts_si ** 2))

        # area = W / (rhos * V)
        # d(area) = dW/(rhos*V) - W*drhos/(rhos^2*V) - W*dV/(rhos*V^2)
        W_si, V_si, area_si = c['W_si'], c['V_si'], c['area_si']
        W_dot_si = W_dot * self._W_to_si
        if V_si > 0 and rhos_si > 0:
            area_dot_si = (W_dot_si / (rhos_si * V_si) -
                          W_si * rhos_dot_si / (rhos_si ** 2 * V_si) -
                          W_si * V_dot_si / (rhos_si * V_si ** 2))
        else:
            area_dot_si = 0.0

        # Convert to output units
        return {
            'Ts': Ts_dot_si * self._T_from_si,
            'Ps': Ps_dot_si * self._P_from_si,
            'hs': hs_dot_si * self._h_from_si,
            'rhos': rhos_dot_si * self._rho_from_si,
            'MN': MN_dot,  # MN is dimensionless and directly input
            'V': V_dot_si * self._V_from_si,
            'Vsonic': Vsonic_dot_si * self._V_from_si,
            'area': area_dot_si * self._area_from_si,
            'gamma': gams_dot,
            'Cp': Cps_dot_si * self._S_from_si,
            'Cv': Cvs_dot_si * self._S_from_si,
            'S': Ss_dot_si * self._S_from_si,
            'R': Rs_dot_si * self._S_from_si,
        }

    def linearize_static_area(self, Tt, Pt, area, W, FAR=None):
        """
        Compute and cache gradients for static_from_area.

        This uses implicit differentiation since MN is solved via brentq.
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
        """
        if FAR is None:
            FAR = self.FAR

        # First compute the solution to get MN
        props = self.static_from_area(Tt, Pt, area, W, FAR=FAR)
        MN = props.MN

        # Now linearize static_from_MN at this MN
        self.linearize_static_MN(Tt, Pt, MN, W, FAR)
        c = self._static_MN_cache

        # Get d(area)/dMN from the static_from_MN linearization
        # We need to compute the JVP with only MN_dot = 1
        jvp_MN = self.jvp_static_MN(0.0, 0.0, 1.0, 0.0, 0.0)
        darea_dMN = jvp_MN['area']

        # Get d(area)/d(Tt, Pt, W, FAR) from static_from_MN linearization
        jvp_Tt = self.jvp_static_MN(1.0, 0.0, 0.0, 0.0, 0.0)
        jvp_Pt = self.jvp_static_MN(0.0, 1.0, 0.0, 0.0, 0.0)
        jvp_W = self.jvp_static_MN(0.0, 0.0, 0.0, 1.0, 0.0)
        jvp_FAR = self.jvp_static_MN(0.0, 0.0, 0.0, 0.0, 1.0)

        darea_dTt = jvp_Tt['area']
        darea_dPt = jvp_Pt['area']
        darea_dW = jvp_W['area']
        darea_dFAR = jvp_FAR['area']

        # Implicit function theorem: dMN/dx = -darea_dx / darea_dMN
        # For area_target: constraint is area_computed - area_target = 0
        # So dMN/d(area_target) = +1 / darea_dMN (positive!)
        if abs(darea_dMN) > 1e-20:
            dMN_dTt = -darea_dTt / darea_dMN
            dMN_dPt = -darea_dPt / darea_dMN
            dMN_darea = 1.0 / darea_dMN  # Fixed: was -1.0, should be +1.0
            dMN_dW = -darea_dW / darea_dMN
            dMN_dFAR = -darea_dFAR / darea_dMN
        else:
            dMN_dTt = dMN_dPt = dMN_darea = dMN_dW = dMN_dFAR = 0.0

        # Store the implicit derivatives
        self._static_area_cache = {
            'MN': MN,
            'FAR': FAR,
            'dMN_dTt': dMN_dTt,
            'dMN_dPt': dMN_dPt,
            'dMN_darea': dMN_darea,
            'dMN_dW': dMN_dW,
            'dMN_dFAR': dMN_dFAR,
            # Also store the per-input JVPs from static_from_MN for total derivatives
            'jvp_Tt': jvp_Tt,
            'jvp_Pt': jvp_Pt,
            'jvp_MN': jvp_MN,
            'jvp_W': jvp_W,
            'jvp_FAR': jvp_FAR,
        }

    def jvp_static_area(self, Tt_dot, Pt_dot, area_dot, W_dot, FAR_dot=0.0):
        """
        Compute JVP for static_from_area using cached linearization.

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
        if not hasattr(self, '_static_area_cache'):
            raise RuntimeError("Must call linearize_static_area() before jvp_static_area()")

        c = self._static_area_cache

        # Compute MN_dot using implicit function theorem
        MN_dot = (c['dMN_dTt'] * Tt_dot + c['dMN_dPt'] * Pt_dot +
                  c['dMN_darea'] * area_dot + c['dMN_dW'] * W_dot +
                  c['dMN_dFAR'] * FAR_dot)

        # Total derivative = direct effect + effect through MN
        # d(output)/d(input) = partial(output)/partial(input) + partial(output)/partial(MN) * dMN/d(input)
        result = {}
        for prop in ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area', 'gamma', 'Cp', 'Cv', 'S', 'R']:
            result[prop] = (c['jvp_Tt'][prop] * Tt_dot +
                           c['jvp_Pt'][prop] * Pt_dot +
                           c['jvp_MN'][prop] * MN_dot +
                           c['jvp_W'][prop] * W_dot +
                           c['jvp_FAR'][prop] * FAR_dot)

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
