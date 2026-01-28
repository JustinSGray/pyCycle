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

    def _lookup_si(self, prop, T_si, P_si):
        """Internal lookup function in SI units."""
        x = np.array([self.FAR, float(P_si), float(T_si)])
        return self._interps[prop].interpolate(x)[0]

    # =========================================================================
    # Linearization and JAX-compatible derivatives
    # =========================================================================

    def linearize(self, T, P):
        """
        Compute and cache property gradients at the given state.
        """
        # Convert inputs to SI for lookup
        T_si = self._convert_T_to_si(T)
        P_si = P * self._P_to_si

        x = np.array([self.FAR, float(P_si), float(T_si)])

        # Store the linearization point
        self._lin_T = T
        self._lin_P = P
        self._lin_T_si = T_si
        self._lin_P_si = P_si

        # Compute and cache gradients for each property (in SI units)
        # gradients are (dProp/dFAR, dProp/dP, dProp/dT)
        self._gradients_si = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self._gradients_si[prop] = self._interps[prop].gradient(x)

    def jvp(self, T_dot, P_dot):
        """
        Compute Jacobian-vector product (forward-mode autodiff).
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
            dprop_dT = grad[2] * self._T_to_si * out_factor
            dprop_dP = grad[1] * self._P_to_si * out_factor

            result[prop] = dprop_dT * T_dot + dprop_dP * P_dot

        return result

    def vjp(self, h_bar=0.0, S_bar=0.0, gamma_bar=0.0, Cp_bar=0.0,
            Cv_bar=0.0, rho_bar=0.0, R_bar=0.0):
        """
        Compute vector-Jacobian product (reverse-mode autodiff).
        """
        if not hasattr(self, '_gradients_si'):
            raise RuntimeError("Must call linearize() before vjp()")

        cotangents = {
            'h': h_bar, 'S': S_bar, 'gamma': gamma_bar,
            'Cp': Cp_bar, 'Cv': Cv_bar, 'rho': rho_bar, 'R': R_bar
        }

        T_bar = 0.0
        P_bar = 0.0

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

                dprop_dT = grad[2] * self._T_to_si * out_factor
                dprop_dP = grad[1] * self._P_to_si * out_factor

                T_bar = T_bar + dprop_dT * cotan
                P_bar = P_bar + dprop_dP * cotan

        return T_bar, P_bar

    # =========================================================================
    # Total property calculations
    # =========================================================================

    def props_TP(self, T, P):
        """Compute all thermodynamic properties from T and P."""
        # Convert inputs to SI
        T_si = self._convert_T_to_si(T)
        P_si = P * self._P_to_si

        # Lookup in SI
        props_si = TotalProps(
            h=self._lookup_si('h', T_si, P_si),
            S=self._lookup_si('S', T_si, P_si),
            gamma=self._lookup_si('gamma', T_si, P_si),
            Cp=self._lookup_si('Cp', T_si, P_si),
            Cv=self._lookup_si('Cv', T_si, P_si),
            rho=self._lookup_si('rho', T_si, P_si),
            R=self._lookup_si('R', T_si, P_si)
        )

        # Convert outputs from SI
        return self._convert_total_props_from_si(props_si)

    def h(self, T, P):
        """Compute enthalpy from T and P."""
        return self.props_TP(T, P).h

    def S(self, T, P):
        """Compute entropy from T and P."""
        return self.props_TP(T, P).S

    def gamma(self, T, P):
        """Compute ratio of specific heats from T and P."""
        return self.props_TP(T, P).gamma

    def Cp(self, T, P):
        """Compute specific heat at constant pressure from T and P."""
        return self.props_TP(T, P).Cp

    def Cv(self, T, P):
        """Compute specific heat at constant volume from T and P."""
        return self.props_TP(T, P).Cv

    def rho(self, T, P):
        """Compute density from T and P."""
        return self.props_TP(T, P).rho

    def R(self, T, P):
        """Compute specific gas constant from T and P."""
        return self.props_TP(T, P).R

    # =========================================================================
    # Inverse calculations (solve for T)
    # =========================================================================

    def T_from_hP(self, h_target, P):
        """Solve for temperature given enthalpy and pressure."""
        from scipy.optimize import brentq

        # Convert inputs to SI
        h_si = h_target * self._h_to_si
        P_si = P * self._P_to_si

        def residual(T_si):
            return float(self._lookup_si('h', T_si, P_si)) - float(h_si)

        T_min, T_max = 150.0, 2500.0
        T_si = brentq(residual, T_min, T_max, xtol=1e-10)

        # Convert output from SI
        return self._convert_T_from_si(T_si)

    def T_from_SP(self, S_target, P):
        """Solve for temperature given entropy and pressure."""
        from scipy.optimize import brentq

        # Convert inputs to SI
        S_si = S_target * self._S_to_si
        P_si = P * self._P_to_si

        def residual(T_si):
            return float(self._lookup_si('S', T_si, P_si)) - float(S_si)

        T_min, T_max = 150.0, 2500.0
        T_si = brentq(residual, T_min, T_max, xtol=1e-10)

        return self._convert_T_from_si(T_si)

    # =========================================================================
    # Static property calculations
    # =========================================================================

    def _static_from_MN_si(self, Tt_si, Pt_si, MN, W_si):
        """Compute static properties in SI units."""
        # Get gamma and R at total conditions (in SI)
        gam = self._lookup_si('gamma', Tt_si, Pt_si)
        R_gas = self._lookup_si('R', Tt_si, Pt_si)

        # Isentropic relations
        temp_ratio = 1.0 / (1.0 + (gam - 1.0) / 2.0 * MN**2)
        Ts = Tt_si * temp_ratio
        Ps = Pt_si * temp_ratio**(gam / (gam - 1.0))

        # Get static enthalpy
        hs = self._lookup_si('h', Ts, Ps)

        # Speed of sound and velocity
        Vsonic = np.sqrt(gam * R_gas * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps / (R_gas * Ts)

        # Area from continuity
        area = W_si / (rhos * V) if V > 0 else np.inf

        return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                          MN=MN, V=V, Vsonic=Vsonic, area=area)

    def static_from_MN(self, Tt, Pt, MN, W):
        """Compute static properties from total conditions and Mach number."""
        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        W_si = W * self._W_to_si

        props_si = self._static_from_MN_si(Tt_si, Pt_si, MN, W_si)

        # Convert outputs from SI
        return self._convert_static_props_from_si(props_si)

    def static_from_area(self, Tt, Pt, area, W, MN_guess=0.5, subsonic=True):
        """Compute static properties from total conditions and flow area."""
        from scipy.optimize import brentq

        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        area_si = area * self._area_to_si
        W_si = W * self._W_to_si

        def area_residual(MN):
            props = self._static_from_MN_si(Tt_si, Pt_si, MN, W_si)
            return float(props.area) - float(area_si)

        if subsonic:
            MN_min, MN_max = 0.01, 0.999
        else:
            MN_min, MN_max = 1.001, 5.0

        MN = brentq(area_residual, MN_min, MN_max, xtol=1e-10)
        props_si = self._static_from_MN_si(Tt_si, Pt_si, MN, W_si)

        return self._convert_static_props_from_si(props_si)

    def static_from_Ps(self, Tt, Pt, Ps, W):
        """Compute static properties from total conditions and static pressure."""
        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        Ps_si = Ps * self._P_to_si
        W_si = W * self._W_to_si

        # Get gamma at total conditions
        gam = self._lookup_si('gamma', Tt_si, Pt_si)
        R_gas = self._lookup_si('R', Tt_si, Pt_si)

        # From isentropic relation
        pressure_ratio = Ps_si / Pt_si
        Ts = Tt_si * pressure_ratio**((gam - 1.0) / gam)

        # Get static enthalpy
        hs = self._lookup_si('h', Ts, Ps_si)

        # Compute Mach number from temperature ratio
        temp_ratio = Ts / Tt_si
        MN_sq = 2.0 / (gam - 1.0) * (1.0 / temp_ratio - 1.0)
        MN = np.sqrt(max(0.0, MN_sq))

        # Speed of sound and velocity
        Vsonic = np.sqrt(gam * R_gas * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps_si / (R_gas * Ts)

        # Area from continuity
        area = W_si / (rhos * V) if V > 0 else np.inf

        props_si = StaticProps(Ts=Ts, Ps=Ps_si, hs=hs, rhos=rhos,
                              MN=MN, V=V, Vsonic=Vsonic, area=area)

        return self._convert_static_props_from_si(props_si)
