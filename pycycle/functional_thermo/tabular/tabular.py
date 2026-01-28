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
    """

    def __init__(self, FAR=0.0, spec=None):
        from openmdao.components.interp_util.interp import InterpND
        from pycycle.constants import AIR_JETA_TAB_SPEC

        super().__init__(composition={'FAR': FAR})

        # Use default spec if not provided
        if spec is None:
            spec = AIR_JETA_TAB_SPEC
        self.spec = spec

        self.FAR = FAR

        # Create interpolators for each property
        # Grid points: (FAR, P, T)
        points = (spec['FAR'], spec['P'], spec['T'])

        self._interps = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self._interps[prop] = InterpND(
                method='slinear',
                points=points,
                values=spec[prop],
                extrapolate=True
            )

    def _lookup(self, prop, T, P):
        """Internal lookup function."""
        x = np.array([self.FAR, float(P), float(T)])
        return self._interps[prop].interpolate(x)[0]

    # =========================================================================
    # Linearization and JAX-compatible derivatives
    # =========================================================================

    def linearize(self, T, P):
        """
        Compute and cache property gradients at the given state.

        This method computes and caches the gradients for use with jvp()
        and vjp() methods. Must be called before using jvp() or vjp().

        Parameters
        ----------
        T : float
            Temperature (K)
        P : float
            Pressure (Pa)
        """
        x = np.array([self.FAR, float(P), float(T)])

        # Store the linearization point
        self._lin_T = T
        self._lin_P = P

        # Compute and cache gradients for each property
        # gradients are (dProp/dFAR, dProp/dP, dProp/dT)
        self._gradients = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self._gradients[prop] = self._interps[prop].gradient(x)

    def jvp(self, T_dot, P_dot):
        """
        Compute Jacobian-vector product (forward-mode autodiff).

        Computes the directional derivative of all properties in the direction
        specified by the tangent vectors. Must call linearize() first.

        Parameters
        ----------
        T_dot : float
            Tangent vector for temperature
        P_dot : float
            Tangent vector for pressure

        Returns
        -------
        dict
            Dictionary of property tangents: {'h': h_dot, 'S': S_dot, ...}
        """
        if not hasattr(self, '_gradients'):
            raise RuntimeError("Must call linearize() before jvp()")

        result = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            # gradients are (dProp/dFAR, dProp/dP, dProp/dT)
            grad = self._gradients[prop]
            result[prop] = grad[2] * T_dot + grad[1] * P_dot

        return result

    def vjp(self, h_bar=0.0, S_bar=0.0, gamma_bar=0.0, Cp_bar=0.0,
            Cv_bar=0.0, rho_bar=0.0, R_bar=0.0):
        """
        Compute vector-Jacobian product (reverse-mode autodiff).

        Computes the gradient of a scalar loss with respect to inputs (T, P),
        given the gradient of the loss with respect to outputs. Must call
        linearize() first.

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
            (T_bar, P_bar) - gradients with respect to T and P
        """
        if not hasattr(self, '_gradients'):
            raise RuntimeError("Must call linearize() before vjp()")

        cotangents = {
            'h': h_bar, 'S': S_bar, 'gamma': gamma_bar,
            'Cp': Cp_bar, 'Cv': Cv_bar, 'rho': rho_bar, 'R': R_bar
        }

        T_bar = 0.0
        P_bar = 0.0

        for prop, cotan in cotangents.items():
            if cotan != 0.0:
                grad = self._gradients[prop]
                # gradients are (dProp/dFAR, dProp/dP, dProp/dT)
                T_bar = T_bar + grad[2] * cotan
                P_bar = P_bar + grad[1] * cotan

        return T_bar, P_bar

    # =========================================================================
    # Total property calculations
    # =========================================================================

    def props_TP(self, T, P):
        """Compute all thermodynamic properties from T and P."""
        return TotalProps(
            h=self.h(T, P),
            S=self.S(T, P),
            gamma=self.gamma(T, P),
            Cp=self.Cp(T, P),
            Cv=self.Cv(T, P),
            rho=self.rho(T, P),
            R=self.R(T, P)
        )

    def h(self, T, P):
        """Compute enthalpy (J/kg) from T (K) and P (Pa)."""
        return self._lookup('h', T, P)

    def S(self, T, P):
        """Compute entropy (J/(kg*K)) from T (K) and P (Pa)."""
        return self._lookup('S', T, P)

    def gamma(self, T, P):
        """Compute ratio of specific heats from T (K) and P (Pa)."""
        return self._lookup('gamma', T, P)

    def Cp(self, T, P):
        """Compute specific heat at constant pressure (J/(kg*K)) from T (K) and P (Pa)."""
        return self._lookup('Cp', T, P)

    def Cv(self, T, P):
        """Compute specific heat at constant volume (J/(kg*K)) from T (K) and P (Pa)."""
        return self._lookup('Cv', T, P)

    def rho(self, T, P):
        """Compute density (kg/m^3) from T (K) and P (Pa)."""
        return self._lookup('rho', T, P)

    def R(self, T, P):
        """Compute specific gas constant (J/(kg*K)) from T (K) and P (Pa)."""
        return self._lookup('R', T, P)

    # =========================================================================
    # Inverse calculations (solve for T)
    # =========================================================================

    def T_from_hP(self, h_target, P):
        """Solve for temperature given enthalpy and pressure."""
        from scipy.optimize import brentq

        def residual(T):
            return float(self.h(T, P)) - float(h_target)

        T_min, T_max = 150.0, 2500.0
        return brentq(residual, T_min, T_max, xtol=1e-10)

    def T_from_SP(self, S_target, P):
        """Solve for temperature given entropy and pressure."""
        from scipy.optimize import brentq

        def residual(T):
            return float(self.S(T, P)) - float(S_target)

        T_min, T_max = 150.0, 2500.0
        return brentq(residual, T_min, T_max, xtol=1e-10)

    # =========================================================================
    # Static property calculations
    # =========================================================================

    def static_from_MN(self, Tt, Pt, MN, W):
        """Compute static properties from total conditions and Mach number."""
        # Get gamma and R at total conditions
        gam = self.gamma(Tt, Pt)
        R_gas = self.R(Tt, Pt)

        # Isentropic relations
        # Ts/Tt = 1 / (1 + (gamma-1)/2 * MN^2)
        temp_ratio = 1.0 / (1.0 + (gam - 1.0) / 2.0 * MN**2)
        Ts = Tt * temp_ratio

        # Ps/Pt = (Ts/Tt)^(gamma/(gamma-1))
        Ps = Pt * temp_ratio**(gam / (gam - 1.0))

        # Get static enthalpy
        hs = self.h(Ts, Ps)

        # Speed of sound and velocity
        Vsonic = np.sqrt(gam * R_gas * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps / (R_gas * Ts)

        # Area from continuity: W = rho * V * A
        area = W / (rhos * V) if V > 0 else np.inf

        return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                          MN=MN, V=V, Vsonic=Vsonic, area=area)

    def static_from_area(self, Tt, Pt, area, W, MN_guess=0.5, subsonic=True):
        """Compute static properties from total conditions and flow area."""
        from scipy.optimize import brentq

        def area_residual(MN):
            props = self.static_from_MN(Tt, Pt, MN, W)
            return float(props.area) - float(area)

        # For subsonic flow, MN is between 0 and 1
        # For supersonic flow, MN is > 1
        if subsonic:
            MN_min, MN_max = 0.01, 0.999
        else:
            MN_min, MN_max = 1.001, 5.0

        MN = brentq(area_residual, MN_min, MN_max, xtol=1e-10)
        return self.static_from_MN(Tt, Pt, MN, W)

    def static_from_Ps(self, Tt, Pt, Ps, W):
        """Compute static properties from total conditions and static pressure."""
        # Get gamma at total conditions
        gam = self.gamma(Tt, Pt)
        R_gas = self.R(Tt, Pt)

        # From isentropic relation: Ps/Pt = (Ts/Tt)^(gamma/(gamma-1))
        # Solve for Ts: Ts = Tt * (Ps/Pt)^((gamma-1)/gamma)
        pressure_ratio = Ps / Pt
        Ts = Tt * pressure_ratio**((gam - 1.0) / gam)

        # Get static enthalpy
        hs = self.h(Ts, Ps)

        # Compute Mach number from temperature ratio
        # Ts/Tt = 1 / (1 + (gamma-1)/2 * MN^2)
        # MN^2 = 2/(gamma-1) * (Tt/Ts - 1)
        temp_ratio = Ts / Tt
        MN_sq = 2.0 / (gam - 1.0) * (1.0 / temp_ratio - 1.0)
        MN = np.sqrt(max(0.0, MN_sq))

        # Speed of sound and velocity
        Vsonic = np.sqrt(gam * R_gas * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps / (R_gas * Ts)

        # Area from continuity
        area = W / (rhos * V) if V > 0 else np.inf

        return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                          MN=MN, V=V, Vsonic=Vsonic, area=area)
