"""
Base class for functional thermodynamic property calculations.

Units Convention (SI):
    T - Temperature (K)
    P - Pressure (Pa)
    h - Enthalpy (J/kg)
    S - Entropy (J/(kg*K))
    Cp - Specific heat at constant pressure (J/(kg*K))
    Cv - Specific heat at constant volume (J/(kg*K))
    gamma - Ratio of specific heats (-)
    rho - Density (kg/m^3)
    R - Specific gas constant (J/(kg*K))
    W - Mass flow rate (kg/s)
    V - Velocity (m/s)
    area - Flow area (m^2)
    MN - Mach number (-)
"""

from collections import namedtuple

# Named tuples for returning grouped properties
TotalProps = namedtuple('TotalProps', ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R'])
StaticProps = namedtuple('StaticProps', ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area'])


class ThermoInterface:
    """
    Base class defining the functional interface for thermodynamic calculations.

    Subclasses must implement all methods that raise NotImplementedError.

    Parameters
    ----------
    composition : dict or array-like
        Gas composition specification. Format depends on the implementation:
        - Tabular: {'FAR': 0.0} or array of mass fractions
        - CEA: {'N': ..., 'O': ..., ...} elemental ratios
    """

    def __init__(self, composition=None):
        self.composition = composition

    # =========================================================================
    # Total property calculations
    # =========================================================================

    def props_TP(self, T, P):
        """
        Compute thermodynamic properties from temperature and pressure.

        Parameters
        ----------
        T : float
            Temperature (K)
        P : float
            Pressure (Pa)

        Returns
        -------
        TotalProps
            Named tuple with (h, S, gamma, Cp, Cv, rho, R)
        """
        raise NotImplementedError("Subclass must implement props_TP")

    def h(self, T, P):
        """Compute enthalpy (J/kg) from T (K) and P (Pa)."""
        raise NotImplementedError("Subclass must implement h")

    def S(self, T, P):
        """Compute entropy (J/(kg*K)) from T (K) and P (Pa)."""
        raise NotImplementedError("Subclass must implement S")

    def gamma(self, T, P):
        """Compute ratio of specific heats from T (K) and P (Pa)."""
        raise NotImplementedError("Subclass must implement gamma")

    def Cp(self, T, P):
        """Compute specific heat at constant pressure (J/(kg*K)) from T (K) and P (Pa)."""
        raise NotImplementedError("Subclass must implement Cp")

    def Cv(self, T, P):
        """Compute specific heat at constant volume (J/(kg*K)) from T (K) and P (Pa)."""
        raise NotImplementedError("Subclass must implement Cv")

    def rho(self, T, P):
        """Compute density (kg/m^3) from T (K) and P (Pa)."""
        raise NotImplementedError("Subclass must implement rho")

    def R(self, T, P):
        """Compute specific gas constant (J/(kg*K)) from T (K) and P (Pa)."""
        raise NotImplementedError("Subclass must implement R")

    # =========================================================================
    # Inverse calculations (solve for T)
    # =========================================================================

    def T_from_hP(self, h, P):
        """
        Solve for temperature given enthalpy and pressure.

        Parameters
        ----------
        h : float
            Target enthalpy (J/kg)
        P : float
            Pressure (Pa)

        Returns
        -------
        float
            Temperature (K) such that h(T, P) = h_target
        """
        raise NotImplementedError("Subclass must implement T_from_hP")

    def T_from_SP(self, S, P):
        """
        Solve for temperature given entropy and pressure.

        Parameters
        ----------
        S : float
            Target entropy (J/(kg*K))
        P : float
            Pressure (Pa)

        Returns
        -------
        float
            Temperature (K) such that S(T, P) = S_target
        """
        raise NotImplementedError("Subclass must implement T_from_SP")

    # =========================================================================
    # Static property calculations (from total conditions)
    # =========================================================================

    def static_from_MN(self, Tt, Pt, MN, W):
        """
        Compute static properties from total conditions and Mach number.

        Uses isentropic relations to compute static temperature and pressure,
        then derives velocity, density, and area from continuity.

        Parameters
        ----------
        Tt : float
            Total temperature (K)
        Pt : float
            Total pressure (Pa)
        MN : float
            Mach number (-)
        W : float
            Mass flow rate (kg/s)

        Returns
        -------
        StaticProps
            Named tuple with (Ts, Ps, hs, rhos, MN, V, Vsonic, area)
        """
        raise NotImplementedError("Subclass must implement static_from_MN")

    def static_from_area(self, Tt, Pt, area, W, MN_guess=0.5, subsonic=True):
        """
        Compute static properties from total conditions and flow area.

        Solves for Mach number that satisfies continuity, then computes
        static properties using isentropic relations.

        Parameters
        ----------
        Tt : float
            Total temperature (K)
        Pt : float
            Total pressure (Pa)
        area : float
            Flow area (m^2)
        W : float
            Mass flow rate (kg/s)
        MN_guess : float, optional
            Initial guess for Mach number
        subsonic : bool, optional
            If True, find subsonic solution; if False, find supersonic

        Returns
        -------
        StaticProps
            Named tuple with (Ts, Ps, hs, rhos, MN, V, Vsonic, area)
        """
        raise NotImplementedError("Subclass must implement static_from_area")

    def static_from_Ps(self, Tt, Pt, Ps, W):
        """
        Compute static properties from total conditions and static pressure.

        Solves for static temperature using isentropic relations, then
        computes Mach number, velocity, density, and area.

        Parameters
        ----------
        Tt : float
            Total temperature (K)
        Pt : float
            Total pressure (Pa)
        Ps : float
            Static pressure (Pa)
        W : float
            Mass flow rate (kg/s)

        Returns
        -------
        StaticProps
            Named tuple with (Ts, Ps, hs, rhos, MN, V, Vsonic, area)
        """
        raise NotImplementedError("Subclass must implement static_from_Ps")

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
        raise NotImplementedError("Subclass must implement linearize")

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
        raise NotImplementedError("Subclass must implement jvp")

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
        raise NotImplementedError("Subclass must implement vjp")
