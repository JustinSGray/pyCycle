"""
Base class for functional thermodynamic property calculations.

Internal Units (SI):
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

The thermo classes can accept inputs in different unit systems via the
`input_units` parameter:
    - 'SI': SI units (default) - no conversion
    - 'English': pyCycle English units (degR, psi, Btu/lbm, etc.)
"""

from collections import namedtuple

from openmdao.utils.units import unit_conversion

# Named tuples for returning grouped properties
TotalProps = namedtuple('TotalProps', ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R'])
StaticProps = namedtuple('StaticProps', ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                                          'gamma', 'Cp', 'Cv', 'S', 'R'])

# SI units (internal)
_SI_UNITS = {
    'T': 'K',
    'P': 'Pa',
    'h': 'J/kg',
    'S': 'J/(kg*K)',
    'Cp': 'J/(kg*K)',
    'Cv': 'J/(kg*K)',
    'R': 'J/(kg*K)',
    'rho': 'kg/m**3',
    'V': 'm/s',
    'Vsonic': 'm/s',
    'area': 'm**2',
    'W': 'kg/s',
}

# pyCycle English units
_ENGLISH_UNITS = {
    'T': 'degR',
    'P': 'lbf/inch**2',
    'h': 'Btu/lbm',
    'S': 'Btu/(lbm*degR)',
    'Cp': 'Btu/(lbm*degR)',
    'Cv': 'Btu/(lbm*degR)',
    'R': 'Btu/(lbm*degR)',
    'rho': 'lbm/ft**3',
    'V': 'ft/s',
    'Vsonic': 'ft/s',
    'area': 'inch**2',
    'W': 'lbm/s',
}


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
    input_units : str, optional
        Unit system for inputs/outputs:
        - 'SI': Use SI units (default) - no conversion
        - 'English': Use pyCycle English units
    """

    def __init__(self, composition=None, input_units='SI'):
        self.composition = composition
        self.input_units = input_units

        # Set up unit conversion factors
        # Factors convert: input_units -> SI (multiply) and SI -> input_units (multiply)
        if input_units == 'SI':
            # No conversion needed - all factors are 1.0
            self._T_to_si = 1.0
            self._T_from_si = 1.0
            self._T_offset_to_si = 0.0
            self._T_offset_from_si = 0.0

            self._P_to_si = 1.0
            self._P_from_si = 1.0

            self._h_to_si = 1.0
            self._h_from_si = 1.0

            self._S_to_si = 1.0
            self._S_from_si = 1.0

            self._rho_to_si = 1.0
            self._rho_from_si = 1.0

            self._V_to_si = 1.0
            self._V_from_si = 1.0

            self._area_to_si = 1.0
            self._area_from_si = 1.0

            self._W_to_si = 1.0
            self._W_from_si = 1.0

        elif input_units == 'English':
            # Get conversion factors from OpenMDAO unit library
            # Temperature (has offset)
            factor, offset = unit_conversion(_ENGLISH_UNITS['T'], _SI_UNITS['T'])
            self._T_to_si = factor
            self._T_offset_to_si = offset
            factor, offset = unit_conversion(_SI_UNITS['T'], _ENGLISH_UNITS['T'])
            self._T_from_si = factor
            self._T_offset_from_si = offset

            # Pressure
            self._P_to_si, _ = unit_conversion(_ENGLISH_UNITS['P'], _SI_UNITS['P'])
            self._P_from_si, _ = unit_conversion(_SI_UNITS['P'], _ENGLISH_UNITS['P'])

            # Enthalpy
            self._h_to_si, _ = unit_conversion(_ENGLISH_UNITS['h'], _SI_UNITS['h'])
            self._h_from_si, _ = unit_conversion(_SI_UNITS['h'], _ENGLISH_UNITS['h'])

            # Entropy / Cp / Cv / R (same units)
            self._S_to_si, _ = unit_conversion(_ENGLISH_UNITS['S'], _SI_UNITS['S'])
            self._S_from_si, _ = unit_conversion(_SI_UNITS['S'], _ENGLISH_UNITS['S'])

            # Density
            self._rho_to_si, _ = unit_conversion(_ENGLISH_UNITS['rho'], _SI_UNITS['rho'])
            self._rho_from_si, _ = unit_conversion(_SI_UNITS['rho'], _ENGLISH_UNITS['rho'])

            # Velocity
            self._V_to_si, _ = unit_conversion(_ENGLISH_UNITS['V'], _SI_UNITS['V'])
            self._V_from_si, _ = unit_conversion(_SI_UNITS['V'], _ENGLISH_UNITS['V'])

            # Area
            self._area_to_si, _ = unit_conversion(_ENGLISH_UNITS['area'], _SI_UNITS['area'])
            self._area_from_si, _ = unit_conversion(_SI_UNITS['area'], _ENGLISH_UNITS['area'])

            # Mass flow
            self._W_to_si, _ = unit_conversion(_ENGLISH_UNITS['W'], _SI_UNITS['W'])
            self._W_from_si, _ = unit_conversion(_SI_UNITS['W'], _ENGLISH_UNITS['W'])

        else:
            raise ValueError(f"Unknown input_units: {input_units}. Use 'SI' or 'English'.")

    # =========================================================================
    # Unit conversion helpers
    # =========================================================================

    def _convert_T_to_si(self, T):
        """Convert temperature from input units to SI (K)."""
        return (T + self._T_offset_to_si) * self._T_to_si

    def _convert_T_from_si(self, T_si):
        """Convert temperature from SI (K) to input units."""
        return (T_si + self._T_offset_from_si) * self._T_from_si

    def _convert_total_props_from_si(self, props):
        """Convert TotalProps from SI to input units."""
        return TotalProps(
            h=props.h * self._h_from_si,
            S=props.S * self._S_from_si,
            gamma=props.gamma,  # dimensionless
            Cp=props.Cp * self._S_from_si,  # same units as S
            Cv=props.Cv * self._S_from_si,
            rho=props.rho * self._rho_from_si,
            R=props.R * self._S_from_si,
        )

    def _convert_static_props_from_si(self, props):
        """Convert StaticProps from SI to input units."""
        return StaticProps(
            Ts=self._convert_T_from_si(props.Ts),
            Ps=props.Ps * self._P_from_si,
            hs=props.hs * self._h_from_si,
            rhos=props.rhos * self._rho_from_si,
            MN=props.MN,  # dimensionless
            V=props.V * self._V_from_si,
            Vsonic=props.Vsonic * self._V_from_si,
            area=props.area * self._area_from_si,
            gamma=props.gamma,  # dimensionless
            Cp=props.Cp * self._S_from_si,  # same units as S
            Cv=props.Cv * self._S_from_si,
            S=props.S * self._S_from_si,
            R=props.R * self._S_from_si,
        )

    # =========================================================================
    # Total property calculations
    # =========================================================================

    def props_TP(self, T, P):
        """
        Compute thermodynamic properties from temperature and pressure.

        Parameters
        ----------
        T : float
            Temperature (in input units)
        P : float
            Pressure (in input units)

        Returns
        -------
        TotalProps
            Named tuple with (h, S, gamma, Cp, Cv, rho, R) in input units
        """
        raise NotImplementedError("Subclass must implement props_TP")

    def h(self, T, P):
        """Compute enthalpy from T and P (in input units)."""
        raise NotImplementedError("Subclass must implement h")

    def S(self, T, P):
        """Compute entropy from T and P (in input units)."""
        raise NotImplementedError("Subclass must implement S")

    def gamma(self, T, P):
        """Compute ratio of specific heats from T and P."""
        raise NotImplementedError("Subclass must implement gamma")

    def Cp(self, T, P):
        """Compute specific heat at constant pressure from T and P (in input units)."""
        raise NotImplementedError("Subclass must implement Cp")

    def Cv(self, T, P):
        """Compute specific heat at constant volume from T and P (in input units)."""
        raise NotImplementedError("Subclass must implement Cv")

    def rho(self, T, P):
        """Compute density from T and P (in input units)."""
        raise NotImplementedError("Subclass must implement rho")

    def R(self, T, P):
        """Compute specific gas constant from T and P (in input units)."""
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
            Target enthalpy (in input units)
        P : float
            Pressure (in input units)

        Returns
        -------
        float
            Temperature (in input units) such that h(T, P) = h_target
        """
        raise NotImplementedError("Subclass must implement T_from_hP")

    def T_from_SP(self, S, P):
        """
        Solve for temperature given entropy and pressure.

        Parameters
        ----------
        S : float
            Target entropy (in input units)
        P : float
            Pressure (in input units)

        Returns
        -------
        float
            Temperature (in input units) such that S(T, P) = S_target
        """
        raise NotImplementedError("Subclass must implement T_from_SP")

    # =========================================================================
    # Static property calculations (from total conditions)
    # =========================================================================

    def static_from_MN(self, Tt, Pt, MN, W):
        """
        Compute static properties from total conditions and Mach number.

        Parameters
        ----------
        Tt : float
            Total temperature (in input units)
        Pt : float
            Total pressure (in input units)
        MN : float
            Mach number (-)
        W : float
            Mass flow rate (in input units)

        Returns
        -------
        StaticProps
            Named tuple with (Ts, Ps, hs, rhos, MN, V, Vsonic, area) in input units
        """
        raise NotImplementedError("Subclass must implement static_from_MN")

    def static_from_area(self, Tt, Pt, area, W, MN_guess=0.5, subsonic=True):
        """
        Compute static properties from total conditions and flow area.

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

        Returns
        -------
        StaticProps
            Named tuple with (Ts, Ps, hs, rhos, MN, V, Vsonic, area) in input units
        """
        raise NotImplementedError("Subclass must implement static_from_area")

    def static_from_Ps(self, Tt, Pt, Ps, W):
        """
        Compute static properties from total conditions and static pressure.

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

        Returns
        -------
        StaticProps
            Named tuple with (Ts, Ps, hs, rhos, MN, V, Vsonic, area) in input units
        """
        raise NotImplementedError("Subclass must implement static_from_Ps")

    # =========================================================================
    # Linearization and JAX-compatible derivatives
    # =========================================================================

    def linearize(self, T, P):
        """
        Compute and cache property gradients at the given state.

        Parameters
        ----------
        T : float
            Temperature (in input units)
        P : float
            Pressure (in input units)
        """
        raise NotImplementedError("Subclass must implement linearize")

    def jvp(self, T_dot, P_dot):
        """
        Compute Jacobian-vector product (forward-mode autodiff).

        Must call linearize() first.

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

        Must call linearize() first.

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
