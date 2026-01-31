"""
JAX-compatible wrappers for functional thermodynamic interfaces.

This module provides:
- JaxThermo: A wrapper class that provides fully JAX-traceable thermo calculations
- Named index classes for accessing property arrays

The wrappers use pure JAX functions that JAX can differentiate through directly,
enabling efficient JIT-compiled Jacobian computation via jacfwd/jacrev.
"""


# =============================================================================
# Named Index Classes for Property Arrays
# =============================================================================

class TotalPropsIdx:
    """Named indices for total property arrays returned by JaxThermo.props_TP."""
    h = 0
    S = 1
    gamma = 2
    Cp = 3
    Cv = 4
    rho = 5
    R = 6

    @classmethod
    def count(cls):
        return 7


class StaticPropsIdx:
    """Named indices for static property arrays returned by JaxThermo.static_from_*."""
    Ts = 0
    Ps = 1
    hs = 2
    rhos = 3
    MN = 4
    V = 5
    Vsonic = 6
    area = 7
    gamma = 8
    Cp = 9
    Cv = 10
    S = 11
    R = 12

    @classmethod
    def count(cls):
        return 13


# =============================================================================
# JaxThermo Wrapper Class
# =============================================================================

class JaxThermo:
    """
    Wrapper that provides fully JAX-traceable interface for tabular thermo calculations.

    Uses pure JAX functions that JAX can differentiate through directly. This allows:
    - JIT compilation of the full computation
    - Efficient Jacobian computation via jacfwd/jacrev
    - One-time tracing with cached compiled functions

    All methods accept composition as an array parameter where composition[0] = FAR.

    Note: This class is designed to be shareable across multiple JaxElement instances.

    Parameters
    ----------
    spec : dict
        Tabular thermo specification dictionary containing grid points and property values.
        Typically AIR_JETA_TAB_SPEC or a custom spec dict.
    """

    def __init__(self, spec):
        # Create pure JAX thermo for computation
        from pycycle.functional_thermo.jax_tabular import JaxTabularThermo
        self._jax_thermo = JaxTabularThermo(spec)

        self._setup_wrappers()

    def _setup_wrappers(self):
        """Create pure JAX wrappers for thermo methods."""
        jax_thermo = self._jax_thermo

        # T_from_hP: Pure JAX, composition[0] = FAR
        def T_from_hP_pure(h, P, composition):
            FAR = composition[0]
            return jax_thermo.T_from_hP(h, P, FAR)

        # props_TP: Pure JAX, composition[0] = FAR
        def props_TP_pure(T, P, composition):
            FAR = composition[0]
            return jax_thermo.props_TP(T, P, FAR)

        # static_from_MN: Pure JAX
        def static_from_MN_pure(Tt, Pt, MN, W, composition):
            FAR = composition[0]
            return jax_thermo.static_from_MN(Tt, Pt, MN, W, FAR)

        # static_from_area: Pure JAX
        def static_from_area_pure(Tt, Pt, area, W, composition):
            FAR = composition[0]
            return jax_thermo.static_from_area(Tt, Pt, area, W, FAR)

        # Assign the pure JAX functions
        self.T_from_hP = T_from_hP_pure
        self.props_TP = props_TP_pure
        self.static_from_MN = static_from_MN_pure
        self.static_from_area = static_from_area_pure
