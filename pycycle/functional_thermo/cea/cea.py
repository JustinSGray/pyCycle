"""
CEA thermodynamic property calculations using Gibbs free energy minimization.

This module provides a functional interface for computing thermodynamic properties
using NASA CEA (Chemical Equilibrium with Applications) methodology.
"""

import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq

from ..base import ThermoInterface, TotalProps, StaticProps
from pycycle.constants import (P_REF, R_UNIVERSAL_ENG, R_UNIVERSAL_SI,
                                MIN_VALID_CONCENTRATION, CEA_AIR_COMPOSITION)
from pycycle.thermo.cea import species_data


# Unit conversion: cal/g to J/kg (internal CEA conversion)
CAL_G_TO_J_KG = 4184.0


def _resid_weighting(n):
    """Sigmoid weighting function for trace species damping."""
    old = np.seterr(under='ignore')
    try:
        return (1 / (1 + np.exp(-1e5 * n)) - 0.5) * 2
    finally:
        np.seterr(**old)


class CEAThermo(ThermoInterface):
    """
    Thermodynamic property calculations using NASA CEA.

    This implementation solves for chemical equilibrium at each state point
    using Gibbs free energy minimization with Lagrange multipliers.

    Parameters
    ----------
    composition : dict, optional
        Elemental composition as atomic ratios.
        Example: {'N': 0.0539, 'O': 0.0145, 'Ar': 0.000323, 'C': 1.1e-5}
        Default is CEA_AIR_COMPOSITION (dry air)
    thermo_data : module, optional
        Species thermodynamic data module. Default is species_data.janaf
    input_units : str, optional
        Unit system for inputs/outputs:
        - 'SI': Use SI units (default)
        - 'English': Use pyCycle English units
    """

    def __init__(self, composition=None, thermo_data=None, input_units='SI'):
        if composition is None:
            composition = CEA_AIR_COMPOSITION
        super().__init__(composition, input_units)

        if thermo_data is None:
            thermo_data = species_data.janaf

        # Create the Properties object that handles NASA polynomials
        self._props = species_data.Properties(thermo_data, init_elements=composition)

        # Store frequently used attributes
        self.num_prod = self._props.num_prod
        self.num_element = self._props.num_element
        self.aij = self._props.aij
        self.b0 = self._props.b0

        # Solver settings
        self._atol = 1e-7
        self._rtol = 1e-7
        self._maxiter = 100
        self._stall_limit = 4
        self._stall_tol = 1e-10

        # Initial guess for species concentrations
        self._n_init = np.ones(self.num_prod) / self.num_prod / 10

        # Cache for equilibrium solution
        self._cached_T = None
        self._cached_P = None
        self._cached_n = None
        self._cached_pi = None
        self._cached_n_moles = None

        # Use trace damping like OpenMDAO implementation
        self._use_trace_damping = True

    def _wrap_T(self, T):
        """Wrap scalar T as array for Properties methods."""
        return np.atleast_1d(T)

    def _solve_equilibrium(self, T_si, P_si):
        """
        Solve chemical equilibrium using Newton iteration.

        Parameters
        ----------
        T_si : float
            Temperature (K) - SI units
        P_si : float
            Pressure (Pa) - SI units

        Returns
        -------
        n : ndarray
            Species molar concentrations
        pi : ndarray
            Lagrange multipliers
        n_moles : float
            Total molar concentration
        """
        # Check if we have a cached solution at this state
        if (self._cached_T is not None and
            np.isclose(T_si, self._cached_T, rtol=1e-12) and
            np.isclose(P_si, self._cached_P, rtol=1e-12)):
            return self._cached_n.copy(), self._cached_pi.copy(), self._cached_n_moles

        # Convert pressure from Pa to bar, then normalize by P_REF
        P_bar = P_si / 100000.0
        P_norm = P_bar / P_REF

        # Get thermodynamic properties at this temperature
        T_arr = self._wrap_T(T_si)
        H0 = self._props.H0(T_arr)
        S0 = self._props.S0(T_arr)

        # Start with initial guess
        n = self._n_init.copy()
        pi = np.zeros(self.num_element)

        # Newton iteration
        prev_norm = np.inf
        stall_count = 0

        for iteration in range(self._maxiter):
            n_moles = np.sum(n)

            # Compute residuals
            resids = self._compute_residuals(n, pi, H0, S0, P_norm, n_moles)
            resid_norm = np.linalg.norm(resids)

            # Check convergence
            if resid_norm < self._atol:
                break

            # Enable trace removal when close to convergence
            remove_trace = resid_norm < 1e-4
            if remove_trace:
                trace_mask = n <= MIN_VALID_CONCENTRATION + 1e-20
                resids[:self.num_prod][trace_mask] = 0.0

            # Compute Jacobian
            J = self._compute_jacobian(n, pi, n_moles, remove_trace)

            # Solve for Newton step
            try:
                delta = solve(J, -resids)
            except np.linalg.LinAlgError:
                J_reg = J + 1e-10 * np.eye(J.shape[0])
                delta = solve(J_reg, -resids)

            # Bounds-enforcing update
            dn = delta[:self.num_prod]
            dpi = delta[self.num_prod:]

            n_new = n + dn
            pi_new = pi + dpi

            # Clip species concentrations to bounds
            n_new = np.maximum(n_new, MIN_VALID_CONCENTRATION)

            n = n_new
            pi = pi_new

            # Check for stall
            if abs(resid_norm - prev_norm) < self._stall_tol:
                stall_count += 1
            else:
                stall_count = 0

            if stall_count >= self._stall_limit:
                break

            prev_norm = resid_norm

        # Cache the solution
        self._cached_T = T_si
        self._cached_P = P_si
        self._cached_n = n.copy()
        self._cached_pi = pi.copy()
        self._cached_n_moles = np.sum(n)

        return n, pi, self._cached_n_moles

    def _compute_residuals(self, n, pi, H0, S0, P_norm, n_moles):
        """Compute equilibrium residual equations."""
        num_prod = self.num_prod
        num_element = self.num_element

        resids = np.zeros(num_prod + num_element)

        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            log_n = np.log(np.maximum(n, MIN_VALID_CONCENTRATION))
            mu = H0 - S0 + log_n + np.log(P_norm) - np.log(n_moles)
        finally:
            np.seterr(**old)

        resids[:num_prod] = mu - self.aij.T @ pi

        if self._use_trace_damping:
            weights = _resid_weighting(n * n_moles)
            resids[:num_prod] *= weights

        resids[num_prod:] = self.aij @ n - self.b0

        return resids

    def _compute_jacobian(self, n, pi, n_moles, remove_trace=False):
        """Compute Jacobian matrix for Newton solver."""
        num_prod = self.num_prod
        num_element = self.num_element
        size = num_prod + num_element

        J = np.zeros((size, size))

        MW = 1.0 / n_moles
        J[:num_prod, :num_prod] = -MW
        diag = 1.0 / n - MW
        np.fill_diagonal(J[:num_prod, :num_prod], diag)

        if self._use_trace_damping:
            weights = _resid_weighting(n * n_moles)
            J[:num_prod, :num_prod] *= weights[:, np.newaxis]

        J[:num_prod, num_prod:] = -self.aij.T
        if self._use_trace_damping:
            J[:num_prod, num_prod:] *= weights[:, np.newaxis]

        J[num_prod:, :num_prod] = self.aij

        if remove_trace:
            for j in range(num_prod):
                if n[j] <= MIN_VALID_CONCENTRATION + 1e-20:
                    J[j, :] = 0.0
                    J[j, j] = -1.0

        return J

    def _build_props_matrices(self, T_si, n, n_moles):
        """Build matrices for property calculations."""
        num_element = self.num_element
        ne1 = num_element + 1

        T_arr = self._wrap_T(T_si)
        H0 = self._props.H0(T_arr)

        lhs_TP = np.zeros((ne1, ne1))
        for i in range(num_element):
            lhs_TP[i, :num_element] = np.dot(self._props.aij_prod[i], n)

        lhs_TP[num_element, :num_element] = self.b0
        lhs_TP[:num_element, num_element] = self.b0
        lhs_TP[num_element, num_element] = 0.0

        n_H0 = n * H0
        rhs_T = np.zeros(ne1)
        rhs_T[:num_element] = np.sum(self.aij * n_H0, axis=1)
        rhs_T[num_element] = np.sum(n_H0)

        rhs_P = np.zeros(ne1)
        rhs_P[:num_element] = self.b0
        rhs_P[num_element] = n_moles

        return lhs_TP, rhs_T, rhs_P

    def _compute_properties_si(self, T_si, P_si, n, n_moles, result_T, result_P):
        """Compute thermodynamic properties in SI units."""
        T_arr = self._wrap_T(T_si)
        P_bar = P_si / 100000.0

        H0 = self._props.H0(T_arr)
        S0 = self._props.S0(T_arr)
        Cp0 = self._props.Cp0(T_arr)

        num_element = self.num_element

        dlnVqdlnP = -1.0 + result_P[num_element]
        dlnVqdlnT = 1.0 - result_T[num_element]

        Cpf = np.sum(n * Cp0)
        n_H0 = n * H0

        Cpe = -np.sum(np.sum(self.aij * n_H0, axis=1) * result_T[:num_element])
        Cpe += np.sum(n_H0 * H0)
        Cpe -= np.sum(n_H0) * result_T[num_element]

        # Enthalpy (cal/g)
        h_eng = np.sum(n_H0) * R_UNIVERSAL_ENG * T_si

        # Entropy (cal/(g*K))
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            S_term = S0 + np.log(n_moles / n / (P_bar / P_REF))
        except FloatingPointError:
            S_term = S0 + np.log(n_moles / n / 1e-5)
        finally:
            np.seterr(**old)
        S_eng = R_UNIVERSAL_ENG * np.sum(n * S_term)

        # Cp (cal/(g*K))
        Cp_eng = (Cpe + Cpf) * R_UNIVERSAL_ENG

        # Cv (cal/(g*K))
        Cv_eng = Cp_eng + n_moles * R_UNIVERSAL_ENG * dlnVqdlnT**2 / dlnVqdlnP

        # Gamma
        gamma = -Cp_eng / Cv_eng / dlnVqdlnP

        # Density (kg/m^3) - directly in SI
        rho = P_si / (n_moles * R_UNIVERSAL_SI * T_si)

        # Gas constant (J/(kg*K))
        R_gas = R_UNIVERSAL_SI * n_moles

        # Convert to SI units
        h = h_eng * CAL_G_TO_J_KG
        S = S_eng * CAL_G_TO_J_KG
        Cp = Cp_eng * CAL_G_TO_J_KG
        Cv = Cv_eng * CAL_G_TO_J_KG

        return TotalProps(h=h, S=S, gamma=gamma, Cp=Cp, Cv=Cv, rho=rho, R=R_gas)

    # =========================================================================
    # Public API - Total property calculations
    # =========================================================================

    def set_total_TP(self, T, P):
        """Compute all thermodynamic properties from T and P."""
        # Convert inputs to SI
        T_si = self._convert_T_to_si(T)
        P_si = P * self._P_to_si

        n, pi, n_moles = self._solve_equilibrium(T_si, P_si)
        lhs_TP, rhs_T, rhs_P = self._build_props_matrices(T_si, n, n_moles)

        result_T = solve(lhs_TP, rhs_T)
        result_P = solve(lhs_TP, rhs_P)

        props_si = self._compute_properties_si(T_si, P_si, n, n_moles, result_T, result_P)

        # Convert outputs from SI
        return self._convert_total_props_from_si(props_si)

    def h(self, T, P):
        """Compute enthalpy from T and P."""
        return self.set_total_TP(T, P).h

    def S(self, T, P):
        """Compute entropy from T and P."""
        return self.set_total_TP(T, P).S

    def gamma(self, T, P):
        """Compute ratio of specific heats from T and P."""
        return self.set_total_TP(T, P).gamma

    def Cp(self, T, P):
        """Compute specific heat at constant pressure."""
        return self.set_total_TP(T, P).Cp

    def Cv(self, T, P):
        """Compute specific heat at constant volume."""
        return self.set_total_TP(T, P).Cv

    def rho(self, T, P):
        """Compute density from T and P."""
        return self.set_total_TP(T, P).rho

    def R(self, T, P):
        """Compute specific gas constant."""
        return self.set_total_TP(T, P).R

    # =========================================================================
    # Inverse calculations
    # =========================================================================

    def set_total_hP(self, h_target, P):
        """Solve for temperature given enthalpy and pressure."""
        # Convert inputs to SI
        h_si = h_target * self._h_to_si
        P_si = P * self._P_to_si

        def residual(T_si):
            n, pi, n_moles = self._solve_equilibrium(T_si, P_si)
            lhs_TP, rhs_T, rhs_P = self._build_props_matrices(T_si, n, n_moles)
            result_T = solve(lhs_TP, rhs_T)
            result_P = solve(lhs_TP, rhs_P)
            props = self._compute_properties_si(T_si, P_si, n, n_moles, result_T, result_P)
            return props.h - h_si

        T_min, T_max = 200.0, 6000.0
        T_si = brentq(residual, T_min, T_max, xtol=1e-10)

        # Convert output from SI
        return self._convert_T_from_si(T_si)

    def T_from_SP(self, S_target, P):
        """Solve for temperature given entropy and pressure."""
        # Convert inputs to SI
        S_si = S_target * self._S_to_si
        P_si = P * self._P_to_si

        def residual(T_si):
            n, pi, n_moles = self._solve_equilibrium(T_si, P_si)
            lhs_TP, rhs_T, rhs_P = self._build_props_matrices(T_si, n, n_moles)
            result_T = solve(lhs_TP, rhs_T)
            result_P = solve(lhs_TP, rhs_P)
            props = self._compute_properties_si(T_si, P_si, n, n_moles, result_T, result_P)
            return props.S - S_si

        T_min, T_max = 200.0, 6000.0
        T_si = brentq(residual, T_min, T_max, xtol=1e-10)

        return self._convert_T_from_si(T_si)

    # =========================================================================
    # Static property calculations
    # =========================================================================

    def _set_static_MN_si(self, Tt_si, Pt_si, MN, W_si):
        """Compute static properties in SI units."""
        # Get gamma and R at total conditions (in SI)
        n, pi, n_moles = self._solve_equilibrium(Tt_si, Pt_si)
        lhs_TP, rhs_T, rhs_P = self._build_props_matrices(Tt_si, n, n_moles)
        result_T = solve(lhs_TP, rhs_T)
        result_P = solve(lhs_TP, rhs_P)
        props = self._compute_properties_si(Tt_si, Pt_si, n, n_moles, result_T, result_P)

        gam = props.gamma
        R_gas = props.R

        # Isentropic relations
        temp_ratio = 1.0 / (1.0 + (gam - 1.0) / 2.0 * MN**2)
        Ts = Tt_si * temp_ratio
        Ps = Pt_si * temp_ratio**(gam / (gam - 1.0))

        # Full static properties at static T and P
        n_s, pi_s, n_moles_s = self._solve_equilibrium(Ts, Ps)
        lhs_TP_s, rhs_T_s, rhs_P_s = self._build_props_matrices(Ts, n_s, n_moles_s)
        result_T_s = solve(lhs_TP_s, rhs_T_s)
        result_P_s = solve(lhs_TP_s, rhs_P_s)
        props_s = self._compute_properties_si(Ts, Ps, n_s, n_moles_s, result_T_s, result_P_s)

        # Speed of sound and velocity (use static gamma and R)
        Vsonic = np.sqrt(props_s.gamma * props_s.R * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps / (props_s.R * Ts)

        # Area from continuity
        area = W_si / (rhos * V) if V > 0 else np.inf

        return StaticProps(Ts=Ts, Ps=Ps, hs=props_s.h, rhos=rhos,
                          MN=MN, V=V, Vsonic=Vsonic, area=area,
                          gamma=props_s.gamma, Cp=props_s.Cp, Cv=props_s.Cv,
                          S=props_s.S, R=props_s.R)

    def set_static_MN(self, Tt, Pt, MN, W):
        """Compute static properties from total conditions and Mach number."""
        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        W_si = W * self._W_to_si

        props_si = self._set_static_MN_si(Tt_si, Pt_si, MN, W_si)

        # Convert outputs from SI
        return self._convert_static_props_from_si(props_si)

    def set_static_area(self, Tt, Pt, area, W, MN_guess=0.5, subsonic=True):
        """Compute static properties from total conditions and flow area."""
        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        area_si = area * self._area_to_si
        W_si = W * self._W_to_si

        def area_residual(MN):
            props = self._set_static_MN_si(Tt_si, Pt_si, MN, W_si)
            return props.area - area_si

        if subsonic:
            MN_min, MN_max = 0.01, 0.999
        else:
            MN_min, MN_max = 1.001, 5.0

        MN = brentq(area_residual, MN_min, MN_max, xtol=1e-10)
        props_si = self._set_static_MN_si(Tt_si, Pt_si, MN, W_si)

        return self._convert_static_props_from_si(props_si)

    def static_from_Ps(self, Tt, Pt, Ps, W):
        """Compute static properties from total conditions and static pressure."""
        # Convert inputs to SI
        Tt_si = self._convert_T_to_si(Tt)
        Pt_si = Pt * self._P_to_si
        Ps_si = Ps * self._P_to_si
        W_si = W * self._W_to_si

        # Get gamma at total conditions for isentropic relation
        n, pi, n_moles = self._solve_equilibrium(Tt_si, Pt_si)
        lhs_TP, rhs_T, rhs_P = self._build_props_matrices(Tt_si, n, n_moles)
        result_T = solve(lhs_TP, rhs_T)
        result_P = solve(lhs_TP, rhs_P)
        props = self._compute_properties_si(Tt_si, Pt_si, n, n_moles, result_T, result_P)

        gam = props.gamma

        # Isentropic relation for temperature
        pressure_ratio = Ps_si / Pt_si
        Ts = Tt_si * pressure_ratio**((gam - 1.0) / gam)

        # Full static properties at static T and P
        n_s, pi_s, n_moles_s = self._solve_equilibrium(Ts, Ps_si)
        lhs_TP_s, rhs_T_s, rhs_P_s = self._build_props_matrices(Ts, n_s, n_moles_s)
        result_T_s = solve(lhs_TP_s, rhs_T_s)
        result_P_s = solve(lhs_TP_s, rhs_P_s)
        props_s = self._compute_properties_si(Ts, Ps_si, n_s, n_moles_s, result_T_s, result_P_s)

        # Mach number from temperature ratio
        temp_ratio = Ts / Tt_si
        MN_sq = 2.0 / (gam - 1.0) * (1.0 / temp_ratio - 1.0)
        MN = np.sqrt(max(0.0, MN_sq))

        # Speed of sound and velocity (use static properties)
        Vsonic = np.sqrt(props_s.gamma * props_s.R * Ts)
        V = MN * Vsonic

        # Density
        rhos = Ps_si / (props_s.R * Ts)

        # Area from continuity
        area = W_si / (rhos * V) if V > 0 else np.inf

        props_si = StaticProps(Ts=Ts, Ps=Ps_si, hs=props_s.h, rhos=rhos,
                              MN=MN, V=V, Vsonic=Vsonic, area=area,
                              gamma=props_s.gamma, Cp=props_s.Cp, Cv=props_s.Cv,
                              S=props_s.S, R=props_s.R)

        return self._convert_static_props_from_si(props_si)

    # =========================================================================
    # Linearization and derivatives
    # =========================================================================

    def linearize(self, T, P):
        """
        Compute and cache property gradients at the given state.

        Uses the implicit function theorem to compute total derivatives
        that account for equilibrium composition changes.
        """
        # Convert inputs to SI
        T_si = self._convert_T_to_si(T)
        P_si = P * self._P_to_si

        # Solve equilibrium
        n, pi, n_moles = self._solve_equilibrium(T_si, P_si)

        # Build props matrices
        lhs_TP, rhs_T, rhs_P = self._build_props_matrices(T_si, n, n_moles)
        result_T = solve(lhs_TP, rhs_T)
        result_P = solve(lhs_TP, rhs_P)

        # Store linearization point
        self._lin_T = T_si
        self._lin_P = P_si

        # Get thermodynamic data
        T_arr = self._wrap_T(T_si)
        P_bar = P_si / 100000.0
        H0 = self._props.H0(T_arr)
        S0 = self._props.S0(T_arr)
        Cp0 = self._props.Cp0(T_arr)
        dH0_dT = self._props.H0_applyJ(T_arr, 1.0)
        dS0_dT = self._props.S0_applyJ(T_arr, 1.0)

        # Build state Jacobian for implicit function theorem
        J_state = self._compute_jacobian(n, pi, n_moles, remove_trace=False)

        num_prod = self.num_prod
        num_element = self.num_element
        size = num_prod + num_element

        # dR/dT
        dR_dT = np.zeros(size)
        if self._use_trace_damping:
            weights = _resid_weighting(n)
            dR_dT[:num_prod] = (dH0_dT - dS0_dT) * weights
        else:
            dR_dT[:num_prod] = dH0_dT - dS0_dT

        # dR/dP
        P_norm = P_bar / P_REF
        qP = 1.0 / P_REF / P_norm / 100000.0
        dR_dP = np.zeros(size)
        if self._use_trace_damping:
            dR_dP[:num_prod] = weights * qP
        else:
            dR_dP[:num_prod] = qP

        # Solve implicit function theorem
        dstate_dT = solve(J_state, -dR_dT)
        dstate_dP = solve(J_state, -dR_dP)

        dn_dT = dstate_dT[:num_prod]
        dn_dP = dstate_dP[:num_prod]
        dn_moles_dT = np.sum(dn_dT)
        dn_moles_dP = np.sum(dn_dP)

        # Compute property partials and total derivatives (in SI)
        self._gradients_si = {}

        # Enthalpy
        n_H0 = n * H0
        dh_dn = R_UNIVERSAL_ENG * T_si * H0
        dh_dT_partial = R_UNIVERSAL_ENG * (np.sum(n * dH0_dT) * T_si + np.sum(n * H0))
        dh_dT_si = (dh_dT_partial + dh_dn @ dn_dT) * CAL_G_TO_J_KG
        dh_dP_si = (dh_dn @ dn_dP) * CAL_G_TO_J_KG
        self._gradients_si['h'] = (dh_dT_si, dh_dP_si)

        # Entropy
        old = np.seterr(divide='ignore')
        try:
            log_term = np.log(n_moles / n / (P_bar / P_REF))
        finally:
            np.seterr(**old)

        dS_dn = R_UNIVERSAL_ENG * (S0 + np.log(n_moles) - np.log(P_bar / P_REF) - np.log(n) - 1)
        trace_mask = n <= MIN_VALID_CONCENTRATION + 1e-20
        dS_dn[trace_mask] = 0.0
        dS_dT_partial = R_UNIVERSAL_ENG * np.sum(n * dS0_dT)
        dS_dP_partial = -R_UNIVERSAL_ENG * np.sum(n) / P_si
        dS_dn_moles = R_UNIVERSAL_ENG * np.sum(n) / n_moles

        dS_dT_si = (dS_dT_partial + dS_dn @ dn_dT + dS_dn_moles * dn_moles_dT) * CAL_G_TO_J_KG
        dS_dP_si = (dS_dP_partial + dS_dn @ dn_dP + dS_dn_moles * dn_moles_dP) * CAL_G_TO_J_KG
        self._gradients_si['S'] = (dS_dT_si, dS_dP_si)

        # Density
        drho_dT = -P_si / (n_moles * R_UNIVERSAL_SI * T_si**2)
        drho_dP_partial = 1.0 / (n_moles * R_UNIVERSAL_SI * T_si)
        drho_dn_moles = -P_si / (n_moles**2 * R_UNIVERSAL_SI * T_si)

        drho_dT_si = drho_dT + drho_dn_moles * dn_moles_dT
        drho_dP_si = drho_dP_partial + drho_dn_moles * dn_moles_dP
        self._gradients_si['rho'] = (drho_dT_si, drho_dP_si)

        # R_gas
        dR_dT_si = R_UNIVERSAL_SI * dn_moles_dT
        dR_dP_si = R_UNIVERSAL_SI * dn_moles_dP
        self._gradients_si['R'] = (dR_dT_si, dR_dP_si)

        # TODO: Replace finite difference with analytical derivatives for Cp, Cv, gamma.
        # Currently using FD because these properties involve second derivatives of the
        # Gibbs function which are complex to differentiate through the equilibrium solver.
        # This is expensive: 4 calls to _solve_equilibrium (T±eps, P±eps), each a Newton iteration.
        # The h, S, rho, R derivatives above are computed analytically via implicit function theorem.
        eps = 1e-6
        props_base = self._compute_properties_si(T_si, P_si, n, n_moles, result_T, result_P)

        n_Tp, _, n_moles_Tp = self._solve_equilibrium(T_si + eps, P_si)
        lhs_Tp, rhs_T_Tp, rhs_P_Tp = self._build_props_matrices(T_si + eps, n_Tp, n_moles_Tp)
        props_T_plus = self._compute_properties_si(T_si + eps, P_si, n_Tp, n_moles_Tp,
                                                   solve(lhs_Tp, rhs_T_Tp), solve(lhs_Tp, rhs_P_Tp))

        n_Tm, _, n_moles_Tm = self._solve_equilibrium(T_si - eps, P_si)
        lhs_Tm, rhs_T_Tm, rhs_P_Tm = self._build_props_matrices(T_si - eps, n_Tm, n_moles_Tm)
        props_T_minus = self._compute_properties_si(T_si - eps, P_si, n_Tm, n_moles_Tm,
                                                    solve(lhs_Tm, rhs_T_Tm), solve(lhs_Tm, rhs_P_Tm))

        n_Pp, _, n_moles_Pp = self._solve_equilibrium(T_si, P_si + eps * P_si)
        lhs_Pp, rhs_T_Pp, rhs_P_Pp = self._build_props_matrices(T_si, n_Pp, n_moles_Pp)
        props_P_plus = self._compute_properties_si(T_si, P_si + eps * P_si, n_Pp, n_moles_Pp,
                                                   solve(lhs_Pp, rhs_T_Pp), solve(lhs_Pp, rhs_P_Pp))

        n_Pm, _, n_moles_Pm = self._solve_equilibrium(T_si, P_si - eps * P_si)
        lhs_Pm, rhs_T_Pm, rhs_P_Pm = self._build_props_matrices(T_si, n_Pm, n_moles_Pm)
        props_P_minus = self._compute_properties_si(T_si, P_si - eps * P_si, n_Pm, n_moles_Pm,
                                                    solve(lhs_Pm, rhs_T_Pm), solve(lhs_Pm, rhs_P_Pm))

        dCp_dT_si = (props_T_plus.Cp - props_T_minus.Cp) / (2 * eps)
        dCp_dP_si = (props_P_plus.Cp - props_P_minus.Cp) / (2 * eps * P_si)
        self._gradients_si['Cp'] = (dCp_dT_si, dCp_dP_si)

        dCv_dT_si = (props_T_plus.Cv - props_T_minus.Cv) / (2 * eps)
        dCv_dP_si = (props_P_plus.Cv - props_P_minus.Cv) / (2 * eps * P_si)
        self._gradients_si['Cv'] = (dCv_dT_si, dCv_dP_si)

        dgamma_dT_si = (props_T_plus.gamma - props_T_minus.gamma) / (2 * eps)
        dgamma_dP_si = (props_P_plus.gamma - props_P_minus.gamma) / (2 * eps * P_si)
        self._gradients_si['gamma'] = (dgamma_dT_si, dgamma_dP_si)

    def jvp(self, T_dot, P_dot):
        """
        Compute Jacobian-vector product (forward-mode autodiff).

        Must call linearize() first.
        """
        if not hasattr(self, '_gradients_si'):
            raise RuntimeError("Must call linearize() before jvp()")

        result = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            dprop_dT_si, dprop_dP_si = self._gradients_si[prop]

            # Convert gradient to input units
            # d(output)/d(input) = d(output_si)/d(input_si) * (d(input_si)/d(input)) / (d(output_si)/d(output))
            if prop == 'gamma':
                out_factor = 1.0
            elif prop in ('S', 'Cp', 'Cv', 'R'):
                out_factor = self._S_from_si
            elif prop == 'h':
                out_factor = self._h_from_si
            elif prop == 'rho':
                out_factor = self._rho_from_si

            dprop_dT = dprop_dT_si * self._T_to_si * out_factor
            dprop_dP = dprop_dP_si * self._P_to_si * out_factor

            result[prop] = dprop_dT * T_dot + dprop_dP * P_dot

        return result

    def vjp(self, h_bar=0.0, S_bar=0.0, gamma_bar=0.0, Cp_bar=0.0,
            Cv_bar=0.0, rho_bar=0.0, R_bar=0.0):
        """
        Compute vector-Jacobian product (reverse-mode autodiff).

        Must call linearize() first.
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
                dprop_dT_si, dprop_dP_si = self._gradients_si[prop]

                if prop == 'gamma':
                    out_factor = 1.0
                elif prop in ('S', 'Cp', 'Cv', 'R'):
                    out_factor = self._S_from_si
                elif prop == 'h':
                    out_factor = self._h_from_si
                elif prop == 'rho':
                    out_factor = self._rho_from_si

                dprop_dT = dprop_dT_si * self._T_to_si * out_factor
                dprop_dP = dprop_dP_si * self._P_to_si * out_factor

                T_bar += dprop_dT * cotan
                P_bar += dprop_dP * cotan

        return T_bar, P_bar
