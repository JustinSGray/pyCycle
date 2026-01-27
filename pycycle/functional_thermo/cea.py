"""
CEA thermodynamic property calculations using Gibbs free energy minimization.

This module provides a functional interface for computing thermodynamic properties
using NASA CEA (Chemical Equilibrium with Applications) methodology.
"""

import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq

from .base import ThermoInterface, TotalProps, StaticProps
from pycycle.constants import (P_REF, R_UNIVERSAL_ENG, R_UNIVERSAL_SI,
                                MIN_VALID_CONCENTRATION, CEA_AIR_COMPOSITION)
from pycycle.thermo.cea import species_data


# Unit conversion: cal/g to J/kg
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
    """

    def __init__(self, composition=None, thermo_data=None):
        if composition is None:
            composition = CEA_AIR_COMPOSITION
        super().__init__(composition)

        if thermo_data is None:
            thermo_data = species_data.janaf

        # Create the Properties object that handles NASA polynomials
        self._props = species_data.Properties(thermo_data, init_elements=composition)

        # Store frequently used attributes
        self.num_prod = self._props.num_prod
        self.num_element = self._props.num_element
        self.aij = self._props.aij
        self.b0 = self._props.b0

        # Solver settings (match OpenMDAO exactly)
        self._atol = 1e-7
        self._rtol = 1e-7
        self._maxiter = 100
        self._stall_limit = 4
        self._stall_tol = 1e-10

        # Initial guess for species concentrations
        # Use same uniform initial guess as OpenMDAO: np.ones(num_prod) / num_prod / 10
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

    def _compute_initial_guess(self):
        """
        Compute a smart initial guess that approximately satisfies mass conservation.

        For each element, allocate to the most stable species (diatomic preferred).
        For elements without pure species, allocate to the simplest compound.
        """
        n_init = np.full(self.num_prod, MIN_VALID_CONCENTRATION)

        b0_remaining = self.b0.copy()

        # First pass: allocate to pure diatomic species (N2, O2) and monatomic (Ar)
        for i in range(self.num_element):
            species_with_element = np.where(self.aij[i, :] > 0)[0]

            # Find pure species with highest stoichiometry (diatomic > monatomic)
            best_species = None
            best_stoich = 0

            for j in species_with_element:
                num_elements_in_species = np.sum(self.aij[:, j] > 0)
                stoich = self.aij[i, j]

                if num_elements_in_species == 1 and stoich > best_stoich:
                    best_species = j
                    best_stoich = stoich

            if best_species is not None and b0_remaining[i] > MIN_VALID_CONCENTRATION:
                n_init[best_species] = b0_remaining[i] / best_stoich
                b0_remaining[i] = 0

        # Second pass: for elements without pure species (like C), use compound species
        # Be careful to account for the oxygen already consumed
        for i in range(self.num_element):
            if b0_remaining[i] > MIN_VALID_CONCENTRATION:
                species_with_element = np.where(self.aij[i, :] > 0)[0]

                # Find species with smallest number of elements
                best_species = None
                best_num_elem = np.inf

                for j in species_with_element:
                    num_elements = np.sum(self.aij[:, j] > 0)
                    if num_elements < best_num_elem:
                        best_species = j
                        best_num_elem = num_elements

                if best_species is not None:
                    stoich = self.aij[i, best_species]
                    n_init[best_species] = max(n_init[best_species], b0_remaining[i] / stoich)

                    # Reduce other elements consumed by this species
                    for k in range(self.num_element):
                        if k != i and self.aij[k, best_species] > 0:
                            consumed = n_init[best_species] * self.aij[k, best_species]
                            # Find the pure species for element k and reduce its concentration
                            for j2 in np.where(self.aij[k, :] > 0)[0]:
                                if np.sum(self.aij[:, j2] > 0) == 1 and n_init[j2] > consumed:
                                    n_init[j2] -= consumed / self.aij[k, j2]
                                    break

        return n_init

    def _solve_equilibrium(self, T, P):
        """
        Solve chemical equilibrium using Newton iteration with bounds-enforcing line search.

        Parameters
        ----------
        T : float
            Temperature (K)
        P : float
            Pressure (Pa)

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
            np.isclose(T, self._cached_T, rtol=1e-12) and
            np.isclose(P, self._cached_P, rtol=1e-12)):
            return self._cached_n.copy(), self._cached_pi.copy(), self._cached_n_moles

        # Convert pressure from Pa to bar, then normalize by P_REF
        P_bar = P / 100000.0
        P_norm = P_bar / P_REF

        # Get thermodynamic properties at this temperature
        T_arr = self._wrap_T(T)
        H0 = self._props.H0(T_arr)
        S0 = self._props.S0(T_arr)

        # Start with initial guess (same as OpenMDAO)
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

            # Enable trace removal when close to convergence (like OpenMDAO)
            # Set flag based on residual norm, then zero out trace species residuals
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
                # If singular, try with regularization
                J_reg = J + 1e-10 * np.eye(J.shape[0])
                delta = solve(J_reg, -resids)

            # Bounds-enforcing update (scalar mode like OpenMDAO BoundsEnforceLS)
            dn = delta[:self.num_prod]
            dpi = delta[self.num_prod:]

            # Take full Newton step then clip each variable independently
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
        self._cached_T = T
        self._cached_P = P
        self._cached_n = n.copy()
        self._cached_pi = pi.copy()
        self._cached_n_moles = np.sum(n)

        return n, pi, self._cached_n_moles

    def _compute_residuals(self, n, pi, H0, S0, P_norm, n_moles):
        """Compute equilibrium residual equations."""
        num_prod = self.num_prod
        num_element = self.num_element

        resids = np.zeros(num_prod + num_element)

        # Chemical potential: mu = H0 - S0 + ln(n) + ln(P) - ln(n_moles)
        # Note: H0 and S0 from CEA are already dimensionless (divided by RT and R)
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            log_n = np.log(np.maximum(n, MIN_VALID_CONCENTRATION))
            mu = H0 - S0 + log_n + np.log(P_norm) - np.log(n_moles)
        finally:
            np.seterr(**old)

        # Species equilibrium residuals: R_n[j] = mu[j] - sum_i(pi[i] * aij[i,j])
        resids[:num_prod] = mu - self.aij.T @ pi

        # Apply trace damping (use n * n_moles like OpenMDAO)
        if self._use_trace_damping:
            weights = _resid_weighting(n * n_moles)
            resids[:num_prod] *= weights

        # Mass conservation residuals: R_pi[i] = sum_j(aij[i,j] * n[j]) - b0[i]
        resids[num_prod:] = self.aij @ n - self.b0

        return resids

    def _compute_jacobian(self, n, pi, n_moles, remove_trace=False):
        """Compute Jacobian matrix for Newton solver."""
        num_prod = self.num_prod
        num_element = self.num_element
        size = num_prod + num_element

        J = np.zeros((size, size))

        # dR_n/dn: diagonal + off-diagonal terms
        MW = 1.0 / n_moles
        J[:num_prod, :num_prod] = -MW
        diag = 1.0 / n - MW
        np.fill_diagonal(J[:num_prod, :num_prod], diag)

        # Apply trace damping to dR_n/dn (use n * n_moles like OpenMDAO)
        if self._use_trace_damping:
            weights = _resid_weighting(n * n_moles)
            J[:num_prod, :num_prod] *= weights[:, np.newaxis]

        # dR_n/dpi
        J[:num_prod, num_prod:] = -self.aij.T
        if self._use_trace_damping:
            J[:num_prod, num_prod:] *= weights[:, np.newaxis]

        # dR_pi/dn
        J[num_prod:, :num_prod] = self.aij

        # dR_pi/dpi = 0 (already zero)

        # Handle trace species removal
        if remove_trace:
            for j in range(num_prod):
                if n[j] <= MIN_VALID_CONCENTRATION + 1e-20:
                    J[j, :] = 0.0
                    J[j, j] = -1.0

        return J

    def _build_props_matrices(self, T, n, n_moles):
        """Build matrices for property calculations (from PropsRHS)."""
        num_element = self.num_element
        ne1 = num_element + 1

        T_arr = self._wrap_T(T)
        H0 = self._props.H0(T_arr)

        # lhs_TP matrix
        lhs_TP = np.zeros((ne1, ne1))
        for i in range(num_element):
            lhs_TP[i, :num_element] = np.dot(self._props.aij_prod[i], n)

        lhs_TP[num_element, :num_element] = self.b0
        lhs_TP[:num_element, num_element] = self.b0
        lhs_TP[num_element, num_element] = 0.0

        # rhs_T
        n_H0 = n * H0
        rhs_T = np.zeros(ne1)
        rhs_T[:num_element] = np.sum(self.aij * n_H0, axis=1)
        rhs_T[num_element] = np.sum(n_H0)

        # rhs_P
        rhs_P = np.zeros(ne1)
        rhs_P[:num_element] = self.b0
        rhs_P[num_element] = n_moles

        return lhs_TP, rhs_T, rhs_P

    def _compute_properties_internal(self, T, P, n, n_moles, result_T, result_P):
        """Compute thermodynamic properties (from PropsCalcs)."""
        T_arr = self._wrap_T(T)
        P_bar = P / 100000.0

        H0 = self._props.H0(T_arr)
        S0 = self._props.S0(T_arr)
        Cp0 = self._props.Cp0(T_arr)

        num_element = self.num_element

        # dlnV/dlnP and dlnV/dlnT
        dlnVqdlnP = -1.0 + result_P[num_element]
        dlnVqdlnT = 1.0 - result_T[num_element]

        # Frozen and equilibrium Cp contributions
        Cpf = np.sum(n * Cp0)
        n_H0 = n * H0

        Cpe = -np.sum(np.sum(self.aij * n_H0, axis=1) * result_T[:num_element])
        Cpe += np.sum(n_H0 * H0)  # n * H0^2
        Cpe -= np.sum(n_H0) * result_T[num_element]

        # Enthalpy (cal/g)
        h_eng = np.sum(n_H0) * R_UNIVERSAL_ENG * T

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
        rho = P / (n_moles * R_UNIVERSAL_SI * T)

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

    def props_TP(self, T, P):
        """Compute all thermodynamic properties from T and P."""
        n, pi, n_moles = self._solve_equilibrium(T, P)
        lhs_TP, rhs_T, rhs_P = self._build_props_matrices(T, n, n_moles)

        result_T = solve(lhs_TP, rhs_T)
        result_P = solve(lhs_TP, rhs_P)

        return self._compute_properties_internal(T, P, n, n_moles, result_T, result_P)

    def h(self, T, P):
        """Compute enthalpy (J/kg) from T (K) and P (Pa)."""
        return self.props_TP(T, P).h

    def S(self, T, P):
        """Compute entropy (J/(kg*K)) from T (K) and P (Pa)."""
        return self.props_TP(T, P).S

    def gamma(self, T, P):
        """Compute ratio of specific heats from T (K) and P (Pa)."""
        return self.props_TP(T, P).gamma

    def Cp(self, T, P):
        """Compute specific heat at constant pressure (J/(kg*K))."""
        return self.props_TP(T, P).Cp

    def Cv(self, T, P):
        """Compute specific heat at constant volume (J/(kg*K))."""
        return self.props_TP(T, P).Cv

    def rho(self, T, P):
        """Compute density (kg/m^3) from T (K) and P (Pa)."""
        return self.props_TP(T, P).rho

    def R(self, T, P):
        """Compute specific gas constant (J/(kg*K))."""
        return self.props_TP(T, P).R

    # =========================================================================
    # Inverse calculations
    # =========================================================================

    def T_from_hP(self, h_target, P):
        """Solve for temperature given enthalpy and pressure."""
        def residual(T):
            return self.h(T, P) - h_target

        T_min, T_max = 200.0, 6000.0
        return brentq(residual, T_min, T_max, xtol=1e-10)

    def T_from_SP(self, S_target, P):
        """Solve for temperature given entropy and pressure."""
        def residual(T):
            return self.S(T, P) - S_target

        T_min, T_max = 200.0, 6000.0
        return brentq(residual, T_min, T_max, xtol=1e-10)

    # =========================================================================
    # Static property calculations
    # =========================================================================

    def static_from_MN(self, Tt, Pt, MN, W):
        """Compute static properties from total conditions and Mach number."""
        gam = self.gamma(Tt, Pt)
        R_gas = self.R(Tt, Pt)

        # Isentropic relations
        temp_ratio = 1.0 / (1.0 + (gam - 1.0) / 2.0 * MN**2)
        Ts = Tt * temp_ratio
        Ps = Pt * temp_ratio**(gam / (gam - 1.0))

        # Static enthalpy
        hs = self.h(Ts, Ps)

        # Speed of sound and velocity
        Vsonic = np.sqrt(gam * R_gas * Ts)
        V = MN * Vsonic

        # Density from ideal gas law
        rhos = Ps / (R_gas * Ts)

        # Area from continuity
        area = W / (rhos * V) if V > 0 else np.inf

        return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                          MN=MN, V=V, Vsonic=Vsonic, area=area)

    def static_from_area(self, Tt, Pt, area, W, MN_guess=0.5, subsonic=True):
        """Compute static properties from total conditions and flow area."""
        def area_residual(MN):
            props = self.static_from_MN(Tt, Pt, MN, W)
            return props.area - area

        if subsonic:
            MN_min, MN_max = 0.01, 0.999
        else:
            MN_min, MN_max = 1.001, 5.0

        MN = brentq(area_residual, MN_min, MN_max, xtol=1e-10)
        return self.static_from_MN(Tt, Pt, MN, W)

    def static_from_Ps(self, Tt, Pt, Ps, W):
        """Compute static properties from total conditions and static pressure."""
        gam = self.gamma(Tt, Pt)
        R_gas = self.R(Tt, Pt)

        # Isentropic relation for temperature
        pressure_ratio = Ps / Pt
        Ts = Tt * pressure_ratio**((gam - 1.0) / gam)

        # Static enthalpy
        hs = self.h(Ts, Ps)

        # Mach number from temperature ratio
        temp_ratio = Ts / Tt
        MN_sq = 2.0 / (gam - 1.0) * (1.0 / temp_ratio - 1.0)
        MN = np.sqrt(max(0.0, MN_sq))

        # Speed of sound and velocity
        Vsonic = np.sqrt(gam * R_gas * Ts)
        V = MN * Vsonic

        # Density
        rhos = Ps / (R_gas * Ts)

        # Area from continuity
        area = W / (rhos * V) if V > 0 else np.inf

        return StaticProps(Ts=Ts, Ps=Ps, hs=hs, rhos=rhos,
                          MN=MN, V=V, Vsonic=Vsonic, area=area)

    # =========================================================================
    # Linearization and derivatives
    # =========================================================================

    def linearize(self, T, P):
        """
        Compute and cache property gradients at the given state.

        Uses the implicit function theorem to compute total derivatives
        that account for equilibrium composition changes.
        """
        # Solve equilibrium
        n, pi, n_moles = self._solve_equilibrium(T, P)

        # Build props matrices
        lhs_TP, rhs_T, rhs_P = self._build_props_matrices(T, n, n_moles)
        result_T = solve(lhs_TP, rhs_T)
        result_P = solve(lhs_TP, rhs_P)

        # Store linearization point
        self._lin_T = T
        self._lin_P = P

        # Get thermodynamic data
        T_arr = self._wrap_T(T)
        P_bar = P / 100000.0
        H0 = self._props.H0(T_arr)
        S0 = self._props.S0(T_arr)
        Cp0 = self._props.Cp0(T_arr)
        dH0_dT = self._props.H0_applyJ(T_arr, 1.0)
        dS0_dT = self._props.S0_applyJ(T_arr, 1.0)

        # Build state Jacobian for implicit function theorem
        J_state = self._compute_jacobian(n, pi, n_moles, remove_trace=False)

        # Build input partial derivatives
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
        qP = 1.0 / P_REF / P_norm / 100000.0  # Convert to per-Pa
        dR_dP = np.zeros(size)
        if self._use_trace_damping:
            dR_dP[:num_prod] = weights * qP
        else:
            dR_dP[:num_prod] = qP

        # Solve implicit function theorem: d(state)/dT = -J^(-1) @ dR/dT
        dstate_dT = solve(J_state, -dR_dT)
        dstate_dP = solve(J_state, -dR_dP)

        dn_dT = dstate_dT[:num_prod]
        dn_dP = dstate_dP[:num_prod]
        dn_moles_dT = np.sum(dn_dT)
        dn_moles_dP = np.sum(dn_dP)

        # Compute property partials and total derivatives
        self._gradients = {}

        # Enthalpy: h = R * T * sum(n * H0)
        n_H0 = n * H0
        h_eng = np.sum(n_H0) * R_UNIVERSAL_ENG * T

        # dh/dn = R * T * H0
        dh_dn = R_UNIVERSAL_ENG * T * H0
        # dh/dT|_n = R * (sum(n * dH0_dT) * T + sum(n * H0))
        dh_dT_partial = R_UNIVERSAL_ENG * (np.sum(n * dH0_dT) * T + np.sum(n * H0))
        # Total derivatives
        dh_dT = (dh_dT_partial + dh_dn @ dn_dT) * CAL_G_TO_J_KG
        dh_dP = (dh_dn @ dn_dP) * CAL_G_TO_J_KG
        self._gradients['h'] = (dh_dT, dh_dP)

        # Entropy: S = R * sum(n * (S0 + ln(n_moles/n) - ln(P/P_ref)))
        old = np.seterr(divide='ignore')
        try:
            log_term = np.log(n_moles / n / (P_bar / P_REF))
        finally:
            np.seterr(**old)

        # dS/dn = R * (S0 + ln(n_moles) - ln(P/P_ref) - ln(n) - 1)
        dS_dn = R_UNIVERSAL_ENG * (S0 + np.log(n_moles) - np.log(P_bar / P_REF) - np.log(n) - 1)
        # Zero out trace species
        trace_mask = n <= MIN_VALID_CONCENTRATION + 1e-20
        dS_dn[trace_mask] = 0.0
        # dS/dT|_n = R * sum(n * dS0_dT)
        dS_dT_partial = R_UNIVERSAL_ENG * np.sum(n * dS0_dT)
        # dS/dP|_n = -R * sum(n) / P
        dS_dP_partial = -R_UNIVERSAL_ENG * np.sum(n) / P
        # dS/dn_moles = R * sum(n) / n_moles
        dS_dn_moles = R_UNIVERSAL_ENG * np.sum(n) / n_moles

        dS_dT = (dS_dT_partial + dS_dn @ dn_dT + dS_dn_moles * dn_moles_dT) * CAL_G_TO_J_KG
        dS_dP = (dS_dP_partial + dS_dn @ dn_dP + dS_dn_moles * dn_moles_dP) * CAL_G_TO_J_KG
        self._gradients['S'] = (dS_dT, dS_dP)

        # Density: rho = P / (n_moles * R_SI * T)
        drho_dT = -P / (n_moles * R_UNIVERSAL_SI * T**2)
        drho_dP_partial = 1.0 / (n_moles * R_UNIVERSAL_SI * T)
        drho_dn_moles = -P / (n_moles**2 * R_UNIVERSAL_SI * T)

        drho_dT_total = drho_dT + drho_dn_moles * dn_moles_dT
        drho_dP_total = drho_dP_partial + drho_dn_moles * dn_moles_dP
        self._gradients['rho'] = (drho_dT_total, drho_dP_total)

        # R_gas = R_SI * n_moles
        dR_dT_total = R_UNIVERSAL_SI * dn_moles_dT
        dR_dP_total = R_UNIVERSAL_SI * dn_moles_dP
        self._gradients['R'] = (dR_dT_total, dR_dP_total)

        # For Cp, Cv, gamma - use finite difference for now (complex chain rule)
        eps = 1e-6
        props_base = self.props_TP(T, P)

        # Finite difference for Cp
        props_T_plus = self.props_TP(T + eps, P)
        props_T_minus = self.props_TP(T - eps, P)
        dCp_dT = (props_T_plus.Cp - props_T_minus.Cp) / (2 * eps)

        props_P_plus = self.props_TP(T, P + eps * P)
        props_P_minus = self.props_TP(T, P - eps * P)
        dCp_dP = (props_P_plus.Cp - props_P_minus.Cp) / (2 * eps * P)
        self._gradients['Cp'] = (dCp_dT, dCp_dP)

        # Finite difference for Cv
        dCv_dT = (props_T_plus.Cv - props_T_minus.Cv) / (2 * eps)
        dCv_dP = (props_P_plus.Cv - props_P_minus.Cv) / (2 * eps * P)
        self._gradients['Cv'] = (dCv_dT, dCv_dP)

        # Finite difference for gamma
        dgamma_dT = (props_T_plus.gamma - props_T_minus.gamma) / (2 * eps)
        dgamma_dP = (props_P_plus.gamma - props_P_minus.gamma) / (2 * eps * P)
        self._gradients['gamma'] = (dgamma_dT, dgamma_dP)

    def jvp(self, T_dot, P_dot):
        """
        Compute Jacobian-vector product (forward-mode autodiff).

        Must call linearize() first.
        """
        if not hasattr(self, '_gradients'):
            raise RuntimeError("Must call linearize() before jvp()")

        result = {}
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            dprop_dT, dprop_dP = self._gradients[prop]
            result[prop] = dprop_dT * T_dot + dprop_dP * P_dot

        return result

    def vjp(self, h_bar=0.0, S_bar=0.0, gamma_bar=0.0, Cp_bar=0.0,
            Cv_bar=0.0, rho_bar=0.0, R_bar=0.0):
        """
        Compute vector-Jacobian product (reverse-mode autodiff).

        Must call linearize() first.
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
                dprop_dT, dprop_dP = self._gradients[prop]
                T_bar += dprop_dT * cotan
                P_bar += dprop_dP * cotan

        return T_bar, P_bar


if __name__ == "__main__":
    # Quick test
    print("Testing CEAThermo functional interface")
    print("=" * 50)

    thermo = CEAThermo()

    T = 1500.0  # K
    P = 101325.0  # Pa (1 atm)

    print(f"\nTest conditions: T = {T} K, P = {P/1000:.1f} kPa")

    props = thermo.props_TP(T, P)
    print(f"\nTotal properties:")
    print(f"  h     = {props.h:.2f} J/kg")
    print(f"  S     = {props.S:.2f} J/(kg*K)")
    print(f"  gamma = {props.gamma:.4f}")
    print(f"  Cp    = {props.Cp:.2f} J/(kg*K)")
    print(f"  Cv    = {props.Cv:.2f} J/(kg*K)")
    print(f"  rho   = {props.rho:.4f} kg/m^3")
    print(f"  R     = {props.R:.2f} J/(kg*K)")

    # Test inverse
    T_recovered = thermo.T_from_hP(props.h, P)
    print(f"\nInverse test (T from h, P):")
    print(f"  Original T = {T} K")
    print(f"  Recovered T = {T_recovered:.2f} K")

    # Test linearize
    thermo.linearize(T, P)
    jvp = thermo.jvp(T_dot=1.0, P_dot=0.0)
    print(f"\nJVP (T_dot=1, P_dot=0):")
    print(f"  dh/dT = {jvp['h']:.2f} J/(kg*K)")
