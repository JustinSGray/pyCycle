"""
Pure JAX implementation of CEA (Chemical Equilibrium with Applications) thermodynamics.

This module provides JAX-traceable versions of the CEA thermo operations,
enabling efficient JIT compilation without Python-level iteration overhead.

The key design principle is NO NESTED NEWTON LOOPS: all implicit solves
(hP, SP, static MN, static area) use a single combined Newton system that
simultaneously solves for equilibrium composition AND the target variable(s).
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ..base import TotalProps, StaticProps, StaticPropsWithDeriv
from pycycle.constants import (P_REF, R_UNIVERSAL_ENG, R_UNIVERSAL_SI,
                                MIN_VALID_CONCENTRATION, CEA_AIR_COMPOSITION)
from pycycle.thermo.cea import species_data


# Unit conversion: cal/g to J/kg
CAL_G_TO_J_KG = 4184.0


# =============================================================================
# Data Preprocessing: Convert species_data into JAX-compatible arrays
# =============================================================================

def _preprocess_thermo_data(thermo_data, composition):
    """
    Convert thermo data module and composition into JAX-compatible arrays.

    Uses the existing species_data.Properties class to extract all needed
    data, then converts to JAX arrays with pre-stacked polynomial coefficients
    for all temperature ranges.

    Parameters
    ----------
    thermo_data : module
        Species thermodynamic data module (e.g., species_data.janaf)
    composition : dict
        Elemental composition (e.g., {'N': 0.054, 'O': 0.014, ...})

    Returns
    -------
    dict
        Dictionary of JAX arrays for use in JIT-compiled functions
    """
    props = species_data.Properties(thermo_data, init_elements=composition)

    num_prod = props.num_prod
    num_element = props.num_element
    prod_data = thermo_data.products

    # Find max number of temperature ranges across all products
    max_ranges = 0
    for name in props.products:
        n_ranges = len(prod_data[name]['ranges']) - 1  # number of intervals
        max_ranges = max(max_ranges, n_ranges)

    # Pre-stack coefficients for all ranges: (max_ranges, num_prod, 10)
    # and range boundaries: (num_prod, max_ranges + 1)
    coeffs_all = np.zeros((max_ranges, num_prod, 10))
    ranges_all = np.zeros((num_prod, max_ranges + 1))
    # Fill ranges_all with large sentinels for padding
    ranges_all[:, :] = 1e30

    for i, name in enumerate(props.products):
        tr = prod_data[name]['ranges']
        coeffs = prod_data[name]['coeffs']
        n_ranges = len(tr) - 1

        ranges_all[i, :len(tr)] = tr

        for j in range(n_ranges):
            data = coeffs[j]
            coeffs_all[j, i, :len(data)] = data

    # aij_prod: precomputed aij[i] * aij[j] for each element pair
    # Shape: (num_element, num_element, num_prod)
    aij_prod = np.zeros((num_element, num_element, num_prod))
    for ii in range(num_element):
        for jj in range(num_element):
            aij_prod[ii, jj] = props.aij[ii] * props.aij[jj]

    return {
        'coeffs': jnp.array(coeffs_all),       # (max_ranges, num_prod, 10)
        'ranges': jnp.array(ranges_all),        # (num_prod, max_ranges+1)
        'max_ranges': max_ranges,
        'aij': jnp.array(props.aij),            # (num_element, num_prod)
        'aij_T': jnp.array(props.aij.T),        # (num_prod, num_element)
        'aij_prod': jnp.array(aij_prod),        # (num_element, num_element, num_prod)
        'b0': jnp.array(props.b0),              # (num_element,)
        'num_prod': num_prod,
        'num_element': num_element,
    }


# =============================================================================
# NASA Polynomial Functions (pure JAX)
# =============================================================================

def _select_coefficients(T, coeffs, ranges, max_ranges):
    """
    Select the correct NASA polynomial coefficients for temperature T.

    For each species, picks the coefficient set corresponding to the
    temperature range that contains T. Uses jnp.where for JAX traceability.

    Parameters
    ----------
    T : scalar
        Temperature (K)
    coeffs : array (max_ranges, num_prod, 10)
        Pre-stacked polynomial coefficients for all ranges
    ranges : array (num_prod, max_ranges+1)
        Temperature range boundaries per species
    max_ranges : int
        Maximum number of ranges

    Returns
    -------
    array (num_prod, 10)
        Selected coefficients for each species at temperature T
    """
    # Start with first range coefficients
    selected = coeffs[0]

    # Progressively override with higher range coefficients where T >= boundary
    for j in range(1, max_ranges):
        mask = T >= ranges[:, j]  # (num_prod,) boolean
        selected = jnp.where(mask[:, None], coeffs[j], selected)

    return selected


def _compute_H0(T, a):
    """
    Compute dimensionless standard-state molar enthalpy H0/RT for all species.

    Parameters
    ----------
    T : scalar
        Temperature (K)
    a : array (num_prod, 10)
        NASA polynomial coefficients

    Returns
    -------
    array (num_prod,)
        H0/RT for each species
    """
    a = a.T  # (10, num_prod)
    return (-a[0] / T**2 + a[1] / T * jnp.log(T) + a[2] +
            a[3] * T / 2.0 + a[4] * T**2 / 3.0 + a[5] * T**3 / 4.0 +
            a[6] * T**4 / 5.0 + a[7] / T)


def _compute_S0(T, a):
    """
    Compute dimensionless standard-state molar entropy S0/R for all species.

    Parameters
    ----------
    T : scalar
        Temperature (K)
    a : array (num_prod, 10)
        NASA polynomial coefficients

    Returns
    -------
    array (num_prod,)
        S0/R for each species
    """
    a = a.T  # (10, num_prod)
    return (-a[0] / (2.0 * T**2) - a[1] / T + a[2] * jnp.log(T) +
            a[3] * T + a[4] * T**2 / 2.0 + a[5] * T**3 / 3.0 +
            a[6] * T**4 / 4.0 + a[8])


def _compute_Cp0(T, a):
    """
    Compute dimensionless standard-state molar heat capacity Cp0/R for all species.

    Parameters
    ----------
    T : scalar
        Temperature (K)
    a : array (num_prod, 10)
        NASA polynomial coefficients

    Returns
    -------
    array (num_prod,)
        Cp0/R for each species
    """
    a = a.T  # (10, num_prod)
    return (a[0] / T**2 + a[1] / T + a[2] +
            a[3] * T + a[4] * T**2 + a[5] * T**3 + a[6] * T**4)


def _compute_dH0_dT(T, a):
    """
    Compute d(H0/RT)/dT for all species.

    Parameters
    ----------
    T : scalar
        Temperature (K)
    a : array (num_prod, 10)
        NASA polynomial coefficients

    Returns
    -------
    array (num_prod,)
        d(H0/RT)/dT for each species
    """
    a = a.T  # (10, num_prod)
    return (2.0 * a[0] / T**3 + a[1] * (1.0 - jnp.log(T)) / T**2 +
            a[3] / 2.0 + 2.0 * a[4] / 3.0 * T + 3.0 * a[5] / 4.0 * T**2 +
            4.0 * a[6] / 5.0 * T**3 - a[7] / T**2)


def _compute_dS0_dT(T, a):
    """
    Compute d(S0/R)/dT for all species.

    Parameters
    ----------
    T : scalar
        Temperature (K)
    a : array (num_prod, 10)
        NASA polynomial coefficients

    Returns
    -------
    array (num_prod,)
        d(S0/R)/dT for each species
    """
    a = a.T  # (10, num_prod)
    return (a[0] / T**3 + a[1] / T**2 + a[2] / T +
            a[3] + a[4] * T + a[5] * T**2 + a[6] * T**3)


def _sigmoid_weight(n, n_moles):
    """
    Sigmoid weighting function for trace species damping.

    Maps to range [0, 1], with small values near 0 for trace species
    and values near 1 for significant species.

    Parameters
    ----------
    n : array (num_prod,)
        Species molar concentrations
    n_moles : scalar
        Total molar concentration

    Returns
    -------
    array (num_prod,)
        Sigmoid weights
    """
    return (1.0 / (1.0 + jnp.exp(-1e5 * n * n_moles)) - 0.5) * 2.0


# =============================================================================
# JaxCEAThermo class
# =============================================================================

class JaxCEAThermo:
    """
    Pure JAX implementation of CEA thermodynamics.

    Provides JAX-traceable, JIT-compiled versions of the key thermo operations.
    All Newton solvers use a single combined system (no nested Newton loops).

    Parameters
    ----------
    composition : dict, optional
        Elemental composition. Default is CEA_AIR_COMPOSITION.
    thermo_data : module, optional
        Species thermodynamic data module. Default is janaf.
    """

    def __init__(self, composition=None, thermo_data=None):
        if composition is None:
            composition = CEA_AIR_COMPOSITION
        if thermo_data is None:
            thermo_data = species_data.janaf

        self.composition = composition
        self.thermo_data = thermo_data

        # Preprocess thermo data into JAX arrays
        self._data = _preprocess_thermo_data(thermo_data, composition)

        self.num_prod = self._data['num_prod']
        self.num_element = self._data['num_element']

        # Initial guess for species concentrations
        self._n_init = jnp.ones(self.num_prod) / self.num_prod / 10.0

        # Unit conversion factors (English <-> SI), same as jax_tabular
        h_to_si = 2326.0         # Btu/lbm -> J/kg
        h_from_si = 1.0 / h_to_si
        P_to_si = 6894.76        # psi -> Pa
        T_to_si_scale = 5.0 / 9.0  # Rankine -> Kelvin scale
        S_from_si = 1.0 / 4186.8   # J/(kg*K) -> Btu/(lbm*R)
        rho_from_si = 0.062428    # kg/m^3 -> lbm/ft^3
        self._unit_conversions = (h_to_si, h_from_si, P_to_si,
                                   T_to_si_scale, S_from_si, rho_from_si)

        # Create JIT-compiled functions
        self._setup_jit_functions()
        self._setup_static_functions()

    def _setup_jit_functions(self):
        """Create JIT-compiled versions of total property functions."""
        data = self._data
        coeffs = data['coeffs']
        ranges = data['ranges']
        max_ranges = data['max_ranges']
        aij = data['aij']
        aij_T = data['aij_T']
        aij_prod = data['aij_prod']
        b0 = data['b0']
        num_prod = data['num_prod']
        num_element = data['num_element']
        n_init = self._n_init

        h_to_si, h_from_si, P_to_si, T_to_si_scale, S_from_si, rho_from_si = self._unit_conversions

        # =====================================================================
        # Helper: get polynomial coefficients at temperature T
        # =====================================================================

        def get_coeffs(T):
            return _select_coefficients(T, coeffs, ranges, max_ranges)

        # =====================================================================
        # Helper: compute equilibrium residuals and Jacobian
        # =====================================================================

        def compute_equilibrium_residuals(n, pi, H0, S0, P_norm, n_moles):
            """Compute residuals for the equilibrium system."""
            size = num_prod + num_element

            # Chemical potential
            log_n = jnp.log(jnp.maximum(n, MIN_VALID_CONCENTRATION))
            mu = H0 - S0 + log_n + jnp.log(P_norm) - jnp.log(n_moles)

            # Species residuals (with trace damping)
            resids_n = mu - aij_T @ pi
            weights = _sigmoid_weight(n, n_moles)
            resids_n = resids_n * weights

            # Element conservation residuals
            resids_pi = aij @ n - b0

            return jnp.concatenate([resids_n, resids_pi]), weights

        def compute_equilibrium_jacobian(n, pi, n_moles, weights, remove_trace):
            """Compute Jacobian for the equilibrium Newton system."""
            size = num_prod + num_element
            J = jnp.zeros((size, size))

            MW = 1.0 / n_moles

            # dR_n/dn block: (1/n_j - 1/n_moles) on diagonal, -1/n_moles off-diagonal
            diag = 1.0 / n - MW
            J_nn = jnp.full((num_prod, num_prod), -MW)
            J_nn = J_nn.at[jnp.arange(num_prod), jnp.arange(num_prod)].set(diag)

            # Apply trace damping to species rows
            J_nn = J_nn * weights[:, None]

            # dR_n/dpi block
            J_npi = -aij_T * weights[:, None]

            # dR_pi/dn block
            J_pin = aij

            # Assemble
            J = J.at[:num_prod, :num_prod].set(J_nn)
            J = J.at[:num_prod, num_prod:].set(J_npi)
            J = J.at[num_prod:, :num_prod].set(J_pin)

            # Handle trace species: zero row, set diagonal to -1
            # Only active when remove_trace is True (near convergence)
            trace_mask = (n <= MIN_VALID_CONCENTRATION + 1e-20) & remove_trace
            row_mask = trace_mask[:, None]  # (num_prod, 1)
            J_top = J[:num_prod, :]
            J_top = jnp.where(row_mask, 0.0, J_top)
            trace_diag = jnp.where(trace_mask, -1.0, J_top[jnp.arange(num_prod), jnp.arange(num_prod)])
            J_top = J_top.at[jnp.arange(num_prod), jnp.arange(num_prod)].set(trace_diag)
            J = J.at[:num_prod, :].set(J_top)

            return J

        # =====================================================================
        # Helper: compute thermodynamic properties from equilibrium state
        # =====================================================================

        def compute_properties_si(T_si, P_si, n, n_moles, a):
            """Compute all thermo properties in SI from converged equilibrium."""
            P_bar = P_si / 100000.0

            H0 = _compute_H0(T_si, a)
            S0 = _compute_S0(T_si, a)
            Cp0 = _compute_Cp0(T_si, a)

            # Build the (ne+1 x ne+1) property matrix and RHS
            ne1 = num_element + 1
            lhs_TP = jnp.zeros((ne1, ne1))

            # lhs_TP[i, :ne] = sum_j(aij_prod[i,k,j] * n_j) for each element pair
            for i in range(num_element):
                row = jnp.sum(aij_prod[i] * n[None, :], axis=1)  # (num_element,)
                lhs_TP = lhs_TP.at[i, :num_element].set(row)

            lhs_TP = lhs_TP.at[num_element, :num_element].set(b0)
            lhs_TP = lhs_TP.at[:num_element, num_element].set(b0)

            # RHS for temperature derivative
            n_H0 = n * H0
            rhs_T = jnp.zeros(ne1)
            rhs_T = rhs_T.at[:num_element].set(jnp.sum(aij * n_H0[None, :], axis=1))
            rhs_T = rhs_T.at[num_element].set(jnp.sum(n_H0))

            # RHS for pressure derivative
            rhs_P = jnp.zeros(ne1)
            rhs_P = rhs_P.at[:num_element].set(b0)
            rhs_P = rhs_P.at[num_element].set(n_moles)

            # Solve the linear systems
            result_T = jnp.linalg.solve(lhs_TP, rhs_T)
            result_P = jnp.linalg.solve(lhs_TP, rhs_P)

            # Compute properties
            dlnVqdlnP = -1.0 + result_P[num_element]
            dlnVqdlnT = 1.0 - result_T[num_element]

            Cpf = jnp.sum(n * Cp0)
            Cpe = (-jnp.sum(jnp.sum(aij * n_H0[None, :], axis=1) * result_T[:num_element])
                   + jnp.sum(n_H0 * H0)
                   - jnp.sum(n_H0) * result_T[num_element])

            # Enthalpy (cal/g -> J/kg)
            h_eng = jnp.sum(n_H0) * R_UNIVERSAL_ENG * T_si
            h = h_eng * CAL_G_TO_J_KG

            # Entropy (cal/(g*K) -> J/(kg*K))
            S_term = S0 + jnp.log(jnp.maximum(n_moles / jnp.maximum(n, MIN_VALID_CONCENTRATION) / (P_bar / P_REF), 1e-30))
            S_eng = R_UNIVERSAL_ENG * jnp.sum(n * S_term)
            S = S_eng * CAL_G_TO_J_KG

            # Cp (cal/(g*K) -> J/(kg*K))
            Cp_eng = (Cpe + Cpf) * R_UNIVERSAL_ENG
            Cp = Cp_eng * CAL_G_TO_J_KG

            # Cv
            Cv_eng = Cp_eng + n_moles * R_UNIVERSAL_ENG * dlnVqdlnT**2 / dlnVqdlnP
            Cv = Cv_eng * CAL_G_TO_J_KG

            # Gamma
            gamma = -Cp_eng / Cv_eng / dlnVqdlnP

            # Density (kg/m^3)
            rho = P_si / (n_moles * R_UNIVERSAL_SI * T_si)

            # Gas constant (J/(kg*K))
            R_gas = R_UNIVERSAL_SI * n_moles

            return TotalProps(h=h, S=S, gamma=gamma, Cp=Cp, Cv=Cv, rho=rho, R=R_gas)

        # =====================================================================
        # Equilibrium Newton solver
        # =====================================================================

        def solve_equilibrium(T_si, P_si):
            """Solve chemical equilibrium at given T, P via Newton iteration."""
            P_bar = P_si / 100000.0
            P_norm = P_bar / P_REF

            a = get_coeffs(T_si)
            H0 = _compute_H0(T_si, a)
            S0 = _compute_S0(T_si, a)

            size = num_prod + num_element

            # State vector: [n, pi, residual_norm, iteration]
            state_size = size + 2

            def init_state():
                n = n_init
                pi = jnp.zeros(num_element)
                n_moles = jnp.sum(n)
                resids, _ = compute_equilibrium_residuals(n, pi, H0, S0, P_norm, n_moles)
                resid_norm = jnp.linalg.norm(resids)
                return jnp.concatenate([n, pi, jnp.array([resid_norm, 0.0])])

            def cond_fn(state):
                resid_norm = state[size]
                iteration = state[size + 1]
                return (resid_norm > 1e-7) & (iteration < 100)

            def body_fn(state):
                n = state[:num_prod]
                pi = state[num_prod:size]
                resid_norm = state[size]
                iteration = state[size + 1]

                n_moles = jnp.sum(n)
                resids, weights = compute_equilibrium_residuals(n, pi, H0, S0, P_norm, n_moles)

                # Only remove trace species when close to convergence
                remove_trace = resid_norm < 1e-4
                trace_mask = (n <= MIN_VALID_CONCENTRATION + 1e-20) & remove_trace
                resids = resids.at[:num_prod].set(
                    jnp.where(trace_mask, 0.0, resids[:num_prod])
                )

                J = compute_equilibrium_jacobian(n, pi, n_moles, weights, remove_trace)

                # Newton step
                delta = jnp.linalg.solve(J, -resids)

                dn = delta[:num_prod]
                dpi = delta[num_prod:size]

                n_new = jnp.maximum(n + dn, MIN_VALID_CONCENTRATION)
                pi_new = pi + dpi

                # Recompute residual norm
                n_moles_new = jnp.sum(n_new)
                resids_new, _ = compute_equilibrium_residuals(n_new, pi_new, H0, S0, P_norm, n_moles_new)
                resid_norm_new = jnp.linalg.norm(resids_new)

                return jnp.concatenate([n_new, pi_new,
                                        jnp.array([resid_norm_new, iteration + 1])])

            final_state = jax.lax.while_loop(cond_fn, body_fn, init_state())

            n = final_state[:num_prod]
            pi = final_state[num_prod:size]
            n_moles = jnp.sum(n)
            return n, pi, n_moles

        # =====================================================================
        # set_total_TP: equilibrium solve + property computation
        # =====================================================================

        @jax.jit
        def _set_total_TP_jit(T, P):
            """JIT-compiled set_total_TP."""
            T_si = T * T_to_si_scale
            P_si = P * P_to_si

            n, pi, n_moles = solve_equilibrium(T_si, P_si)
            a = get_coeffs(T_si)
            props_si = compute_properties_si(T_si, P_si, n, n_moles, a)

            return TotalProps(
                h=props_si.h * h_from_si,
                S=props_si.S * S_from_si,
                gamma=props_si.gamma,
                Cp=props_si.Cp * S_from_si,
                Cv=props_si.Cv * S_from_si,
                rho=props_si.rho * rho_from_si,
                R=props_si.R * S_from_si,
            )

        # =====================================================================
        # set_total_hP: combined Newton for (T, n, pi)
        # =====================================================================

        @jax.jit
        def _set_total_hP_jit(h_target, P):
            """JIT-compiled set_total_hP with combined Newton solver."""
            h_target_si = h_target * h_to_si
            P_si = P * P_to_si
            P_bar = P_si / 100000.0
            P_norm = P_bar / P_REF

            # State vector: [T, n, pi, residual_norm, iteration]
            full_size = 1 + num_prod + num_element
            state_size = full_size + 2

            def compute_h_si(T_si, n, a):
                """Compute enthalpy in SI (J/kg) from state."""
                H0 = _compute_H0(T_si, a)
                return jnp.sum(n * H0) * R_UNIVERSAL_ENG * T_si * CAL_G_TO_J_KG

            def compute_residuals_hP(T_si, n, pi):
                """Compute residuals for the combined hP system."""
                a = get_coeffs(T_si)
                H0 = _compute_H0(T_si, a)
                S0 = _compute_S0(T_si, a)
                n_moles = jnp.sum(n)

                # Enthalpy residual
                h_si = jnp.sum(n * H0) * R_UNIVERSAL_ENG * T_si * CAL_G_TO_J_KG
                R_h = h_si - h_target_si

                # Equilibrium residuals
                log_n = jnp.log(jnp.maximum(n, MIN_VALID_CONCENTRATION))
                mu = H0 - S0 + log_n + jnp.log(P_norm) - jnp.log(n_moles)
                resids_n = mu - aij_T @ pi
                weights = _sigmoid_weight(n, n_moles)
                resids_n = resids_n * weights

                # Mass conservation
                resids_pi = aij @ n - b0

                # Handle trace species
                trace_mask = n <= MIN_VALID_CONCENTRATION + 1e-20
                resids_n = jnp.where(trace_mask, 0.0, resids_n)

                return jnp.concatenate([jnp.array([R_h]), resids_n, resids_pi]), weights

            def compute_jacobian_hP(T_si, n, pi, weights):
                """Compute Jacobian for the combined hP system."""
                a = get_coeffs(T_si)
                H0 = _compute_H0(T_si, a)
                dH0_dT = _compute_dH0_dT(T_si, a)
                dS0_dT = _compute_dS0_dT(T_si, a)
                n_moles = jnp.sum(n)

                J = jnp.zeros((full_size, full_size))

                # Row 0: dR_h/dT, dR_h/dn, dR_h/dpi
                dh_dT = (jnp.sum(n * dH0_dT) * T_si + jnp.sum(n * H0)) * R_UNIVERSAL_ENG * CAL_G_TO_J_KG
                dh_dn = R_UNIVERSAL_ENG * T_si * H0 * CAL_G_TO_J_KG  # (num_prod,)
                J = J.at[0, 0].set(dh_dT)
                J = J.at[0, 1:1 + num_prod].set(dh_dn)
                # dR_h/dpi = 0 (enthalpy doesn't depend on Lagrange multipliers)

                # Rows 1..num_prod: equilibrium species residuals
                # dR_n/dT: (dH0_dT - dS0_dT) * weights
                J_n_T = (dH0_dT - dS0_dT) * weights
                J = J.at[1:1 + num_prod, 0].set(J_n_T)

                # dR_n/dn block
                MW = 1.0 / n_moles
                diag = 1.0 / n - MW
                J_nn = jnp.full((num_prod, num_prod), -MW)
                J_nn = J_nn.at[jnp.arange(num_prod), jnp.arange(num_prod)].set(diag)
                J_nn = J_nn * weights[:, None]
                J = J.at[1:1 + num_prod, 1:1 + num_prod].set(J_nn)

                # dR_n/dpi block
                J_npi = -aij_T * weights[:, None]
                J = J.at[1:1 + num_prod, 1 + num_prod:].set(J_npi)

                # Rows num_prod+1..end: mass conservation
                J = J.at[1 + num_prod:, 1:1 + num_prod].set(aij)

                # Handle trace species
                trace_mask = n <= MIN_VALID_CONCENTRATION + 1e-20
                row_indices = jnp.arange(1, 1 + num_prod)
                J_species = J[1:1 + num_prod, :]
                J_species = jnp.where(trace_mask[:, None], 0.0, J_species)
                species_diag = jnp.where(trace_mask, -1.0,
                                         J_species[jnp.arange(num_prod), jnp.arange(num_prod) + 1])
                # The +1 offset is because we skip column 0 (T) when looking at diagonal
                # Actually, trace species diagonal should be at J[j+1, j+1] in full matrix
                # In J_species (which is rows 1..np of J), the diagonal at column j+1 is index j+1
                J_species = J_species.at[jnp.arange(num_prod), jnp.arange(num_prod) + 1].set(species_diag)
                J = J.at[1:1 + num_prod, :].set(J_species)

                return J

            # Initial guess
            T_si_init = jnp.clip(jnp.abs(h_target_si) / 1000.0 + 300.0, 300.0, 3000.0)

            def init_state():
                T_si = T_si_init
                n = n_init
                pi = jnp.zeros(num_element)
                resids, _ = compute_residuals_hP(T_si, n, pi)
                resid_norm = jnp.linalg.norm(resids)
                return jnp.concatenate([jnp.array([T_si]), n, pi,
                                        jnp.array([resid_norm, 0.0])])

            def cond_fn(state):
                resid_norm = state[full_size]
                iteration = state[full_size + 1]
                return (resid_norm > 1e-7) & (iteration < 100)

            def body_fn(state):
                T_si = state[0]
                n = state[1:1 + num_prod]
                pi = state[1 + num_prod:full_size]
                iteration = state[full_size + 1]

                resids, weights = compute_residuals_hP(T_si, n, pi)
                J = compute_jacobian_hP(T_si, n, pi, weights)

                delta = jnp.linalg.solve(J, -resids)

                T_new = jnp.clip(T_si + delta[0], 200.0, 6000.0)
                n_new = jnp.maximum(n + delta[1:1 + num_prod], MIN_VALID_CONCENTRATION)
                pi_new = pi + delta[1 + num_prod:]

                resids_new, _ = compute_residuals_hP(T_new, n_new, pi_new)
                resid_norm_new = jnp.linalg.norm(resids_new)

                return jnp.concatenate([jnp.array([T_new]), n_new, pi_new,
                                        jnp.array([resid_norm_new, iteration + 1])])

            final_state = jax.lax.while_loop(cond_fn, body_fn, init_state())
            T_si = final_state[0]

            return T_si / T_to_si_scale

        # Store references
        self._set_total_TP_jit = _set_total_TP_jit
        self._set_total_hP_jit = _set_total_hP_jit
        self._solve_equilibrium = solve_equilibrium
        self._compute_properties_si = compute_properties_si
        self._get_coeffs = get_coeffs

    def _setup_static_functions(self):
        """Create JIT-compiled versions of static property functions."""
        data = self._data
        coeffs = data['coeffs']
        ranges = data['ranges']
        max_ranges = data['max_ranges']
        aij = data['aij']
        aij_T = data['aij_T']
        aij_prod = data['aij_prod']
        b0 = data['b0']
        num_prod = data['num_prod']
        num_element = data['num_element']
        n_init = self._n_init

        h_to_si, h_from_si, P_to_si, T_to_si_scale, S_from_si, rho_from_si = self._unit_conversions

        solve_equilibrium = self._solve_equilibrium
        compute_properties_si = self._compute_properties_si
        get_coeffs = self._get_coeffs

        # Additional conversion factors
        W_to_si = 0.45359237      # lbm/s -> kg/s
        area_to_si = 0.00064516   # inch^2 -> m^2
        area_from_si = 1.0 / area_to_si
        V_from_si = 3.28084       # m/s -> ft/s

        def convert_static_to_english(Ts_si, Ps_si, hs_si, rho_s, MN, V_s, Vsonic_s,
                                       area_s, gamma_s, Cp_s, Cv_s, S_s, R_s):
            """Convert static properties from SI to English units."""
            return StaticProps(
                Ts=Ts_si / T_to_si_scale,
                Ps=Ps_si / P_to_si,
                hs=hs_si * h_from_si,
                rhos=rho_s * rho_from_si,
                MN=MN,
                V=V_s * V_from_si,
                Vsonic=Vsonic_s * V_from_si,
                area=area_s * area_from_si,
                gamma=gamma_s,
                Cp=Cp_s * S_from_si,
                Cv=Cv_s * S_from_si,
                S=S_s * S_from_si,
                R=R_s * S_from_si,
            )

        # =====================================================================
        # set_static_MN: isentropic relations + equilibrium at static conditions
        # =====================================================================

        def compute_static_MN_si(Tt_si, Pt_si, MN, W_si):
            """Compute static properties in SI from total conditions and MN."""
            # Get total properties via equilibrium
            n_t, pi_t, n_moles_t = solve_equilibrium(Tt_si, Pt_si)
            a_t = get_coeffs(Tt_si)
            props_t = compute_properties_si(Tt_si, Pt_si, n_t, n_moles_t, a_t)

            gam = props_t.gamma
            R_t = props_t.R

            MN_clamped = jnp.maximum(MN, 1e-10)
            MN_sq = MN_clamped ** 2

            # Isentropic relations using total-condition gamma
            temp_ratio = 1.0 / (1.0 + (gam - 1.0) / 2.0 * MN_sq)
            Ts = Tt_si * temp_ratio
            Ps = Pt_si * temp_ratio ** (gam / (gam - 1.0))

            # Solve equilibrium at static conditions
            n_s, pi_s, n_moles_s = solve_equilibrium(Ts, Ps)
            a_s = get_coeffs(Ts)
            props_s = compute_properties_si(Ts, Ps, n_s, n_moles_s, a_s)

            # Speed of sound and velocity using static properties
            Vsonic = jnp.sqrt(props_s.gamma * props_s.R * Ts)
            V = MN_clamped * Vsonic

            # Density from ideal gas law
            rhos = Ps / (props_s.R * Ts)

            # Area from continuity
            area = W_si / (rhos * V)

            return (Ts, Ps, props_s.h, rhos, MN, V, Vsonic, area,
                    props_s.gamma, props_s.Cp, props_s.Cv, props_s.S, props_s.R,
                    props_t)

        def set_static_MN_impl(Tt, Pt, MN, W):
            """Pure JAX set_static_MN."""
            Tt_si = Tt * T_to_si_scale
            Pt_si = Pt * P_to_si
            W_si = W * W_to_si

            (Ts_si, Ps_si, hs_si, rho_s, MN_out, V, Vsonic, area_si,
             gamma_s, Cp_s, Cv_s, S_s, R_s, props_t) = compute_static_MN_si(
                Tt_si, Pt_si, MN, W_si)

            # Handle zero MN
            is_zero_MN = MN < 1e-10
            Ts_si = jnp.where(is_zero_MN, Tt_si, Ts_si)
            Ps_si = jnp.where(is_zero_MN, Pt_si, Ps_si)
            hs_si = jnp.where(is_zero_MN, props_t.h, hs_si)
            rho_s = jnp.where(is_zero_MN, Pt_si / (props_t.R * Tt_si), rho_s)
            MN_out = jnp.where(is_zero_MN, 0.0, MN)
            V = jnp.where(is_zero_MN, 0.0, V)
            Vsonic = jnp.where(is_zero_MN, jnp.sqrt(props_t.gamma * props_t.R * Tt_si), Vsonic)
            area_si = jnp.where(is_zero_MN, jnp.inf, area_si)
            gamma_s = jnp.where(is_zero_MN, props_t.gamma, gamma_s)
            Cp_s = jnp.where(is_zero_MN, props_t.Cp, Cp_s)
            Cv_s = jnp.where(is_zero_MN, props_t.Cv, Cv_s)
            S_s = jnp.where(is_zero_MN, props_t.S, S_s)
            R_s = jnp.where(is_zero_MN, props_t.R, R_s)

            static = convert_static_to_english(
                Ts_si, Ps_si, hs_si, rho_s, MN_out, V, Vsonic,
                area_si, gamma_s, Cp_s, Cv_s, S_s, R_s
            )

            # darea/dMN via forward-mode autodiff (reverse mode doesn't support while_loop)
            def area_from_MN(mn):
                _, _, _, rho, _, _, _, a, _, _, _, _, _, _ = compute_static_MN_si(
                    Tt_si, Pt_si, mn, W_si)
                return a

            darea_dMN_si = jax.jvp(area_from_MN, (MN,), (jnp.ones_like(MN),))[1]
            darea_dMN = jnp.where(is_zero_MN, -1e30, darea_dMN_si * area_from_si)

            return StaticPropsWithDeriv(
                Ts=static.Ts, Ps=static.Ps, hs=static.hs, rhos=static.rhos,
                MN=static.MN, V=static.V, Vsonic=static.Vsonic, area=static.area,
                gamma=static.gamma, Cp=static.Cp, Cv=static.Cv, S=static.S, R=static.R,
                darea_dMN=darea_dMN,
            )

        # =====================================================================
        # set_static_area: 1D Newton on MN to match target area
        # =====================================================================

        def set_static_area_impl(Tt, Pt, area, W):
            """Pure JAX set_static_area with 1D Newton on MN."""
            Tt_si = Tt * T_to_si_scale
            Pt_si = Pt * P_to_si
            W_si = W * W_to_si
            area_si = area * area_to_si

            def area_residual(MN):
                """Compute area - area_target at given MN."""
                _, _, _, rho_s, _, _, _, a_computed, _, _, _, _, _, _ = \
                    compute_static_MN_si(Tt_si, Pt_si, MN, W_si)
                return a_computed - area_si

            # 1D Newton on MN using forward-mode AD for derivative
            # State: [MN, residual, iteration]
            def cond_fn(state):
                resid = jnp.abs(state[1])
                iteration = state[2]
                return (resid > 1e-10) & (iteration < 50)

            def body_fn(state):
                MN = state[0]
                iteration = state[2]

                # Forward-mode AD: compute residual and its derivative simultaneously
                r, dr = jax.jvp(area_residual, (MN,), (jnp.ones_like(MN),))

                # Newton step with bounds
                dMN = -r / dr
                MN_new = jnp.clip(MN + dMN, 0.01, 0.999)

                r_new = area_residual(MN_new)

                return jnp.array([MN_new, r_new, iteration + 1])

            MN0 = 0.5
            r0 = area_residual(MN0)
            init = jnp.array([MN0, r0, 0.0])

            final = jax.lax.while_loop(cond_fn, body_fn, init)
            MN_final = final[0]

            # Get full static properties at converged MN
            (Ts_si, Ps_si, hs_si, rho_s, _, V, Vsonic, _,
             gamma_s, Cp_s, Cv_s, S_s, R_s, _) = compute_static_MN_si(
                Tt_si, Pt_si, MN_final, W_si)

            area_final = W_si / (rho_s * V)

            return convert_static_to_english(
                Ts_si, Ps_si, hs_si, rho_s, MN_final, V, Vsonic,
                area_final, gamma_s, Cp_s, Cv_s, S_s, R_s
            )

        # JIT compile
        self._set_static_MN_jit = jax.jit(set_static_MN_impl)
        self._set_static_area_jit = jax.jit(set_static_area_impl)

    # =========================================================================
    # Public API
    # =========================================================================

    def set_total_TP(self, T, P, FAR=None):
        """
        Get all thermodynamic properties at given T, P.

        Parameters
        ----------
        T : float
            Temperature (English units: Rankine)
        P : float
            Pressure (English units: psi)
        FAR : ignored
            Accepted for API compatibility with JaxTabularThermo.

        Returns
        -------
        TotalProps
            Named tuple with (h, S, gamma, Cp, Cv, rho, R) in English units
        """
        return self._set_total_TP_jit(T, P)

    def set_total_hP(self, h_target, P, FAR=None):
        """
        Solve for temperature given enthalpy and pressure.

        Uses a combined Newton solver that simultaneously solves for T
        and chemical equilibrium composition (no nested Newton loops).

        Parameters
        ----------
        h_target : float
            Target enthalpy (English units: Btu/lbm)
        P : float
            Pressure (English units: psi)
        FAR : ignored

        Returns
        -------
        float
            Temperature (English units: Rankine)
        """
        return self._set_total_hP_jit(h_target, P)

    def set_static_MN(self, Tt, Pt, MN, W, FAR=None):
        """
        Compute static properties from total conditions and Mach number.

        Uses a combined Newton solver for (Ts, Ps, n, pi) simultaneously.

        Parameters
        ----------
        Tt : float
            Total temperature (English units: Rankine)
        Pt : float
            Total pressure (English units: psi)
        MN : float
            Mach number
        W : float
            Mass flow rate (English units: lbm/s)
        FAR : ignored

        Returns
        -------
        StaticPropsWithDeriv
            Named tuple with static properties plus darea/dMN
        """
        return self._set_static_MN_jit(Tt, Pt, MN, W)

    def set_static_area(self, Tt, Pt, area, W, FAR=None):
        """
        Compute static properties from total conditions and flow area.

        Uses a combined Newton solver for (Ts, Ps, MN, n, pi) simultaneously.

        Parameters
        ----------
        Tt : float
            Total temperature (English units: Rankine)
        Pt : float
            Total pressure (English units: psi)
        area : float
            Flow area (English units: inch^2)
        W : float
            Mass flow rate (English units: lbm/s)
        FAR : ignored

        Returns
        -------
        StaticProps
            Named tuple with static properties
        """
        return self._set_static_area_jit(Tt, Pt, area, W)
