"""Tests for JaxTabularThermo - matches pycycle/thermo/test/test_thermo_total_static_tabulated.py."""

import unittest

import numpy as np

from pycycle.constants import AIR_JETA_TAB_SPEC
from pycycle.functional_thermo.tabular.jax_tabular import JaxTabularThermo


# Unit conversions for reference values
_K_TO_R = 9.0 / 5.0            # Kelvin -> Rankine
_BAR_TO_PSI = 14.5038           # bar -> psi
_J_PER_KG_K_TO_BTU_PER_LBM_R = 1.0 / 4186.8  # J/(kg*K) -> Btu/(lbm*R)

# Default composition for tabular thermo (FAR=0)
_COMP_ZERO = np.array([0.0])


class TestSetTotalTP(unittest.TestCase):
    """Matches SetTotalSimpleTestCase.test_set_total_TP (first method)."""

    def setUp(self):
        self.thermo = JaxTabularThermo(AIR_JETA_TAB_SPEC)

    def test_TP_at_2000K(self):
        """T=2000K, P=1.034210 bar -> gamma=1.27532298."""
        T_R = 2000.0 * _K_TO_R
        P_psi = 1.034210 * _BAR_TO_PSI

        props = self.thermo.set_total_TP(T_R, P_psi, _COMP_ZERO)
        np.testing.assert_allclose(float(props.gamma), 1.27532298, rtol=1e-4)

    def test_TP_at_1500K(self):
        """T=1500K, P=1.034210 bar -> gamma=1.30444708."""
        T_R = 1500.0 * _K_TO_R
        P_psi = 1.034210 * _BAR_TO_PSI

        props = self.thermo.set_total_TP(T_R, P_psi, _COMP_ZERO)
        np.testing.assert_allclose(float(props.gamma), 1.30444708, rtol=1e-4)


class TestSetTotalhP(unittest.TestCase):
    """Matches SetTotalSimpleTestCase.test_set_total_hP."""

    def setUp(self):
        self.thermo = JaxTabularThermo(AIR_JETA_TAB_SPEC)

    def test_hP_at_2000K(self):
        """h from TP at 2000K, P=1.034210 bar -> recovers T=2000K, gamma=1.27532298."""
        T_R = 2000.0 * _K_TO_R
        P_psi = 1.034210 * _BAR_TO_PSI

        props = self.thermo.set_total_TP(T_R, P_psi, _COMP_ZERO)
        h = float(props.h)

        T_recovered = float(self.thermo.set_total_hP(h, P_psi, _COMP_ZERO))
        np.testing.assert_allclose(T_recovered, T_R, rtol=1e-4)

        props2 = self.thermo.set_total_TP(T_recovered, P_psi, _COMP_ZERO)
        np.testing.assert_allclose(float(props2.gamma), 1.27532298, rtol=1e-4)

    def test_hP_at_1500K(self):
        """h from TP at 1500K, P=1.034210 bar -> recovers T=1500K, gamma=1.30444708."""
        T_R = 1500.0 * _K_TO_R
        P_psi = 1.034210 * _BAR_TO_PSI

        props = self.thermo.set_total_TP(T_R, P_psi, _COMP_ZERO)
        h = float(props.h)

        T_recovered = float(self.thermo.set_total_hP(h, P_psi, _COMP_ZERO))
        np.testing.assert_allclose(T_recovered, T_R, rtol=1e-4)

        props2 = self.thermo.set_total_TP(T_recovered, P_psi, _COMP_ZERO)
        np.testing.assert_allclose(float(props2.gamma), 1.30444708, rtol=1e-4)


class TestSetTotalSP(unittest.TestCase):
    """Matches SetTotalSimpleTestCase.test_set_total_SP."""

    def setUp(self):
        self.thermo = JaxTabularThermo(AIR_JETA_TAB_SPEC)

    def test_SP_at_2000K(self):
        """S=8982.03057206 J/kg/K, P=1.034210 bar -> T=2000K, gamma=1.27532298."""
        S_eng = 8982.03057206 * _J_PER_KG_K_TO_BTU_PER_LBM_R
        P_psi = 1.034210 * _BAR_TO_PSI

        T_recovered = float(self.thermo.set_total_SP(S_eng, P_psi, _COMP_ZERO))
        T_expected = 2000.0 * _K_TO_R

        np.testing.assert_allclose(T_recovered, T_expected, rtol=5e-3)

        props = self.thermo.set_total_TP(T_recovered, P_psi, _COMP_ZERO)
        np.testing.assert_allclose(float(props.gamma), 1.27532298, rtol=5e-4)

    def test_SP_at_1500K(self):
        """S=8615.116554906986 J/kg/K, P=1.034210 bar -> T=1500K, gamma=1.30444708."""
        S_eng = 8615.116554906986 * _J_PER_KG_K_TO_BTU_PER_LBM_R
        P_psi = 1.034210 * _BAR_TO_PSI

        T_recovered = float(self.thermo.set_total_SP(S_eng, P_psi, _COMP_ZERO))
        T_expected = 1500.0 * _K_TO_R

        np.testing.assert_allclose(T_recovered, T_expected, rtol=5e-3)

        props = self.thermo.set_total_TP(T_recovered, P_psi, _COMP_ZERO)
        np.testing.assert_allclose(float(props.gamma), 1.30444708, rtol=5e-4)


class TestSetTotalEquivalence(unittest.TestCase):
    """Matches TestSetTotalTabular.test_set_total_equivalence.

    Runs TP to get h and S, then verifies hP and SP recover the same T.
    """

    def setUp(self):
        self.thermo = JaxTabularThermo(AIR_JETA_TAB_SPEC)

    def _check_equivalence(self, T_R, P_psi):
        """Verify TP, hP, and SP all produce the same temperature."""
        props = self.thermo.set_total_TP(T_R, P_psi, _COMP_ZERO)
        h = float(props.h)
        S = float(props.S)

        T_from_hP = float(self.thermo.set_total_hP(h, P_psi, _COMP_ZERO))
        T_from_SP = float(self.thermo.set_total_SP(S, P_psi, _COMP_ZERO))

        np.testing.assert_allclose(T_from_hP, T_R, rtol=1e-4,
                                   err_msg=f"hP failed at T={T_R}R, P={P_psi}psi")
        np.testing.assert_allclose(T_from_SP, T_R, rtol=1e-4,
                                   err_msg=f"SP failed at T={T_R}R, P={P_psi}psi")

    def test_sea_level_standard(self):
        """T=518 degR, P=14.7 psi."""
        self._check_equivalence(518.0, 14.7)

    def test_high_temp(self):
        """T=3000 degR, P=30 psi."""
        self._check_equivalence(3000.0, 30.0)

    def test_mid_temp_high_pressure(self):
        """T=1500 degR, P=80 psi."""
        self._check_equivalence(1500.0, 80.0)


class TestDerivatives_CS(unittest.TestCase):
    """Verify JAX jacfwd derivatives match complex-step for all tabular thermo methods."""

    CS_STEP = 1e-20

    def setUp(self):
        self.thermo = JaxTabularThermo(AIR_JETA_TAB_SPEC)
        self.comp = _COMP_ZERO

    def _check_derivs(self, fn, x0, labels, atol=1e-8, rtol=1e-8):
        """Compare jacfwd vs complex-step for a scalar-input, vector-output function."""
        import jax
        import jax.numpy as jnp

        jac_ad = np.asarray(jax.jacfwd(fn)(x0))
        out_cs = fn(x0 + self.CS_STEP * 1j)
        jac_cs = np.imag(np.asarray(out_cs)) / self.CS_STEP

        for i, label in enumerate(labels):
            with self.subTest(output=label):
                np.testing.assert_allclose(
                    jac_ad[i], jac_cs[i], atol=atol, rtol=rtol,
                    err_msg=f"Derivative mismatch for {label}")

    # ----- set_total_TP -----

    def test_total_TP_wrt_T(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696
        labels = ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']

        def fn(T_in):
            p = self.thermo.set_total_TP(T_in, P, self.comp)
            return jnp.array([p.h, p.S, p.gamma, p.Cp, p.Cv, p.rho, p.R])

        self._check_derivs(fn, T, labels)

    def test_total_TP_wrt_P(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696
        labels = ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']

        def fn(P_in):
            p = self.thermo.set_total_TP(T, P_in, self.comp)
            return jnp.array([p.h, p.S, p.gamma, p.Cp, p.Cv, p.rho, p.R])

        self._check_derivs(fn, P, labels)

    # ----- set_total_hP -----

    def test_total_hP_wrt_h(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696
        props = self.thermo.set_total_TP(T, P, self.comp)
        h = float(props.h)

        def fn(h_in):
            return jnp.array([self.thermo.set_total_hP(h_in, P, self.comp)])

        self._check_derivs(fn, h, ['T_from_hP'])

    def test_total_hP_wrt_P(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696
        props = self.thermo.set_total_TP(T, P, self.comp)
        h = float(props.h)

        def fn(P_in):
            return jnp.array([self.thermo.set_total_hP(h, P_in, self.comp)])

        self._check_derivs(fn, P, ['T_from_hP'])

    # ----- set_static_MN -----

    def test_static_MN_wrt_MN(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696; MN = 0.5; W = 100.0
        labels = ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                  'gamma', 'Cp', 'Cv', 'S', 'R']

        def fn(MN_in):
            r = self.thermo.set_static_MN(T, P, MN_in, W, self.comp)
            return jnp.array([r.Ts, r.Ps, r.hs, r.rhos, r.MN, r.V, r.Vsonic,
                              r.area, r.gamma, r.Cp, r.Cv, r.S, r.R])

        self._check_derivs(fn, MN, labels)

    def test_static_MN_wrt_T(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696; MN = 0.5; W = 100.0
        labels = ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                  'gamma', 'Cp', 'Cv', 'S', 'R']

        def fn(T_in):
            r = self.thermo.set_static_MN(T_in, P, MN, W, self.comp)
            return jnp.array([r.Ts, r.Ps, r.hs, r.rhos, r.MN, r.V, r.Vsonic,
                              r.area, r.gamma, r.Cp, r.Cv, r.S, r.R])

        self._check_derivs(fn, T, labels)

    def test_static_MN_wrt_W(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696; MN = 0.5; W = 100.0
        labels = ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                  'gamma', 'Cp', 'Cv', 'S', 'R']

        def fn(W_in):
            r = self.thermo.set_static_MN(T, P, MN, W_in, self.comp)
            return jnp.array([r.Ts, r.Ps, r.hs, r.rhos, r.MN, r.V, r.Vsonic,
                              r.area, r.gamma, r.Cp, r.Cv, r.S, r.R])

        self._check_derivs(fn, W, labels)

    # ----- set_static_area -----

    def test_static_area_wrt_area(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696; MN = 0.5; W = 100.0
        static_mn = self.thermo.set_static_MN(T, P, MN, W, self.comp)
        area = float(static_mn.area)
        labels = ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                  'gamma', 'Cp', 'Cv', 'S', 'R']

        def fn(area_in):
            r = self.thermo.set_static_area(T, P, area_in, W, self.comp)
            return jnp.array([r.Ts, r.Ps, r.hs, r.rhos, r.MN, r.V, r.Vsonic,
                              r.area, r.gamma, r.Cp, r.Cv, r.S, r.R])

        self._check_derivs(fn, area, labels)

    def test_static_area_wrt_T(self):
        import jax.numpy as jnp
        T = 1800.0; P = 14.696; MN = 0.5; W = 100.0
        static_mn = self.thermo.set_static_MN(T, P, MN, W, self.comp)
        area = float(static_mn.area)
        labels = ['Ts', 'Ps', 'hs', 'rhos', 'MN', 'V', 'Vsonic', 'area',
                  'gamma', 'Cp', 'Cv', 'S', 'R']

        def fn(T_in):
            r = self.thermo.set_static_area(T_in, P, area, W, self.comp)
            return jnp.array([r.Ts, r.Ps, r.hs, r.rhos, r.MN, r.V, r.Vsonic,
                              r.area, r.gamma, r.Cp, r.Cv, r.S, r.R])

        self._check_derivs(fn, T, labels)


if __name__ == "__main__":
    unittest.main()
