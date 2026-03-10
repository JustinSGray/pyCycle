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


if __name__ == "__main__":
    unittest.main()
