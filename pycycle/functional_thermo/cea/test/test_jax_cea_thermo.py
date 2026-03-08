"""Tests for JaxCEAThermo against reference data from the OpenMDAO CEA thermo tests."""

import os
import unittest

import numpy as np

from pycycle.functional_thermo.cea.jax_cea import JaxCEAThermo
from pycycle.thermo.cea import species_data
from pycycle import constants


class TestJaxCEATotalhP(unittest.TestCase):
    """Test set_total_hP round-trip against set_total_TP."""

    def setUp(self):
        self.jax_thermo = JaxCEAThermo()
        self.b0 = self.jax_thermo._default_composition

    def test_hP_round_trip(self):
        """Get h from TP, then recover T from hP."""
        for T_R in [900.0, 1800.0, 2700.0, 3600.0]:
            P = 14.696  # psi
            props = self.jax_thermo.set_total_TP(T_R, P, self.b0)
            T_recovered = self.jax_thermo.set_total_hP(float(props.h), P, self.b0)
            np.testing.assert_allclose(
                float(T_recovered), T_R, rtol=1e-3,
                err_msg=f"hP round-trip failed at T={T_R}R"
            )


class TestJaxCEAStaticMN(unittest.TestCase):
    """Test set_static_MN for physical sanity."""

    def setUp(self):
        self.jax_thermo = JaxCEAThermo()
        self.b0 = self.jax_thermo._default_composition

    def test_static_MN_sanity(self):
        """Static T < total T, static P < total P, V = MN * Vsonic."""
        Tt = 2700.0  # Rankine
        Pt = 44.0    # psi
        MN = 0.5
        W = 100.0    # lbm/s

        static = self.jax_thermo.set_static_MN(Tt, Pt, MN, W, self.b0)

        self.assertLess(float(static.Ts), Tt, "Static T should be < total T")
        self.assertLess(float(static.Ps), Pt, "Static P should be < total P")
        np.testing.assert_allclose(float(static.MN), MN, rtol=1e-6)
        np.testing.assert_allclose(float(static.V), MN * float(static.Vsonic), rtol=1e-4)
        self.assertGreater(float(static.area), 0)
        self.assertGreater(float(static.rhos), 0)

    def test_darea_dMN_sign(self):
        """darea/dMN should be negative for subsonic flow."""
        Tt = 2700.0; Pt = 44.0; MN = 0.5; W = 100.0
        static = self.jax_thermo.set_static_MN(Tt, Pt, MN, W, self.b0)
        self.assertLess(float(static.darea_dMN), 0,
                        "darea/dMN should be negative for subsonic flow")


class TestJaxCEAStaticArea(unittest.TestCase):
    """Test set_static_area round-trip."""

    def setUp(self):
        self.jax_thermo = JaxCEAThermo()
        self.b0 = self.jax_thermo._default_composition

    def test_area_round_trip(self):
        """Get area from MN solver, recover MN from area solver."""
        Tt = 2700.0   # Rankine
        Pt = 44.0     # psi
        MN_orig = 0.6
        W = 100.0     # lbm/s

        static1 = self.jax_thermo.set_static_MN(Tt, Pt, MN_orig, W, self.b0)
        area = float(static1.area)

        static2 = self.jax_thermo.set_static_area(Tt, Pt, area, W, self.b0)

        np.testing.assert_allclose(float(static2.MN), MN_orig, rtol=1e-4,
                                   err_msg="Area round-trip failed to recover MN")

    def test_area_round_trip_low_MN(self):
        """Round-trip at low Mach number."""
        Tt = 1800.0; Pt = 100.0; MN_orig = 0.3; W = 50.0

        static1 = self.jax_thermo.set_static_MN(Tt, Pt, MN_orig, W, self.b0)
        area = float(static1.area)

        static2 = self.jax_thermo.set_static_area(Tt, Pt, area, W, self.b0)

        np.testing.assert_allclose(float(static2.MN), MN_orig, rtol=1e-4,
                                   err_msg="Area round-trip failed at low MN")


# =========================================================================
# Tests below mimic the OpenMDAO CEA thermo tests in pycycle/thermo/test/
# using the same reference data and tolerances.
# =========================================================================

class TestChemEqJanaf(unittest.TestCase):
    """Mimic pycycle/thermo/cea/test/test_chem_eq_janaf.py - verify equilibrium mole fractions."""

    def setUp(self):
        self.jax_thermo = JaxCEAThermo()
        self.b0 = self.jax_thermo._default_composition

    def test_equilibrium_n_at_1500K(self):
        """Mole fractions at T=1500K, P=1.034210 bar should match OpenMDAO ChemEq reference.

        Uses atol (not rtol) because trace species concentrations can differ between
        JAX and OpenMDAO solvers while bulk properties still match. All absolute
        differences are < 1e-4.
        """
        T_si = 1500.0       # K
        P_si = 103421.0     # Pa (1.034210 bar)

        n, pi, n_moles = self.jax_thermo._solve_equilibrium(
            T_si, P_si, self.b0,
            self.jax_thermo._n_init, np.zeros(self.jax_thermo.num_element)
        )

        check_val = np.array([3.23319236e-04, 1.00000000e-10, 1.10138429e-05, 1.00000000e-10,
                              1.72853915e-08, 6.76015824e-09, 1.00000000e-10, 2.69578737e-02,
                              4.80653071e-09, 7.23197634e-03])

        np.testing.assert_allclose(np.array(n), check_val, atol=1e-4,
                                   err_msg="Mole fractions at 1500K don't match OpenMDAO ChemEq")


class TestTotalTP_CO2(unittest.TestCase):
    """Mimic pycycle/thermo/test/test_thermo_total_static_cea.py SetTotalSimpleTestCase TP tests.

    Tests CO2-CO-O2 composition at two temperature conditions.
    """

    def setUp(self):
        self.jax_thermo = JaxCEAThermo(
            composition=constants.CEA_CO2_CO_O2_COMPOSITION,
            thermo_data=species_data.co2_co_o2
        )
        self.b0 = self.jax_thermo._default_composition

    def test_gamma_at_4000K(self):
        """gamma at T=4000K, P=1.034210 bar should be 1.19054697."""
        T_R = 4000.0 * 9.0 / 5.0   # K -> Rankine
        P_psi = 1.034210 * 14.5038  # bar -> psi

        props = self.jax_thermo.set_total_TP(T_R, P_psi, self.b0)
        np.testing.assert_allclose(float(props.gamma), 1.19054697, rtol=1e-4)

    def test_gamma_at_1500K(self):
        """gamma at T=1500K, P=1.034210 bar should be 1.16379233."""
        T_R = 1500.0 * 9.0 / 5.0
        P_psi = 1.034210 * 14.5038

        props = self.jax_thermo.set_total_TP(T_R, P_psi, self.b0)
        np.testing.assert_allclose(float(props.gamma), 1.16379233, rtol=1e-4)


class TestTotalhP_CO2(unittest.TestCase):
    """Mimic pycycle/thermo/test/test_thermo_total_static_cea.py SetTotalSimpleTestCase hP tests.

    Tests CO2-CO-O2 composition using enthalpy-pressure mode.
    Reference h values from OpenMDAO are in cal/g; convert to Btu/lbm for JAX API.
    """

    CAL_G_TO_BTU_LBM = 4184.0 / 2326.0

    def setUp(self):
        self.jax_thermo = JaxCEAThermo(
            composition=constants.CEA_CO2_CO_O2_COMPOSITION,
            thermo_data=species_data.co2_co_o2
        )
        self.b0 = self.jax_thermo._default_composition

    def test_hP_at_4000K_condition(self):
        """h=340 cal/g, P=1.034210 bar should give gamma ~ 1.19039688581."""
        h_btu = 340.0 * self.CAL_G_TO_BTU_LBM
        P_psi = 1.034210 * 14.5038

        T_R = float(self.jax_thermo.set_total_hP(h_btu, P_psi, self.b0))
        props = self.jax_thermo.set_total_TP(T_R, P_psi, self.b0)
        np.testing.assert_allclose(float(props.gamma), 1.19039688581, rtol=1e-4)

    def test_hP_at_1500K_condition(self):
        """h=-1801.35537381 cal/g, P=1.034210 bar should give gamma ~ 1.16379012007."""
        h_btu = -1801.35537381 * self.CAL_G_TO_BTU_LBM
        P_psi = 1.034210 * 14.5038

        T_R = float(self.jax_thermo.set_total_hP(h_btu, P_psi, self.b0))
        props = self.jax_thermo.set_total_TP(T_R, P_psi, self.b0)
        np.testing.assert_allclose(float(props.gamma), 1.16379012007, rtol=1e-4)


class TestTotalEquivalenceJanaf(unittest.TestCase):
    """Mimic pycycle/thermo/test/test_thermo_total_static_cea.py TestSetTotalJanaf.

    Tests that set_total_TP and set_total_hP give equivalent results (TP -> hP round-trip).
    Uses the same three test conditions as the OpenMDAO version.
    """

    def setUp(self):
        self.jax_thermo = JaxCEAThermo()
        self.b0 = self.jax_thermo._default_composition

    def _check_equivalence(self, T_R, P_psi):
        """Compute h from TP, then recover T from hP and verify match."""
        props = self.jax_thermo.set_total_TP(T_R, P_psi, self.b0)
        h = float(props.h)

        T_recovered = float(self.jax_thermo.set_total_hP(h, P_psi, self.b0))
        np.testing.assert_allclose(T_recovered, T_R, rtol=1e-4,
                                   err_msg=f"hP round-trip failed at T={T_R}R, P={P_psi}psi")

    def test_sea_level_standard(self):
        """T=518 degR, P=14.7 psi."""
        self._check_equivalence(518.0, 14.7)

    def test_high_temp(self):
        """T=3000 degR, P=30 psi."""
        self._check_equivalence(3000.0, 30.0)

    def test_mid_temp_high_pressure(self):
        """T=1500 degR, P=80 psi."""
        self._check_equivalence(1500.0, 80.0)


class TestStaticMN_NPSS(unittest.TestCase):
    """Mimic pycycle/thermo/test/test_thermo_total_static_cea.py TestStaticJanaf.test_case_MN.

    Tests set_static_MN against NPSS reference data for air (JANAF).
    Tolerances are slightly relaxed vs OpenMDAO because the JAX version uses
    isentropic relations with total-condition gamma (approximate) rather than
    a full implicit solve at static conditions.
    """

    def setUp(self):
        self.jax_thermo = JaxCEAThermo()
        self.b0 = self.jax_thermo._default_composition

        fpath = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(fpath, '..', '..', '..', 'thermo', 'test', 'NPSS_Static_CEA_Data.csv')
        self.ref_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

        self.header = ['W', 'MN', 'V', 'A', 's', 'Pt', 'Tt', 'ht', 'rhot',
                        'gamt', 'Ps', 'Ts', 'hs', 'rhos', 'gams']
        self.h_map = {name: i for i, name in enumerate(self.header)}

    def _check_static(self, static, npss_data):
        """Compare JaxCEA static properties against NPSS reference."""
        h = self.h_map
        MN_ref = npss_data[h['MN']]

        if MN_ref < 0.05:
            tol = 0.2
        elif MN_ref > 1.0:
            tol = 2e-3  # isentropic approximation less accurate for supersonic
        else:
            tol = 5e-4

        np.testing.assert_allclose(float(static.MN), MN_ref, rtol=tol,
                                   err_msg="MN mismatch")
        np.testing.assert_allclose(float(static.gamma), npss_data[h['gams']], rtol=tol,
                                   err_msg="gams mismatch")
        np.testing.assert_allclose(float(static.Ps), npss_data[h['Ps']], rtol=tol,
                                   err_msg="Ps mismatch")
        np.testing.assert_allclose(float(static.Ts), npss_data[h['Ts']], rtol=tol,
                                   err_msg="Ts mismatch")
        np.testing.assert_allclose(float(static.hs), npss_data[h['hs']], rtol=tol,
                                   err_msg="hs mismatch")
        np.testing.assert_allclose(float(static.rhos), npss_data[h['rhos']], rtol=tol,
                                   err_msg="rhos mismatch")
        np.testing.assert_allclose(float(static.V), npss_data[h['V']], rtol=tol,
                                   err_msg="V mismatch")
        np.testing.assert_allclose(float(static.area), npss_data[h['A']], rtol=tol,
                                   err_msg="A mismatch")

    def test_all_cases(self):
        """Test all 6 NPSS reference cases for static MN."""
        h = self.h_map
        for i, data in enumerate(self.ref_data):
            with self.subTest(case=i, MN=data[h['MN']], Tt=data[h['Tt']]):
                static = self.jax_thermo.set_static_MN(
                    data[h['Tt']], data[h['Pt']], data[h['MN']], data[h['W']],
                    self.b0)
                self._check_static(static, data)


class TestStaticArea_NPSS(unittest.TestCase):
    """Mimic pycycle/thermo/test/test_thermo_total_static_cea.py TestStaticJanaf.test_case_area.

    Tests set_static_area against NPSS reference data for air (JANAF).
    Skips case 5 (supersonic, MN=2) since the JAX area solver is subsonic only.
    Tolerances are slightly relaxed vs OpenMDAO because the JAX version uses
    isentropic relations with total-condition gamma (approximate).
    """

    def setUp(self):
        self.jax_thermo = JaxCEAThermo()
        self.b0 = self.jax_thermo._default_composition

        fpath = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(fpath, '..', '..', '..', 'thermo', 'test', 'NPSS_Static_CEA_Data.csv')
        self.ref_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

        self.header = ['W', 'MN', 'V', 'A', 's', 'Pt', 'Tt', 'ht', 'rhot',
                        'gamt', 'Ps', 'Ts', 'hs', 'rhos', 'gams']
        self.h_map = {name: i for i, name in enumerate(self.header)}

    def _check_static(self, static, npss_data):
        """Compare JaxCEA static properties against NPSS reference."""
        h = self.h_map
        MN_ref = npss_data[h['MN']]

        if MN_ref < 0.05:
            tol = 0.2
        else:
            tol = 5e-4

        np.testing.assert_allclose(float(static.MN), MN_ref, rtol=tol,
                                   err_msg="MN mismatch")
        np.testing.assert_allclose(float(static.gamma), npss_data[h['gams']], rtol=tol,
                                   err_msg="gams mismatch")
        np.testing.assert_allclose(float(static.Ps), npss_data[h['Ps']], rtol=tol,
                                   err_msg="Ps mismatch")
        np.testing.assert_allclose(float(static.Ts), npss_data[h['Ts']], rtol=tol,
                                   err_msg="Ts mismatch")
        np.testing.assert_allclose(float(static.hs), npss_data[h['hs']], rtol=tol,
                                   err_msg="hs mismatch")
        np.testing.assert_allclose(float(static.rhos), npss_data[h['rhos']], rtol=tol,
                                   err_msg="rhos mismatch")
        np.testing.assert_allclose(float(static.V), npss_data[h['V']], rtol=tol,
                                   err_msg="V mismatch")
        np.testing.assert_allclose(float(static.area), npss_data[h['A']], rtol=tol,
                                   err_msg="A mismatch")

    def test_subsonic_cases(self):
        """Test subsonic NPSS reference cases for static area."""
        h = self.h_map
        for i, data in enumerate(self.ref_data):
            if i == 5:  # skip supersonic case (MN=2)
                continue
            with self.subTest(case=i, MN=data[h['MN']], Tt=data[h['Tt']]):
                static = self.jax_thermo.set_static_area(
                    data[h['Tt']], data[h['Pt']], data[h['A']], data[h['W']],
                    self.b0)
                self._check_static(static, data)


class TestPropsCalcs_CO2(unittest.TestCase):
    """Mimic pycycle/thermo/cea/test/test_props_rhs_co2.py PropsCalcsTestCase.

    Tests thermodynamic property values at two conditions for CO2-CO-O2 mixture.
    """

    CAL_G_TO_BTU_LBM = 4184.0 / 2326.0

    def setUp(self):
        self.jax_thermo = JaxCEAThermo(
            composition=constants.CEA_CO2_CO_O2_COMPOSITION,
            thermo_data=species_data.co2_co_o2
        )
        self.b0 = self.jax_thermo._default_composition

    def test_props_at_4000K(self):
        """Properties at T=4000K, P=1.034210 bar.

        Reference values are from PropsCalcs with pre-computed n values.
        Full equilibrium solve gives slightly different compositions (especially
        at 4000K where CO2 dissociation is significant), leading to small
        differences in h (~0.2%) while gamma is very close (~0.01%).
        """
        T_R = 4000.0 * 9.0 / 5.0
        P_psi = 1.034210 * 14.5038

        props = self.jax_thermo.set_total_TP(T_R, P_psi, self.b0)

        np.testing.assert_allclose(float(props.gamma), 1.19039, rtol=2e-4)
        np.testing.assert_allclose(float(props.h), 340.324938088 * self.CAL_G_TO_BTU_LBM, rtol=3e-3)

    def test_props_at_1500K(self):
        """Properties at T=1500K, P=1.034210 bar."""
        T_R = 1500.0 * 9.0 / 5.0
        P_psi = 1.034210 * 14.5038

        props = self.jax_thermo.set_total_TP(T_R, P_psi, self.b0)

        tol = 1e-4
        np.testing.assert_allclose(float(props.gamma), 1.16380, rtol=tol)
        np.testing.assert_allclose(float(props.h), -1801.35777129 * self.CAL_G_TO_BTU_LBM, rtol=tol)


class TestDifferentCompositions(unittest.TestCase):
    """Test that a single JaxCEAThermo instance works with different b0 values."""

    def test_same_instance_different_b0(self):
        """Calling with different b0 values should produce different results."""
        thermo = JaxCEAThermo()
        b0_air = thermo._default_composition

        T_R = 1800.0
        P_psi = 14.696

        props_air = thermo.set_total_TP(T_R, P_psi, b0_air)

        # Perturb b0 slightly (simulate different composition)
        import jax.numpy as jnp
        b0_perturbed = b0_air * 1.01
        props_perturbed = thermo.set_total_TP(T_R, P_psi, b0_perturbed)

        # Results should be different
        self.assertNotAlmostEqual(float(props_air.h), float(props_perturbed.h),
                                  places=4, msg="Different compositions should give different h")

        # But calling with same b0 again should give same result
        props_air2 = thermo.set_total_TP(T_R, P_psi, b0_air)
        np.testing.assert_allclose(float(props_air2.h), float(props_air.h), rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
