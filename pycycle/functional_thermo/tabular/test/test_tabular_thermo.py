"""Tests for the TabularThermo functional interface."""

import unittest

import numpy as np

from pycycle.functional_thermo import TabularThermo


class TabularThermoPropsTestCase(unittest.TestCase):
    """Test basic property calculations."""

    def setUp(self):
        self.thermo = TabularThermo(FAR=0.0)
        self.T = 500.0  # K
        self.P = 101325.0 * 3  # Pa (3 atm)

    def test_props_TP(self):
        """Test that props_TP returns all properties."""
        props = self.thermo.props_TP(self.T, self.P)

        # Check all properties are returned
        self.assertIsNotNone(props.h)
        self.assertIsNotNone(props.S)
        self.assertIsNotNone(props.gamma)
        self.assertIsNotNone(props.Cp)
        self.assertIsNotNone(props.Cv)
        self.assertIsNotNone(props.rho)
        self.assertIsNotNone(props.R)

        # Check physical sanity
        self.assertGreater(props.h, 0)  # Enthalpy positive at 500K
        self.assertGreater(props.S, 0)  # Entropy positive
        self.assertGreater(props.gamma, 1.0)  # gamma > 1 for real gases
        self.assertLess(props.gamma, 1.7)  # gamma < 1.7 for air
        self.assertGreater(props.Cp, props.Cv)  # Cp > Cv always
        self.assertGreater(props.rho, 0)
        self.assertGreater(props.R, 0)

    def test_individual_properties(self):
        """Test individual property methods match props_TP."""
        props = self.thermo.props_TP(self.T, self.P)

        self.assertAlmostEqual(self.thermo.h(self.T, self.P), props.h)
        self.assertAlmostEqual(self.thermo.S(self.T, self.P), props.S)
        self.assertAlmostEqual(self.thermo.gamma(self.T, self.P), props.gamma)
        self.assertAlmostEqual(self.thermo.Cp(self.T, self.P), props.Cp)
        self.assertAlmostEqual(self.thermo.Cv(self.T, self.P), props.Cv)
        self.assertAlmostEqual(self.thermo.rho(self.T, self.P), props.rho)
        self.assertAlmostEqual(self.thermo.R(self.T, self.P), props.R)

    def test_far_effect(self):
        """Test that FAR affects properties."""
        thermo_air = TabularThermo(FAR=0.0)
        thermo_fuel = TabularThermo(FAR=0.03)

        h_air = thermo_air.h(self.T, self.P)
        h_fuel = thermo_fuel.h(self.T, self.P)

        # Properties should differ with different FAR
        self.assertNotAlmostEqual(h_air, h_fuel, places=2)


class TabularThermoInverseTestCase(unittest.TestCase):
    """Test inverse calculations (solving for T)."""

    def setUp(self):
        self.thermo = TabularThermo(FAR=0.0)

    def test_T_from_hP(self):
        """Test recovering T from h and P."""
        T_original = 500.0
        P = 200000.0

        h = self.thermo.h(T_original, P)
        T_recovered = self.thermo.T_from_hP(h, P)

        self.assertAlmostEqual(T_original, T_recovered, places=6)

    def test_T_from_SP(self):
        """Test recovering T from S and P."""
        T_original = 600.0
        P = 150000.0

        S = self.thermo.S(T_original, P)
        T_recovered = self.thermo.T_from_SP(S, P)

        self.assertAlmostEqual(T_original, T_recovered, places=6)

    def test_T_from_hP_range(self):
        """Test T_from_hP across temperature range."""
        P = 101325.0

        for T_original in [200.0, 400.0, 800.0, 1200.0, 2000.0]:
            h = self.thermo.h(T_original, P)
            T_recovered = self.thermo.T_from_hP(h, P)
            self.assertAlmostEqual(T_original, T_recovered, places=4)


class TabularThermoStaticTestCase(unittest.TestCase):
    """Test static property calculations."""

    def setUp(self):
        self.thermo = TabularThermo(FAR=0.0)
        self.Tt = 500.0  # K
        self.Pt = 300000.0  # Pa
        self.W = 10.0  # kg/s

    def test_static_from_MN(self):
        """Test static properties from Mach number."""
        MN = 0.5
        static = self.thermo.static_from_MN(self.Tt, self.Pt, MN, self.W)

        # Static temperature < total temperature
        self.assertLess(static.Ts, self.Tt)

        # Static pressure < total pressure
        self.assertLess(static.Ps, self.Pt)

        # Mach number should match input
        self.assertAlmostEqual(static.MN, MN)

        # Velocity should be MN * Vsonic
        self.assertAlmostEqual(static.V, MN * static.Vsonic, places=6)

        # Area and density should be positive
        self.assertGreater(static.area, 0)
        self.assertGreater(static.rhos, 0)

    def test_static_from_area(self):
        """Test that static_from_area recovers correct Mach number."""
        MN_original = 0.6
        static1 = self.thermo.static_from_MN(self.Tt, self.Pt, MN_original, self.W)

        static2 = self.thermo.static_from_area(self.Tt, self.Pt, static1.area, self.W)

        self.assertAlmostEqual(static2.MN, MN_original, places=5)
        self.assertAlmostEqual(static2.Ts, static1.Ts, places=5)
        self.assertAlmostEqual(static2.Ps, static1.Ps, places=5)

    def test_static_from_Ps(self):
        """Test that static_from_Ps recovers correct Mach number."""
        MN_original = 0.4
        static1 = self.thermo.static_from_MN(self.Tt, self.Pt, MN_original, self.W)

        static2 = self.thermo.static_from_Ps(self.Tt, self.Pt, static1.Ps, self.W)

        self.assertAlmostEqual(static2.MN, MN_original, places=6)
        self.assertAlmostEqual(static2.Ts, static1.Ts, places=6)

    def test_low_mach_static_near_total(self):
        """Test that at low Mach, static ≈ total conditions."""
        MN = 0.05
        static = self.thermo.static_from_MN(self.Tt, self.Pt, MN, self.W)

        # At low Mach, ratios should be close to 1
        self.assertAlmostEqual(static.Ts / self.Tt, 1.0, places=2)
        self.assertAlmostEqual(static.Ps / self.Pt, 1.0, places=2)


class TabularThermoLinearizeTestCase(unittest.TestCase):
    """Test linearization and derivative methods."""

    def setUp(self):
        self.thermo = TabularThermo(FAR=0.0)
        self.T = 500.0
        self.P = 101325.0 * 3

    def test_linearize_required_before_jvp(self):
        """Test that jvp fails without linearize."""
        with self.assertRaises(RuntimeError):
            self.thermo.jvp(T_dot=1.0, P_dot=0.0)

    def test_linearize_required_before_vjp(self):
        """Test that vjp fails without linearize."""
        with self.assertRaises(RuntimeError):
            self.thermo.vjp(h_bar=1.0)

    def test_jvp_T_direction(self):
        """Test JVP in T direction."""
        self.thermo.linearize(self.T, self.P)
        jvp = self.thermo.jvp(T_dot=1.0, P_dot=0.0)

        # All properties should have tangents
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self.assertIn(prop, jvp)

        # dh/dT should be approximately Cp
        Cp = self.thermo.Cp(self.T, self.P)
        self.assertAlmostEqual(jvp['h'], Cp, delta=Cp * 0.01)

    def test_jvp_P_direction(self):
        """Test JVP in P direction."""
        self.thermo.linearize(self.T, self.P)
        jvp = self.thermo.jvp(T_dot=0.0, P_dot=1.0)

        # dh/dP should be small for ideal-ish gas
        self.assertLess(abs(jvp['h']), 1e-3)

    def test_jvp_linearity(self):
        """Test that JVP is linear."""
        self.thermo.linearize(self.T, self.P)

        jvp1 = self.thermo.jvp(T_dot=1.0, P_dot=0.0)
        jvp2 = self.thermo.jvp(T_dot=2.0, P_dot=0.0)

        # 2x input should give 2x output
        self.assertAlmostEqual(jvp2['h'], 2.0 * jvp1['h'], places=10)

    def test_vjp_consistency_with_jvp(self):
        """Test that VJP is consistent with JVP (transpose relationship)."""
        self.thermo.linearize(self.T, self.P)

        # JVP: tangent in T direction
        jvp_T = self.thermo.jvp(T_dot=1.0, P_dot=0.0)

        # VJP: cotangent from h
        T_bar, P_bar = self.thermo.vjp(h_bar=1.0)

        # dh/dT from JVP should equal T_bar from VJP with h_bar=1
        self.assertAlmostEqual(jvp_T['h'], T_bar, places=10)

        # JVP: tangent in P direction
        jvp_P = self.thermo.jvp(T_dot=0.0, P_dot=1.0)

        # dh/dP from JVP should equal P_bar from VJP with h_bar=1
        self.assertAlmostEqual(jvp_P['h'], P_bar, places=10)

    def test_vjp_multiple_outputs(self):
        """Test VJP with multiple output cotangents."""
        self.thermo.linearize(self.T, self.P)

        # Get individual contributions
        T_bar_h, P_bar_h = self.thermo.vjp(h_bar=1.0)
        T_bar_S, P_bar_S = self.thermo.vjp(S_bar=1.0)

        # Combined should be sum
        T_bar_both, P_bar_both = self.thermo.vjp(h_bar=1.0, S_bar=1.0)

        self.assertAlmostEqual(T_bar_both, T_bar_h + T_bar_S, places=10)
        self.assertAlmostEqual(P_bar_both, P_bar_h + P_bar_S, places=10)

    def test_gradients_finite_difference(self):
        """Test gradients against finite difference."""
        self.thermo.linearize(self.T, self.P)
        jvp = self.thermo.jvp(T_dot=1.0, P_dot=0.0)

        # Finite difference for dh/dT
        eps = 1e-6
        h_plus = self.thermo.h(self.T + eps, self.P)
        h_minus = self.thermo.h(self.T - eps, self.P)
        dh_dT_fd = (h_plus - h_minus) / (2 * eps)

        self.assertAlmostEqual(jvp['h'], dh_dT_fd, places=4)


class TabularThermoStaticDerivativesTestCase(unittest.TestCase):
    """Test derivatives of static property calculations against finite difference."""

    def setUp(self):
        self.thermo = TabularThermo(FAR=0.0, input_units='English')
        self.Tt = 500.0  # degR
        self.Pt = 14.696  # psi
        self.W = 100.0  # lbm/s
        self.MN = 0.5
        self.h = 1e-6  # FD step size

    def test_static_from_MN_derivatives_dTt(self):
        """Test static_from_MN derivatives w.r.t. Tt against FD."""
        self.thermo.linearize_static_MN(self.Tt, self.Pt, self.MN, self.W)
        jvp = self.thermo.jvp_static_MN(1.0, 0.0, 0.0, 0.0)

        # Finite difference
        props_p = self.thermo.static_from_MN(self.Tt + self.h, self.Pt, self.MN, self.W)
        props_m = self.thermo.static_from_MN(self.Tt - self.h, self.Pt, self.MN, self.W)

        for prop in ['Ts', 'Ps', 'hs', 'V', 'Vsonic', 'area', 'gamma']:
            fd = (getattr(props_p, prop) - getattr(props_m, prop)) / (2 * self.h)
            if abs(fd) > 1e-10:
                rel_err = abs(jvp[prop] - fd) / abs(fd)
                self.assertLess(rel_err, 1e-4, f"d{prop}/dTt: ana={jvp[prop]:.8g}, fd={fd:.8g}")

    def test_static_from_MN_derivatives_dPt(self):
        """Test static_from_MN derivatives w.r.t. Pt against FD."""
        self.thermo.linearize_static_MN(self.Tt, self.Pt, self.MN, self.W)
        jvp = self.thermo.jvp_static_MN(0.0, 1.0, 0.0, 0.0)

        props_p = self.thermo.static_from_MN(self.Tt, self.Pt + self.h, self.MN, self.W)
        props_m = self.thermo.static_from_MN(self.Tt, self.Pt - self.h, self.MN, self.W)

        for prop in ['Ts', 'Ps', 'hs', 'V', 'Vsonic', 'area', 'gamma']:
            fd = (getattr(props_p, prop) - getattr(props_m, prop)) / (2 * self.h)
            if abs(fd) > 1e-10:
                rel_err = abs(jvp[prop] - fd) / abs(fd)
                self.assertLess(rel_err, 1e-4, f"d{prop}/dPt: ana={jvp[prop]:.8g}, fd={fd:.8g}")

    def test_static_from_MN_derivatives_dMN(self):
        """Test static_from_MN derivatives w.r.t. MN against FD."""
        self.thermo.linearize_static_MN(self.Tt, self.Pt, self.MN, self.W)
        jvp = self.thermo.jvp_static_MN(0.0, 0.0, 1.0, 0.0)

        props_p = self.thermo.static_from_MN(self.Tt, self.Pt, self.MN + self.h, self.W)
        props_m = self.thermo.static_from_MN(self.Tt, self.Pt, self.MN - self.h, self.W)

        for prop in ['Ts', 'Ps', 'hs', 'V', 'Vsonic', 'area', 'gamma']:
            fd = (getattr(props_p, prop) - getattr(props_m, prop)) / (2 * self.h)
            if abs(fd) > 1e-10:
                rel_err = abs(jvp[prop] - fd) / abs(fd)
                self.assertLess(rel_err, 1e-4, f"d{prop}/dMN: ana={jvp[prop]:.8g}, fd={fd:.8g}")

    def test_static_from_MN_derivatives_dW(self):
        """Test static_from_MN derivatives w.r.t. W against FD."""
        self.thermo.linearize_static_MN(self.Tt, self.Pt, self.MN, self.W)
        jvp = self.thermo.jvp_static_MN(0.0, 0.0, 0.0, 1.0)

        props_p = self.thermo.static_from_MN(self.Tt, self.Pt, self.MN, self.W + self.h)
        props_m = self.thermo.static_from_MN(self.Tt, self.Pt, self.MN, self.W - self.h)

        for prop in ['Ts', 'Ps', 'hs', 'V', 'Vsonic', 'area', 'gamma']:
            fd = (getattr(props_p, prop) - getattr(props_m, prop)) / (2 * self.h)
            if abs(fd) > 1e-10:
                rel_err = abs(jvp[prop] - fd) / abs(fd)
                self.assertLess(rel_err, 1e-4, f"d{prop}/dW: ana={jvp[prop]:.8g}, fd={fd:.8g}")

    def test_static_from_area_derivatives_dTt(self):
        """Test static_from_area derivatives w.r.t. Tt against FD."""
        # Get a valid area
        props_mn = self.thermo.static_from_MN(self.Tt, self.Pt, self.MN, self.W)
        area = props_mn.area

        self.thermo.linearize_static_area(self.Tt, self.Pt, area, self.W)
        jvp = self.thermo.jvp_static_area(1.0, 0.0, 0.0, 0.0)

        props_p = self.thermo.static_from_area(self.Tt + self.h, self.Pt, area, self.W)
        props_m = self.thermo.static_from_area(self.Tt - self.h, self.Pt, area, self.W)

        for prop in ['Ts', 'Ps', 'MN', 'V', 'Vsonic']:
            fd = (getattr(props_p, prop) - getattr(props_m, prop)) / (2 * self.h)
            if abs(fd) > 1e-10:
                rel_err = abs(jvp[prop] - fd) / abs(fd)
                self.assertLess(rel_err, 1e-4, f"d{prop}/dTt: ana={jvp[prop]:.8g}, fd={fd:.8g}")

    def test_static_from_area_derivatives_darea(self):
        """Test static_from_area derivatives w.r.t. area against FD."""
        props_mn = self.thermo.static_from_MN(self.Tt, self.Pt, self.MN, self.W)
        area = props_mn.area

        self.thermo.linearize_static_area(self.Tt, self.Pt, area, self.W)
        jvp = self.thermo.jvp_static_area(0.0, 0.0, 1.0, 0.0)

        props_p = self.thermo.static_from_area(self.Tt, self.Pt, area + self.h, self.W)
        props_m = self.thermo.static_from_area(self.Tt, self.Pt, area - self.h, self.W)

        for prop in ['Ts', 'Ps', 'MN', 'V', 'Vsonic', 'area']:
            fd = (getattr(props_p, prop) - getattr(props_m, prop)) / (2 * self.h)
            if abs(fd) > 1e-10:
                rel_err = abs(jvp[prop] - fd) / abs(fd)
                self.assertLess(rel_err, 1e-4, f"d{prop}/darea: ana={jvp[prop]:.8g}, fd={fd:.8g}")


if __name__ == "__main__":
    unittest.main()
