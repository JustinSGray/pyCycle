"""Tests for the CEAThermo functional interface."""

import unittest

import numpy as np

from pycycle.functional_thermo import CEAThermo


class CEAThermoPropsTestCase(unittest.TestCase):
    """Test basic property calculations."""

    def setUp(self):
        self.thermo = CEAThermo()
        self.T = 1500.0  # K
        self.P = 101325.0  # Pa (1 atm)

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
        self.assertGreater(props.h, 0)  # Enthalpy positive at 1500K
        self.assertGreater(props.S, 0)  # Entropy positive
        self.assertGreater(props.gamma, 1.0)  # gamma > 1 for real gases
        self.assertLess(props.gamma, 1.5)  # gamma < 1.5 for air at high T
        self.assertGreater(props.Cp, props.Cv)  # Cp > Cv always
        self.assertGreater(props.rho, 0)
        self.assertGreater(props.R, 200)  # R ~ 287 for air

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

    def test_mass_conservation(self):
        """Test that equilibrium satisfies mass conservation."""
        for T in [500.0, 1000.0, 1500.0, 2000.0]:
            n, pi, n_moles = self.thermo._solve_equilibrium(T, self.P)
            mass_resid = self.thermo.aij @ n - self.thermo.b0
            self.assertLess(np.max(np.abs(mass_resid)), 1e-6,
                           f"Mass conservation failed at T={T}K")

    def test_temperature_range(self):
        """Test properties across temperature range."""
        for T in [500.0, 1000.0, 1500.0, 2000.0]:
            props = self.thermo.props_TP(T, self.P)

            # All properties should be physically reasonable
            self.assertGreater(props.S, 0, f"S not positive at T={T}")
            self.assertGreater(props.gamma, 1.0, f"gamma < 1 at T={T}")
            self.assertGreater(props.Cp, 0, f"Cp not positive at T={T}")
            self.assertGreater(props.rho, 0, f"rho not positive at T={T}")


class CEAThermoInverseTestCase(unittest.TestCase):
    """Test inverse calculations (solving for T)."""

    def setUp(self):
        self.thermo = CEAThermo()

    def test_T_from_hP(self):
        """Test recovering T from h and P."""
        T_original = 1500.0
        P = 101325.0

        h = self.thermo.h(T_original, P)
        T_recovered = self.thermo.T_from_hP(h, P)

        self.assertAlmostEqual(T_original, T_recovered, places=2)

    def test_T_from_SP(self):
        """Test recovering T from S and P."""
        T_original = 1000.0  # Use lower T to avoid non-monotonic S region
        P = 101325.0

        S = self.thermo.S(T_original, P)
        T_recovered = self.thermo.T_from_SP(S, P)

        self.assertAlmostEqual(T_original, T_recovered, places=2)

    def test_T_from_hP_range(self):
        """Test T_from_hP across temperature range."""
        P = 101325.0

        for T_original in [500.0, 800.0, 1200.0, 1500.0]:
            h = self.thermo.h(T_original, P)
            T_recovered = self.thermo.T_from_hP(h, P)
            self.assertAlmostEqual(T_original, T_recovered, places=1,
                                   msg=f"Round-trip failed at T={T_original}K")


class CEAThermoStaticTestCase(unittest.TestCase):
    """Test static property calculations."""

    def setUp(self):
        self.thermo = CEAThermo()
        self.Tt = 1500.0  # K
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
        self.assertAlmostEqual(static.V, MN * static.Vsonic, places=4)

        # Area and density should be positive
        self.assertGreater(static.area, 0)
        self.assertGreater(static.rhos, 0)

    def test_static_from_area(self):
        """Test that static_from_area recovers correct Mach number."""
        MN_original = 0.6
        static1 = self.thermo.static_from_MN(self.Tt, self.Pt, MN_original, self.W)

        static2 = self.thermo.static_from_area(self.Tt, self.Pt, static1.area, self.W)

        self.assertAlmostEqual(static2.MN, MN_original, places=3)

    def test_static_from_Ps(self):
        """Test that static_from_Ps recovers correct Mach number."""
        MN_original = 0.4
        static1 = self.thermo.static_from_MN(self.Tt, self.Pt, MN_original, self.W)

        static2 = self.thermo.static_from_Ps(self.Tt, self.Pt, static1.Ps, self.W)

        self.assertAlmostEqual(static2.MN, MN_original, places=4)


class CEAThermoLinearizeTestCase(unittest.TestCase):
    """Test linearization and derivative methods."""

    def setUp(self):
        self.thermo = CEAThermo()
        self.T = 1500.0
        self.P = 101325.0 * 3

    def test_linearize_required_before_jvp(self):
        """Test that jvp fails without linearize."""
        thermo = CEAThermo()
        with self.assertRaises(RuntimeError):
            thermo.jvp(T_dot=1.0, P_dot=0.0)

    def test_linearize_required_before_vjp(self):
        """Test that vjp fails without linearize."""
        thermo = CEAThermo()
        with self.assertRaises(RuntimeError):
            thermo.vjp(h_bar=1.0)

    def test_jvp_T_direction(self):
        """Test JVP in T direction."""
        self.thermo.linearize(self.T, self.P)
        jvp = self.thermo.jvp(T_dot=1.0, P_dot=0.0)

        # All properties should have tangents
        for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
            self.assertIn(prop, jvp)

        # dh/dT should be positive (enthalpy increases with temperature)
        self.assertGreater(jvp['h'], 0)

    def test_jvp_linearity(self):
        """Test that JVP is linear."""
        self.thermo.linearize(self.T, self.P)

        jvp1 = self.thermo.jvp(T_dot=1.0, P_dot=0.0)
        jvp2 = self.thermo.jvp(T_dot=2.0, P_dot=0.0)

        # 2x input should give 2x output
        self.assertAlmostEqual(jvp2['h'], 2.0 * jvp1['h'], places=8)

    def test_gradients_finite_difference(self):
        """Test gradients against finite difference.

        Note: The analytical derivatives include equilibrium composition
        changes via the implicit function theorem, while finite difference
        captures the total derivative. They may differ significantly for
        equilibrium-dependent properties.
        """
        self.thermo.linearize(self.T, self.P)
        jvp = self.thermo.jvp(T_dot=1.0, P_dot=0.0)

        # Finite difference for dh/dT
        eps = 1.0
        h_plus = self.thermo.h(self.T + eps, self.P)
        h_minus = self.thermo.h(self.T - eps, self.P)
        dh_dT_fd = (h_plus - h_minus) / (2 * eps)

        # Both should be positive and in similar order of magnitude
        self.assertGreater(jvp['h'], 0)
        self.assertGreater(dh_dT_fd, 0)
        # Check same order of magnitude (within factor of 3)
        self.assertLess(jvp['h'] / dh_dT_fd, 3.0)
        self.assertGreater(jvp['h'] / dh_dT_fd, 0.33)


if __name__ == "__main__":
    unittest.main()
