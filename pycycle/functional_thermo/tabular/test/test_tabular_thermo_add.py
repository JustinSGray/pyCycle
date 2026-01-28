"""Tests for the functional Tabular ThermoAdd interface."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.constants import TAB_AIR_FUEL_COMPOSITION, AIR_JETA_TAB_SPEC
from pycycle.thermo.tabular.thermo_add import ThermoAdd as ThermoAddOM

from pycycle.functional_thermo.tabular import ThermoAdd


class ThermoAddReactantTestCase(unittest.TestCase):
    """Test reactant mode (fuel addition)."""

    def test_mix_1fuel(self):
        """Test mixing with single fuel."""
        # OpenMDAO version
        p = om.Problem()
        p.model.add_subsystem('thermo_add',
                              ThermoAddOM(spec=AIR_JETA_TAB_SPEC,
                                        inflow_composition=TAB_AIR_FUEL_COMPOSITION,
                                        mix_mode='reactant',
                                        mix_composition='FAR',
                                        mix_names='fuel'),
                              promotes=['*'])
        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 38.8
        p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:composition'] = [0.0]  # FAR = 0
        p['fuel:ratio'] = 0.02673

        p.run_model()

        om_h = p['mass_avg_h'][0]
        om_W = p['Wout'][0]
        om_comp = p['composition_out']
        om_fuel_W = p['fuel:W'][0]

        # Functional version
        mixer = ThermoAdd(inflow_composition=TAB_AIR_FUEL_COMPOSITION,
                         mix_mode='reactant',
                         mix_composition='FAR',
                         mix_names='fuel')

        # Convert units: lbm/s -> kg/s, Btu/lbm -> J/kg
        W = 38.8 * 0.453592  # kg/s
        h = 181.381769 * 2326.0  # J/kg
        ratio = 0.02673
        composition = np.array([0.0])  # FAR = 0

        result = mixer.compute(W=W, h=h, composition=composition,
                              ratio={'fuel': ratio}, h_mix={'fuel': 0.0})

        # Convert back for comparison
        func_h = result.mass_avg_h / 2326.0  # Back to Btu/lbm
        func_W = result.Wout / 0.453592  # Back to lbm/s
        func_fuel_W = result.W_mix[0] / 0.453592

        tol = 1e-5
        assert_near_equal(func_h, om_h, tolerance=tol)
        assert_near_equal(func_W, om_W, tolerance=tol)
        assert_near_equal(result.composition_out, om_comp, tolerance=tol)
        assert_near_equal(func_fuel_W, om_fuel_W, tolerance=tol)

    def test_mix_2fuel(self):
        """Test mixing with two fuel ports."""
        # OpenMDAO version
        p = om.Problem()
        p.model.add_subsystem('thermo_add',
                              ThermoAddOM(spec=AIR_JETA_TAB_SPEC,
                                        inflow_composition=TAB_AIR_FUEL_COMPOSITION,
                                        mix_mode='reactant',
                                        mix_composition='FAR',
                                        mix_names=['fuel1', 'fuel2']),
                              promotes=['*'])
        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 38.8
        p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:composition'] = [0.0]

        ratio = 0.02673 / 2.
        p['fuel1:ratio'] = ratio
        p['fuel2:ratio'] = ratio

        p.run_model()

        om_h = p['mass_avg_h'][0]
        om_W = p['Wout'][0]
        om_comp = p['composition_out']

        # Functional version
        mixer = ThermoAdd(inflow_composition=TAB_AIR_FUEL_COMPOSITION,
                         mix_mode='reactant',
                         mix_composition='FAR',
                         mix_names=['fuel1', 'fuel2'])

        W = 38.8 * 0.453592
        h = 181.381769 * 2326.0
        composition = np.array([0.0])

        result = mixer.compute(W=W, h=h, composition=composition,
                              ratio={'fuel1': ratio, 'fuel2': ratio},
                              h_mix={'fuel1': 0.0, 'fuel2': 0.0})

        func_h = result.mass_avg_h / 2326.0
        func_W = result.Wout / 0.453592

        tol = 1e-5
        assert_near_equal(func_h, om_h, tolerance=tol)
        assert_near_equal(func_W, om_W, tolerance=tol)
        assert_near_equal(result.composition_out, om_comp, tolerance=tol)


class ThermoAddFlowTestCase(unittest.TestCase):
    """Test flow mode (stream mixing)."""

    def test_mix_1flow(self):
        """Test mixing with single flow stream."""
        # OpenMDAO version
        p = om.Problem()
        p.model.add_subsystem('thermo_add',
                              ThermoAddOM(spec=AIR_JETA_TAB_SPEC,
                                        inflow_composition=TAB_AIR_FUEL_COMPOSITION,
                                        mix_mode='flow',
                                        mix_names='mix'),
                              promotes=['*'])
        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 62.15
        p['Fl_I:tot:composition'] = [0.01]  # FAR = 0.01
        p['Fl_I:tot:h'] = 10.

        p['mix:W'] = 4.44635
        p['mix:composition'] = [0.0]  # Pure air
        p['mix:h'] = 5

        p.run_model()

        om_h = p['mass_avg_h'][0]
        om_W = p['Wout'][0]
        om_comp = p['composition_out']

        # Functional version
        mixer = ThermoAdd(inflow_composition=TAB_AIR_FUEL_COMPOSITION,
                         mix_mode='flow',
                         mix_names='mix')

        W = 62.15 * 0.453592
        h = 10. * 2326.0
        composition = np.array([0.01])

        W_mix = 4.44635 * 0.453592
        h_mix = 5. * 2326.0
        composition_mix = np.array([0.0])

        result = mixer.compute(W=W, h=h, composition=composition,
                              W_mix={'mix': W_mix}, h_mix={'mix': h_mix},
                              composition_mix={'mix': composition_mix})

        func_h = result.mass_avg_h / 2326.0
        func_W = result.Wout / 0.453592

        tol = 1e-5
        assert_near_equal(func_h, om_h, tolerance=tol)
        assert_near_equal(func_W, om_W, tolerance=tol)
        assert_near_equal(result.composition_out, om_comp, tolerance=tol)


class ThermoAddMassConservationTestCase(unittest.TestCase):
    """Test mass conservation properties."""

    def test_mass_conservation_reactant(self):
        """Test that total mass out equals mass in plus reactant mass."""
        mixer = ThermoAdd(mix_mode='reactant', mix_composition='FAR')

        W = 100.0  # kg/s
        h = 500000.0  # J/kg
        composition = np.array([0.0])
        ratio = 0.03

        result = mixer.compute(W=W, h=h, composition=composition,
                              ratio={'mix': ratio})

        expected_W = W * (1 + ratio)
        self.assertAlmostEqual(result.Wout, expected_W, places=10)

    def test_mass_conservation_flow(self):
        """Test that total mass out equals sum of input masses."""
        mixer = ThermoAdd(mix_mode='flow')

        W = 100.0  # kg/s
        W_mix = 25.0  # kg/s
        h = 500000.0
        h_mix = 400000.0
        composition = np.array([0.0])

        result = mixer.compute(W=W, h=h, composition=composition,
                              W_mix={'mix': W_mix}, h_mix={'mix': h_mix},
                              composition_mix={'mix': composition})

        expected_W = W + W_mix
        self.assertAlmostEqual(result.Wout, expected_W, places=10)

    def test_enthalpy_conservation(self):
        """Test that enthalpy is mass-averaged correctly."""
        mixer = ThermoAdd(mix_mode='flow')

        W = 80.0
        W_mix = 20.0
        h = 500000.0
        h_mix = 300000.0
        composition = np.array([0.0])

        result = mixer.compute(W=W, h=h, composition=composition,
                              W_mix={'mix': W_mix}, h_mix={'mix': h_mix},
                              composition_mix={'mix': composition})

        expected_h = (W * h + W_mix * h_mix) / (W + W_mix)
        self.assertAlmostEqual(result.mass_avg_h, expected_h, places=6)


class ThermoAddJAXDerivativesTestCase(unittest.TestCase):
    """Test JAX-based derivatives for Tabular ThermoAdd."""

    @classmethod
    def setUpClass(cls):
        """Check if JAX is available."""
        try:
            import jax
            cls.jax_available = True
        except ImportError:
            cls.jax_available = False

    def test_linearize_reactant_mode(self):
        """Test that linearize runs without error in reactant mode."""
        if not self.jax_available:
            self.skipTest("JAX not available")

        mixer = ThermoAdd(mix_mode='reactant', mix_composition='FAR', mix_names='fuel')

        W = 100.0
        h = 500000.0
        composition = np.array([0.0])
        ratio = 0.03

        # Should not raise
        mixer.linearize(W=W, h=h, composition=composition,
                       ratio={'fuel': ratio}, h_mix={'fuel': 0.0})

    def test_linearize_flow_mode(self):
        """Test that linearize runs without error in flow mode."""
        if not self.jax_available:
            self.skipTest("JAX not available")

        mixer = ThermoAdd(mix_mode='flow', mix_names='mix')
        composition = np.array([0.0])

        W = 100.0
        h = 500000.0

        mixer.linearize(W=W, h=h, composition=composition,
                       W_mix={'mix': 25.0}, h_mix={'mix': 400000.0},
                       composition_mix={'mix': composition})

    def test_jvp_vs_finite_difference_reactant(self):
        """Test JVP against finite difference in reactant mode."""
        if not self.jax_available:
            self.skipTest("JAX not available")

        mixer = ThermoAdd(mix_mode='reactant', mix_composition='FAR', mix_names='fuel')

        W = 100.0
        h = 500000.0
        composition = np.array([0.0])
        ratio = 0.03

        mixer.linearize(W=W, h=h, composition=composition,
                       ratio={'fuel': ratio}, h_mix={'fuel': 0.0})

        # Test derivative with respect to W
        eps = 1e-6
        result0 = mixer.compute(W=W, h=h, composition=composition,
                               ratio={'fuel': ratio}, h_mix={'fuel': 0.0})
        result1 = mixer.compute(W=W + eps, h=h, composition=composition,
                               ratio={'fuel': ratio}, h_mix={'fuel': 0.0})

        fd_Wout = (result1.Wout - result0.Wout) / eps

        jvp_result = mixer.jvp(W_dot=1.0)

        assert_near_equal(jvp_result['Wout'], fd_Wout, tolerance=1e-5)

    def test_jvp_vs_finite_difference_ratio(self):
        """Test JVP with respect to ratio against finite difference."""
        if not self.jax_available:
            self.skipTest("JAX not available")

        mixer = ThermoAdd(mix_mode='reactant', mix_composition='FAR', mix_names='fuel')

        W = 100.0
        h = 500000.0
        composition = np.array([0.0])
        ratio = 0.03

        mixer.linearize(W=W, h=h, composition=composition,
                       ratio={'fuel': ratio}, h_mix={'fuel': 0.0})

        eps = 1e-7
        result0 = mixer.compute(W=W, h=h, composition=composition,
                               ratio={'fuel': ratio}, h_mix={'fuel': 0.0})
        result1 = mixer.compute(W=W, h=h, composition=composition,
                               ratio={'fuel': ratio + eps}, h_mix={'fuel': 0.0})

        fd_Wout = (result1.Wout - result0.Wout) / eps

        jvp_result = mixer.jvp(ratio_dot={'fuel': 1.0})

        assert_near_equal(jvp_result['Wout'], fd_Wout, tolerance=1e-4)

    def test_vjp_consistency_with_jvp(self):
        """Test that VJP is consistent with JVP (transpose relationship)."""
        if not self.jax_available:
            self.skipTest("JAX not available")

        mixer = ThermoAdd(mix_mode='reactant', mix_composition='FAR', mix_names='fuel')

        W = 100.0
        h = 500000.0
        composition = np.array([0.0])
        ratio = 0.03

        mixer.linearize(W=W, h=h, composition=composition,
                       ratio={'fuel': ratio}, h_mix={'fuel': 0.0})

        # JVP with W_dot = 1
        jvp_result = mixer.jvp(W_dot=1.0)

        # VJP with unit cotangent on mass_avg_h
        vjp_result = mixer.vjp(mass_avg_h_bar=1.0)

        # The inner product <jvp(v), u> should equal <v, vjp(u)>
        assert_near_equal(jvp_result['mass_avg_h'], vjp_result['W'], tolerance=1e-10)

    def test_vjp_returns_correct_keys_reactant(self):
        """Test that VJP returns correct keys in reactant mode."""
        if not self.jax_available:
            self.skipTest("JAX not available")

        mixer = ThermoAdd(mix_mode='reactant', mix_composition='FAR', mix_names='fuel')

        mixer.linearize(W=100.0, h=500000.0, composition=np.array([0.0]),
                       ratio={'fuel': 0.03}, h_mix={'fuel': 0.0})

        vjp_result = mixer.vjp(mass_avg_h_bar=1.0)

        self.assertIn('W', vjp_result)
        self.assertIn('h', vjp_result)
        self.assertIn('composition', vjp_result)
        self.assertIn('ratio', vjp_result)
        self.assertIn('h_mix', vjp_result)
        self.assertNotIn('W_mix', vjp_result)

    def test_vjp_returns_correct_keys_flow(self):
        """Test that VJP returns correct keys in flow mode."""
        if not self.jax_available:
            self.skipTest("JAX not available")

        mixer = ThermoAdd(mix_mode='flow', mix_names='mix')
        composition = np.array([0.0])

        mixer.linearize(W=100.0, h=500000.0, composition=composition,
                       W_mix={'mix': 25.0}, h_mix={'mix': 400000.0},
                       composition_mix={'mix': composition})

        vjp_result = mixer.vjp(Wout_bar=1.0)

        self.assertIn('W', vjp_result)
        self.assertIn('h', vjp_result)
        self.assertIn('composition', vjp_result)
        self.assertIn('W_mix', vjp_result)
        self.assertIn('h_mix', vjp_result)
        self.assertIn('composition_mix', vjp_result)
        self.assertNotIn('ratio', vjp_result)


if __name__ == "__main__":
    unittest.main()
