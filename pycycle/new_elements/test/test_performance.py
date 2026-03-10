"""Tests for the NewPerformance component."""

import unittest

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.new_elements.performance import NewPerformance


class NewPerformanceTestCase(unittest.TestCase):
    """Test basic performance calculations with 1 nozzle, 1 burner."""

    def setUp(self):
        self.prob = Problem()
        self.prob.model.add_subsystem('perf', NewPerformance(), promotes=['*'])

        self.prob.model.set_input_defaults('Pt2', 14.696, units='lbf/inch**2')
        self.prob.model.set_input_defaults('Pt3', 44.088, units='lbf/inch**2')
        self.prob.model.set_input_defaults('ram_drag', 100.0, units='lbf')
        self.prob.model.set_input_defaults('power', 200.0, units='hp')
        self.prob.model.set_input_defaults('Fg_0', 1200.0, units='lbf')
        self.prob.model.set_input_defaults('Wfuel_0', 2.0, units='lbm/s')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_values(self):
        prob = self.prob
        prob.run_model()

        assert_near_equal(prob['OPR'], 44.088 / 14.696, 1e-10)
        assert_near_equal(prob['Fg'], 1200.0, 1e-10)
        assert_near_equal(prob['Fn'], 1100.0, 1e-10)
        assert_near_equal(prob['Wfuel'], 2.0, 1e-10)
        assert_near_equal(prob['TSFC'], 2.0 * 3600.0 / 1100.0, 1e-6)
        assert_near_equal(prob['PSFC'], 2.0 * 3600.0 / 200.0, 1e-10)

    def test_partials(self):
        self.prob.run_model()
        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class NewPerformanceMultiNozzleTestCase(unittest.TestCase):
    """Test with 2 nozzles and 1 burner."""

    def test_values_and_partials(self):
        prob = Problem()
        prob.model.add_subsystem('perf', NewPerformance(num_nozzles=2), promotes=['*'])

        prob.model.set_input_defaults('Pt2', 204.696, units='lbf/inch**2')
        prob.model.set_input_defaults('Pt3', 104.696, units='lbf/inch**2')
        prob.model.set_input_defaults('ram_drag', 100.0, units='lbf')
        prob.model.set_input_defaults('power', 200.0, units='hp')
        prob.model.set_input_defaults('Fg_0', 1200.0, units='lbf')
        prob.model.set_input_defaults('Fg_1', 2000.0, units='lbf')
        prob.model.set_input_defaults('Wfuel_0', 2.0, units='lbm/s')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob['OPR'], 104.696 / 204.696, 1e-10)
        assert_near_equal(prob['Fg'], 3200.0, 1e-10)
        assert_near_equal(prob['Fn'], 3100.0, 1e-10)
        assert_near_equal(prob['Wfuel'], 2.0, 1e-10)
        assert_near_equal(prob['TSFC'], 2.0 * 3600.0 / 3100.0, 1e-6)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class NewPerformanceMultiBurnerTestCase(unittest.TestCase):
    """Test with 2 nozzles and 2 burners."""

    def test_values_and_partials(self):
        prob = Problem()
        prob.model.add_subsystem('perf',
                                 NewPerformance(num_nozzles=2, num_burners=2),
                                 promotes=['*'])

        prob.model.set_input_defaults('Pt2', 14.696, units='lbf/inch**2')
        prob.model.set_input_defaults('Pt3', 73.48, units='lbf/inch**2')
        prob.model.set_input_defaults('ram_drag', 500.0, units='lbf')
        prob.model.set_input_defaults('power', 1000.0, units='hp')
        prob.model.set_input_defaults('Fg_0', 5000.0, units='lbf')
        prob.model.set_input_defaults('Fg_1', 3000.0, units='lbf')
        prob.model.set_input_defaults('Wfuel_0', 1.5, units='lbm/s')
        prob.model.set_input_defaults('Wfuel_1', 0.8, units='lbm/s')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob['OPR'], 73.48 / 14.696, 1e-10)
        assert_near_equal(prob['Fg'], 8000.0, 1e-10)
        assert_near_equal(prob['Fn'], 7500.0, 1e-10)
        assert_near_equal(prob['Wfuel'], 2.3, 1e-10)
        assert_near_equal(prob['TSFC'], 2.3 * 3600.0 / 7500.0, 1e-6)
        assert_near_equal(prob['PSFC'], 2.3 * 3600.0 / 1000.0, 1e-10)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class NewPerformanceNoBurnersTestCase(unittest.TestCase):
    """Test with 0 burners (no fuel outputs)."""

    def test_values_and_partials(self):
        prob = Problem()
        prob.model.add_subsystem('perf',
                                 NewPerformance(num_nozzles=1, num_burners=0),
                                 promotes=['*'])

        prob.model.set_input_defaults('Pt2', 14.696, units='lbf/inch**2')
        prob.model.set_input_defaults('Pt3', 44.088, units='lbf/inch**2')
        prob.model.set_input_defaults('ram_drag', 100.0, units='lbf')
        prob.model.set_input_defaults('Fg_0', 1200.0, units='lbf')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob['OPR'], 44.088 / 14.696, 1e-10)
        assert_near_equal(prob['Fg'], 1200.0, 1e-10)
        assert_near_equal(prob['Fn'], 1100.0, 1e-10)

        # Should not have fuel-related outputs
        self.assertFalse(prob.model.perf._var_allprocs_abs2meta['output'].get('perf.TSFC'))

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
