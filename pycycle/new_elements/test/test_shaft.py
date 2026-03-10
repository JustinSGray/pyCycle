"""Tests the NewShaft component."""

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.new_elements.shaft import NewShaft


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(os.path.join(fpath, '..', '..', 'test_data', 'shaft.csv'),
                      delimiter=",", skiprows=1)

header = [
    'trqLoad1',
    'trqLoad2',
    'trqLoad3',
    'Nmech',
    'HPX',
    'fracLoss',
    'trqIn',
    'trqOut',
    'trqNet',
    'pwrIn',
    'pwrOut',
    'pwrNet']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class NewShaftTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()
        self.prob.model = Group()
        self.prob.model.add_subsystem("shaft", NewShaft(num_ports=3), promotes=["*"])

        self.prob.model.set_input_defaults('trq_0', 17., units='ft*lbf')
        self.prob.model.set_input_defaults('trq_1', 17., units='ft*lbf')
        self.prob.model.set_input_defaults('trq_2', 17., units='ft*lbf')
        self.prob.model.set_input_defaults('Nmech', 17., units='rpm')
        self.prob.model.set_input_defaults('HPX', 17., units='hp')
        self.prob.model.set_input_defaults('fracLoss', 17.)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_regression(self):
        """Test against NPSS reference data with 3 ports."""
        for i, data in enumerate(ref_data):
            # input torques
            self.prob['trq_0'] = data[h_map['trqLoad1']]
            self.prob['trq_1'] = data[h_map['trqLoad2']]
            self.prob['trq_2'] = data[h_map['trqLoad3']]

            # shaft inputs
            self.prob['Nmech'] = data[h_map['Nmech']]
            self.prob['HPX'] = data[h_map['HPX']]
            self.prob['fracLoss'] = data[h_map['fracLoss']]
            self.prob.run_model()

            # check outputs
            tol = 1.0e-4
            assert_near_equal(self.prob['trq_in'], data[h_map['trqIn']], tol)
            assert_near_equal(self.prob['trq_out'], data[h_map['trqOut']], tol)
            assert_near_equal(self.prob['trq_net'], data[h_map['trqNet']], tol)
            assert_near_equal(self.prob['pwr_in'], data[h_map['pwrIn']], tol)
            assert_near_equal(self.prob['pwr_out'], data[h_map['pwrOut']], tol)
            assert_near_equal(self.prob['pwr_net'], data[h_map['pwrNet']], tol)

            partial_data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class NewShaftTwoPortTestCase(unittest.TestCase):
    """Test with default 2 ports to verify variable port count works."""

    def test_two_ports(self):
        prob = Problem()
        prob.model.add_subsystem("shaft", NewShaft(num_ports=2), promotes=["*"])

        prob.model.set_input_defaults('trq_0', 100., units='ft*lbf')
        prob.model.set_input_defaults('trq_1', -50., units='ft*lbf')
        prob.model.set_input_defaults('Nmech', 5000., units='rpm')

        prob.setup(check=False, force_alloc_complex=True)

        prob['trq_0'] = 500.0
        prob['trq_1'] = -300.0
        prob['Nmech'] = 8000.0
        prob['HPX'] = 0.0
        prob['fracLoss'] = 0.0
        prob.run_model()

        assert_near_equal(prob['trq_in'], 500.0, 1e-10)
        assert_near_equal(prob['trq_out'], -300.0, 1e-10)
        assert_near_equal(prob['trq_net'], 200.0, 1e-10)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class NewShaftFivePortTestCase(unittest.TestCase):
    """Test with 5 ports to verify larger port counts work."""

    def test_five_ports(self):
        prob = Problem()
        prob.model.add_subsystem("shaft", NewShaft(num_ports=5), promotes=["*"])

        prob.model.set_input_defaults('Nmech', 10000., units='rpm')

        prob.setup(check=False, force_alloc_complex=True)

        # Mix of positive and negative torques
        prob['trq_0'] = 1000.0
        prob['trq_1'] = -400.0
        prob['trq_2'] = 500.0
        prob['trq_3'] = -300.0
        prob['trq_4'] = -200.0
        prob['Nmech'] = 10000.0
        prob['HPX'] = 100.0
        prob['fracLoss'] = 0.02
        prob.run_model()

        # trq_in = 1000 + 500 = 1500
        # trq_out = -400 + -300 + -200 = -900
        assert_near_equal(prob['trq_in'], 1500.0, 1e-10)
        assert_near_equal(prob['trq_out'], -900.0, 1e-10)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
