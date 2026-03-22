"""Tests the NewCombustor component."""

import unittest
import os

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.mp_cycle import Cycle
from pycycle.elements.flow_start import FlowStart
from pycycle.new_elements.combustor import NewCombustor
from pycycle.thermo.cea import species_data
from pycycle.constants import AIR_JETA_TAB_SPEC


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(os.path.join(fpath, '..', '..', 'test_data', 'combustorJP7.csv'),
                      delimiter=",", skiprows=1)

header = ['Fl_I.W', 'Fl_I.Pt', 'Fl_I.Tt', 'Fl_I.ht', 'Fl_I.s', 'Fl_I.MN', 'FAR', 'eff',
          'Fl_O.MN', 'Fl_O.Pt', 'Fl_O.Tt', 'Fl_O.ht', 'Fl_O.s', 'Wfuel',
          'Fl_O.Ps', 'Fl_O.Ts', 'Fl_O.hs', 'Fl_O.rhos', 'Fl_O.gams']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class NewCombustorCEATestCase(unittest.TestCase):

    def test_cea_values(self):
        prob = Problem()
        cycle = prob.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = species_data.janaf

        cycle.add_subsystem('flow_start', FlowStart())
        cycle.add_subsystem('combustor', NewCombustor())

        cycle.pyc_connect_flow('flow_start.Fl_O', 'combustor.Fl_I')

        cycle.set_input_defaults('combustor.Fl_I:FAR', 0.0)
        cycle.set_input_defaults('combustor.MN', 0.5)

        prob.set_solver_print(level=-1)
        prob.setup(check=False, force_alloc_complex=True)

        for i, data in enumerate(ref_data):
            prob.set_val('flow_start.P', data[h_map['Fl_I.Pt']], units='psi')
            prob.set_val('flow_start.T', data[h_map['Fl_I.Tt']], units='degR')
            prob.set_val('flow_start.W', data[h_map['Fl_I.W']], units='lbm/s')
            prob.set_val('flow_start.MN', data[h_map['Fl_I.MN']])
            prob['combustor.Fl_I:FAR'] = data[h_map['FAR']]
            prob['combustor.MN'] = data[h_map['Fl_O.MN']]

            prob.run_model()

            tol = 1e-2
            assert_near_equal(prob['combustor.Fl_O:tot:P'], data[h_map['Fl_O.Pt']], tol)
            assert_near_equal(prob['combustor.Fl_O:tot:T'], data[h_map['Fl_O.Tt']], tol)
            assert_near_equal(prob['combustor.Fl_O:tot:h'], data[h_map['Fl_O.ht']], tol)
            assert_near_equal(prob['combustor.Fl_O:stat:P'], data[h_map['Fl_O.Ps']], tol)
            assert_near_equal(prob['combustor.Fl_O:stat:T'], data[h_map['Fl_O.Ts']], tol)

            Wfuel_expected = data[h_map['Fl_I.W']] * data[h_map['FAR']]
            assert_near_equal(prob['combustor.Wfuel'], Wfuel_expected, tol)


class NewCombustorTabularTestCase(unittest.TestCase):

    def test_tabular_values(self):
        prob = Problem()
        cycle = prob.model = Cycle()
        cycle.options['thermo_method'] = 'TABULAR'
        cycle.options['thermo_data'] = AIR_JETA_TAB_SPEC

        cycle.add_subsystem('flow_start', FlowStart())
        cycle.add_subsystem('combustor', NewCombustor())

        cycle.pyc_connect_flow('flow_start.Fl_O', 'combustor.Fl_I')

        cycle.set_input_defaults('combustor.Fl_I:FAR', 0.0)
        cycle.set_input_defaults('combustor.MN', 0.5)

        prob.set_solver_print(level=-1)
        prob.setup(check=False, force_alloc_complex=True)

        # Use first reference case
        data = ref_data[0]
        prob.set_val('flow_start.P', data[h_map['Fl_I.Pt']], units='psi')
        prob.set_val('flow_start.T', data[h_map['Fl_I.Tt']], units='degR')
        prob.set_val('flow_start.W', data[h_map['Fl_I.W']], units='lbm/s')
        prob.set_val('flow_start.MN', data[h_map['Fl_I.MN']])
        prob['combustor.Fl_I:FAR'] = data[h_map['FAR']]
        prob['combustor.MN'] = data[h_map['Fl_O.MN']]

        prob.run_model()

        # Tabular tolerances are looser than CEA
        tol = 2e-2
        assert_near_equal(prob['combustor.Fl_O:tot:P'], data[h_map['Fl_O.Pt']], tol)
        assert_near_equal(prob['combustor.Fl_O:tot:T'], data[h_map['Fl_O.Tt']], tol)

        Wfuel_expected = data[h_map['Fl_I.W']] * data[h_map['FAR']]
        assert_near_equal(prob['combustor.Wfuel'], Wfuel_expected, tol)

    def test_tabular_partials(self):
        prob = Problem()
        cycle = prob.model = Cycle()
        cycle.options['thermo_method'] = 'TABULAR'
        cycle.options['thermo_data'] = AIR_JETA_TAB_SPEC

        cycle.add_subsystem('flow_start', FlowStart())
        cycle.add_subsystem('combustor', NewCombustor())

        cycle.pyc_connect_flow('flow_start.Fl_O', 'combustor.Fl_I')

        cycle.set_input_defaults('combustor.Fl_I:FAR', 0.02)
        cycle.set_input_defaults('combustor.MN', 0.5)
        cycle.set_input_defaults('flow_start.P', 158., units='psi')
        cycle.set_input_defaults('flow_start.T', 1278., units='degR')
        cycle.set_input_defaults('flow_start.W', 38.8, units='lbm/s')
        cycle.set_input_defaults('flow_start.MN', 0.3)

        prob.set_solver_print(level=-1)
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        partial_data = prob.check_partials(out_stream=None, method='cs',
                                           includes=['combustor'])
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
