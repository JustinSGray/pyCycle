"""Tests the NewFlowStart component."""

import unittest
import os

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.new_elements.flow_start import NewFlowStart
from pycycle.thermo.cea import species_data
from pycycle.constants import (CEA_AIR_COMPOSITION, CEA_WET_AIR_COMPOSITION,
                               AIR_JETA_TAB_SPEC, TAB_AIR_FUEL_COMPOSITION)


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(os.path.join(fpath, '..', '..', 'test_data', 'flowstart.csv'),
                      delimiter=",", skiprows=1)

header = [
    'W', 'MN', 'V', 'A', 's', 'Pt', 'Tt', 'ht', 'rhot', 'gamt',
    'Ps', 'Ts', 'hs', 'rhos', 'gams']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class NewFlowStartTestCase(unittest.TestCase):

    def test_cea(self):

        prob = Problem()
        prob.model.set_input_defaults('fl_start.P', 17., units='psi')
        prob.model.set_input_defaults('fl_start.T', 500., units='degR')
        prob.model.set_input_defaults('fl_start.MN', 0.5)
        prob.model.set_input_defaults('fl_start.W', 100., units='lbm/s')

        fl_start = prob.model.add_subsystem('fl_start',
                                            NewFlowStart(thermo_method='CEA',
                                                         thermo_data=species_data.janaf,
                                                         composition=CEA_AIR_COMPOSITION))
        fl_start.pyc_setup_output_ports()

        prob.set_solver_print(level=-1)
        prob.setup(check=False, force_alloc_complex=True)

        # 6 cases to check against
        for i, data in enumerate(ref_data):
            prob.set_val('fl_start.P', data[h_map['Pt']], units='psi')
            prob['fl_start.T'] = data[h_map['Tt']]
            prob['fl_start.W'] = data[h_map['W']]
            prob['fl_start.MN'] = data[h_map['MN']]

            prob.run_model()

            tol = 1.0e-3
            # The Mach 2.0 case is at a ridiculously low temperature, so accuracy is questionable
            if data[h_map['MN']] >= 2.:
                tol = 5e-2

            assert_near_equal(prob['fl_start.Fl_O:tot:P'], data[h_map['Pt']], tol)
            assert_near_equal(prob['fl_start.Fl_O:tot:T'], data[h_map['Tt']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:W'], data[h_map['W']], tol)
            assert_near_equal(prob['fl_start.Fl_O:tot:h'], data[h_map['ht']], tol)
            assert_near_equal(prob['fl_start.Fl_O:tot:S'], data[h_map['s']], tol)
            assert_near_equal(prob['fl_start.Fl_O:tot:rho'], data[h_map['rhot']], tol)
            assert_near_equal(prob['fl_start.Fl_O:tot:gamma'], data[h_map['gamt']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:MN'], data[h_map['MN']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:P'], data[h_map['Ps']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:T'], data[h_map['Ts']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:h'], data[h_map['hs']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:rho'], data[h_map['rhos']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:gamma'], data[h_map['gams']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:V'], data[h_map['V']], tol)
            assert_near_equal(prob['fl_start.Fl_O:stat:area'], data[h_map['A']], tol)

    def test_tabular(self):

        prob = Problem()
        prob.model.set_input_defaults('fl_start.P', 17., units='psi')
        prob.model.set_input_defaults('fl_start.T', 500., units='degR')
        prob.model.set_input_defaults('fl_start.MN', 0.5)
        prob.model.set_input_defaults('fl_start.W', 100., units='lbm/s')

        fl_start = prob.model.add_subsystem('fl_start',
                                            NewFlowStart(thermo_method='TABULAR',
                                                         thermo_data=AIR_JETA_TAB_SPEC,
                                                         composition=TAB_AIR_FUEL_COMPOSITION))
        fl_start.pyc_setup_output_ports()

        prob.set_solver_print(level=-1)
        prob.setup(check=False, force_alloc_complex=True)

        prob['fl_start.P'] = 5.27
        prob['fl_start.T'] = 444.23
        prob['fl_start.W'] = 100.0
        prob['fl_start.MN'] = 0.8

        prob.run_model()

        tol = 1e-2
        assert_near_equal(prob['fl_start.Fl_O:tot:P'], 5.27, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:T'], 444.23, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:h'], -24.02365656, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:S'], 1.66403163, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:gamma'], 1.40086187, tol)

        assert_near_equal(prob['fl_start.Fl_O:stat:W'], 100.0, tol)
        assert_near_equal(prob['fl_start.Fl_O:stat:MN'], 0.8, tol)
        assert_near_equal(prob['fl_start.Fl_O:stat:area'], 778.26812382, tol)

    def test_tabular_partials(self):

        prob = Problem()
        prob.model.set_input_defaults('fl_start.P', 17., units='psi')
        prob.model.set_input_defaults('fl_start.T', 500., units='degR')
        prob.model.set_input_defaults('fl_start.MN', 0.5)
        prob.model.set_input_defaults('fl_start.W', 100., units='lbm/s')

        fl_start = prob.model.add_subsystem('fl_start',
                                            NewFlowStart(thermo_method='TABULAR',
                                                         thermo_data=AIR_JETA_TAB_SPEC,
                                                         composition=TAB_AIR_FUEL_COMPOSITION))
        fl_start.pyc_setup_output_ports()

        prob.set_solver_print(level=-1)
        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        partial_data = prob.check_partials(out_stream=None, method='cs',
                                           includes=['fl_start'])
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class NewFlowStartWARTestCase(unittest.TestCase):

    def test_cea_with_water(self):

        prob = Problem()
        prob.model.set_input_defaults('fl_start.P', 17., units='psi')
        prob.model.set_input_defaults('fl_start.T', 500., units='degR')
        prob.model.set_input_defaults('fl_start.MN', 0.5)
        prob.model.set_input_defaults('fl_start.W', 100., units='lbm/s')
        prob.model.set_input_defaults('fl_start.WAR', .01)

        fl_start = prob.model.add_subsystem('fl_start',
                                            NewFlowStart(thermo_method='CEA',
                                                         thermo_data=species_data.wet_air,
                                                         composition=CEA_WET_AIR_COMPOSITION,
                                                         reactant="Water",
                                                         mix_ratio_name='WAR'))
        fl_start.pyc_setup_output_ports()

        prob.set_solver_print(level=-1)
        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][0], 3.18139345e-04, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][1], 1.08367806e-05, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][2], 1.77859e-03, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][3], 5.305198e-02, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][4], 1.51432e-02, tol)


if __name__ == "__main__":
    unittest.main()
