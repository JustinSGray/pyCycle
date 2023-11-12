import unittest
import os

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.thermo import ThermoStation, Thermo
from pycycle.thermo.cea import species_data
from pycycle import constants



class SetTotalSimpleTestCase(unittest.TestCase):

    def test_set_total_TP(self):        
        thermo = ThermoStation("flow", mode="total_TP", 
                               method="CEA", 
                               thermo_kwargs={'composition': constants.CEA_CO2_CO_O2_COMPOSITION, 
                                              'spec': species_data.co2_co_o2 })

        
        T = om.convert_units(4000, "degK", "degR") 
        P = om.convert_units(1.034210, "bar", "lbf/inch**2") 
        flow_data = thermo.compute(T=T, P=P) 

        print(flow_data)

        gamma = flow_data[4]      
        assert_near_equal(gamma, 1.19054697, 1e-4)

        # T = om.convert_units(1500, "degK", "degR") 
        # P = om.convert_units(1.034210, "bar", "lbf/inch**2") 
        # flow_data = thermo.compute(T=T, P=P)

        # gamma = flow_data[5]      
        # assert_near_equal(gamma, 1.16379233, 1e-4)


    def _test_set_total_TP(self):
        p = om.Problem()
        p.model.add_subsystem('thermo', Thermo(mode='total_TP', 
                                               method='CEA', 
                                               thermo_kwargs={'composition': constants.CEA_CO2_CO_O2_COMPOSITION, 
                                                              'spec': species_data.co2_co_o2 }), 
                              promotes=['*'])


        p.setup(check=False)
        p.set_solver_print(level=-1)
        p.final_setup()

        # p.set_val('T', 4000, units='degK')
        # p.set_val('P', 1.034210, units='bar')

        p.set_val('T', 7200, units='degR')
        p.set_val('P', 14.9999, units='lbf/inch**2')
        
        p.run_model()

        p.model.list_outputs()

        # assert_near_equal(p['gamma'], 1.19054697, 1e-4)

        # p.set_val('T', 1500, units='degK')
        # p.set_val('P', 1.034210, units='bar')
        # p.run_model()

        # assert_near_equal(p['gamma'], 1.16379233, 1e-4)


if __name__ == "__main__": 
    unittest.main()