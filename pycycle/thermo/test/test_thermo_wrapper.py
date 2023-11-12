import unittest
import os

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.thermo import ThermoStation
from pycycle.thermo.cea import species_data
from pycycle import constants



class SetTotalSimpleTestCase(unittest.TestCase):

    def test_set_total_TP(self):        
        thermo = ThermoStation("flow", mode="total_TP", 
                               method="CEA", 
                               thermo_kwargs={'composition': constants.CEA_CO2_CO_O2_COMPOSITION, 
                                              'spec': species_data.co2_co_o2 })

        
        thermo.set_val('T', 4000, units='degK')
        thermo.set_val('P', 1.034210, units='bar')
        thermo.compute()       
        assert_near_equal(thermo.get_val('gamma'), 1.19054697, 1e-4)

        thermo.set_val('T', 1500, units='degK')
        thermo.set_val('P', 1.034210, units='bar')
        thermo.compute()
        assert_near_equal(thermo.get_val('gamma'), 1.16379233, 1e-4)


if __name__ == "__main__": 
    unittest.main()