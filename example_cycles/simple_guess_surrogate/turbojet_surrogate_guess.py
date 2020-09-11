import sys

import numpy as np

import openmdao.api as om

import pycycle.api as pyc

from data_helper import init_data, record_data_point, save_data, get_training_data, get_engine_interp


class Turbojet(pyc.Cycle):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']

        # Add engine elements
        self.pyc_add_element('fc', pyc.FlightConditions(thermo_data=thermo_spec,
                                    elements=pyc.AIR_MIX))
        self.pyc_add_element('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec,
                                    elements=pyc.AIR_MIX))
        self.pyc_add_element('comp', pyc.Compressor(map_data=pyc.AXI5, design=design,
                                    thermo_data=thermo_spec, elements=pyc.AIR_MIX, map_extrap=True),
                                    promotes_inputs=['Nmech'])
        self.pyc_add_element('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                    inflow_elements=pyc.AIR_MIX,
                                    air_fuel_elements=pyc.AIR_FUEL_MIX,
                                    fuel_type='JP-7'))
        self.pyc_add_element('turb', pyc.Turbine(map_data=pyc.LPT2269, design=design,
                                    thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,),
                                    promotes_inputs=['Nmech'])
        self.pyc_add_element('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv',
                                    thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.pyc_add_element('shaft', pyc.Shaft(num_ports=2),promotes_inputs=['Nmech'])
        self.pyc_add_element('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # Connect flow stations
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O', 'nozz.Fl_I')

        # Make other non-flow connections
        # Connect turbomachinery elements to shaft
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')

        # Connnect nozzle exhaust to freestream static conditions
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        # Connect outputs to perfomance element
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        # Add balances for design and off-design
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('W', units='lbm/s', eq_units='lbf', rhs_name='Fn_target')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('perf.Fn', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017, rhs_name='T4_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('turb_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('shaft.pwr_net', 'balance.lhs:turb_PR')

        else:

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.3, rhs_name='T4_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('Nmech', val=1.5, units='rpm', lower=0.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

            balance.add_balance('W', val=168.0, units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

        # Setup solver to converge engine
        self.set_order(['fc', 'inlet', 'comp', 'burner', 'turb', 'nozz', 'shaft', 'perf', 'balance'])

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        
        # newton.linesearch.options['print_bound_enforce'] = True
        
        self.linear_solver = om.DirectSolver()

def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    summary_data = (prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'], prob[pt+'.inlet.Fl_O:stat:W'], 
                    prob[pt+'.perf.Fn'], prob[pt+'.perf.Fg'], prob[pt+'.inlet.F_ram'],
                    prob[pt+'.perf.OPR'], prob[pt+'.perf.TSFC'])

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC  ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" %summary_data, file=file, flush=True)


    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'comp.Fl_O', 'burner.Fl_O',
                'turb.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['comp']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['turb']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)


class MPTurbojet(pyc.MPCycle):

    def setup(self):

        # Create design instance of model
        self.pyc_add_pnt('DESIGN', Turbojet())

        self.set_input_defaults('DESIGN.Nmech', 1, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN', 0.60)
        self.set_input_defaults('DESIGN.comp.MN', 0.020)#.2
        self.set_input_defaults('DESIGN.burner.MN', 0.020)#.2
        self.set_input_defaults('DESIGN.turb.MN', 0.4)

        self.pyc_add_cycle_param('burner.dPqP', 0.03)
        self.pyc_add_cycle_param('nozz.Cv', 0.99)

        #Define the design point
        self.set_input_defaults('DESIGN.fc.alt', 0, units='ft')
        self.set_input_defaults('DESIGN.fc.MN', 0.000001)
        self.set_input_defaults('DESIGN.balance.Fn_target', 11800.0, units='lbf')
        self.set_input_defaults('DESIGN.balance.T4_target', 2370.0, units='degR') 
        self.set_input_defaults('DESIGN.comp.PR', 13.5) 
        self.set_input_defaults('DESIGN.comp.eff', 0.83)
        self.set_input_defaults('DESIGN.turb.eff', 0.86)

        # define the single off-design point. 
        # Use loop around run_model walk this 
        # single point across the flight envelop
        self.pyc_add_pnt('OD0', Turbojet(design=False))

        self.set_input_defaults('OD0.fc.MN', val=0.00001)
        self.set_input_defaults('OD0.fc.alt', 0.0, units='ft')
        self.set_input_defaults('OD0.balance.T4_target', 2370.0, units='degR')  

        self.pyc_connect_des_od('comp.s_PR', 'comp.s_PR')
        self.pyc_connect_des_od('comp.s_Wc', 'comp.s_Wc')
        self.pyc_connect_des_od('comp.s_eff', 'comp.s_eff')
        self.pyc_connect_des_od('comp.s_Nc', 'comp.s_Nc')

        self.pyc_connect_des_od('turb.s_PR', 'turb.s_PR')
        self.pyc_connect_des_od('turb.s_Wp', 'turb.s_Wp')
        self.pyc_connect_des_od('turb.s_eff', 'turb.s_eff')
        self.pyc_connect_des_od('turb.s_Np', 'turb.s_Np')

        self.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.pyc_connect_des_od('comp.Fl_O:stat:area', 'comp.area')
        self.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        self.pyc_connect_des_od('turb.Fl_O:stat:area', 'turb.area')

        self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

if __name__ == "__main__":

    import time
    import sys

    prob = om.Problem()

    prob.model = mp_turbojet = MPTurbojet()

    prob.setup(check=False)

    # Set initial guesses for design balances
    prob['DESIGN.balance.FAR'] = 0.015
    prob['DESIGN.balance.W'] = 150
    prob['DESIGN.balance.turb_PR'] = 4.0

    
    # initial guesses OD balances
    W_guess = 150
    FAR_guess = 0.015
    Nmech_guess = 1.
    PR_guess = 4.6690

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    if len(sys.argv) > 1 and sys.argv[1] == 'training_data': 
        input_vars = ['OD0.fc.MN', 'OD0.fc.alt']
        state_vars =['OD0.balance.W', 'OD0.balance.FAR', 'OD0.balance.Nmech', 'OD0.turb.PR']

        # initialize the dict storage for the surrogate data
        OD_data = init_data(input_vars, state_vars)

        MNs = np.linspace(0.001, 0.3, 3)
        alts = np.linspace(0, 10000, 3)
        for alt in alts: 

            # restore the converged values from the last MN=0 point, 
            # then increment the altitude and start again
            prob['OD0.balance.W'] = W_guess 
            prob['OD0.balance.FAR'] = FAR_guess 
            prob['OD0.balance.Nmech'] = Nmech_guess
            prob['OD0.turb.PR'] = PR_guess 
            

            for MN in MNs: 
                prob['OD0.fc.MN'] = MN
                prob['OD0.fc.alt'] = alt
                print()
                print(50*'#')
                print(f'running MN={MN}, alt={alt}')
                prob.run_model()

                record_data_point(OD_data, prob)

                if MN < .01: # save off guesses for low-mach
                    W_guess = prob['OD0.balance.W'][0]
                    FAR_guess = prob['OD0.balance.FAR'][0] 
                    Nmech_guess = prob['OD0.balance.Nmech'][0] 
                    PR_guess = prob['OD0.turb.PR'][0]

        save_data(OD_data, 'test.pkl')

        for pt in ['DESIGN', 'OD0']:
            viewer(prob, pt)


    # grab the data and train a surrogate

    xt, yt, xlimits, scaling_factors = get_training_data('test.pkl')

    interp = get_engine_interp(xt, yt, xlimits, scaling_factors)

    xp = np.array([[0.25, 1200]])

    print(interp.predict_values(xp))

    print("time", time.time() - st)