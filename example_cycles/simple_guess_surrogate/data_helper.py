from __future__ import print_function, division
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
import os
from smt.surrogate_models import RMTB, RMTC, KRG


guess_vars = [
    'OD0.balance.W', 'OD0.balance.FAR', 'OD0.balance.Nmech', 'OD0.turb.PR'
]



def init_data(inputs, states, pkl_file_name=None): 
    """
    initialize the dictionary that stores the surrogate training data

    Parameters
    ----------
    inputs: list of string
        names of the variables that will be used as surrogate inputs
    states: list of string
        names of the variables that will be used as surrogate outputs
    pkl_file_name: string
        name of an existing pickle file that should be used to initialize the data 

    Returns
    -------
    nested dictionary
    """

    if pkl_file_name is not None: 
        with open(pkl_file_name, 'rb') as f: 
            data = pickle.load(f)
        # check that the data matches the inputs/states given 
        pkl_inputs = set(data['inputs'].keys())
        if pkl_inputs != set(inputs): 
            raise ValueError('inputs in pkl_file do not match the specified inputs.')
        pkl_states = set(data['states'].keys())
        if pkl_states != set(states): 
            raise ValueError('states in pkl_file do not match the specified states.')
    else: # initialize an empty dict instead
        data = {'inputs':{},  'states':{}}
        for inp in inputs:
            data['inputs'][inp] = []
        for st in states:
            data['states'][st] = []

    return data


def record_data_point(data, problem): 
    """records a new point into the data""" 

    for k,v in data['inputs'].items(): 
        v.append(problem[k][0])
    for k,v in data['states'].items(): 
        v.append(problem[k][0])


def save_data(data, pkl_file_name): 

    with open(pkl_file_name, 'wb') as f: 
        pickle.dump(data, f)


def get_training_data(pkl_file_name):
    this_dir = os.path.split(__file__)[0]

    with open(pkl_file_name, 'rb') as f:
        data = pickle.load(f)


    in_data = [v for k,v in data['inputs'].items()]
    xt = np.vstack(in_data).T

    state_vars = sorted(data['states'].keys())
    state_data = data['states']
    yt = np.vstack([state_data[k] for k in state_vars]).T

    # Scale inputs
    xt[:, 1] /= 1e4

    # Get the min and max of all the outputs so we can scale the training data.
    # We'll pass these scaling factors back out so we can unscale the data
    # after we query the interpolant.
    scaling_factors = np.vstack((np.min(yt, axis=0), np.max(yt, axis=0))).T

    # Actaully scale the output training data such that all the outputs are
    # between 0 and 1.

    yt = (yt - scaling_factors[:, 0]) / (scaling_factors[:, 1] - scaling_factors[:, 0])

    # Set the limits for the inputs; Mach and scaled altitude
    xlimits = np.array([
        [-0.02, 2.52],
        [-0.1, 1.2],
    ])

    return xt, yt, xlimits, scaling_factors

def get_engine_interp(xt, yt, xlimits, scaling_factors):
    this_dir = os.path.split(__file__)[0]

    # Create the _smt_cache directory if it doesn't exist
    if not os.path.exists(os.path.join(this_dir, '_smt_cache')):
        os.makedirs(os.path.join(this_dir, '_smt_cache'))

    interp = RMTB(xlimits=xlimits, num_ctrl_pts=10, order=3,
        approx_order=2, nonlinear_maxiter=40, solver_tolerance=1.e-20,
        # solver='lu', derivative_solver='lu',
        energy_weight=1.e-4, regularization_weight=1.e-5, extrapolate=False, print_global=False,
        data_dir=os.path.join(this_dir, '_smt_cache'),
    )

    interp.set_training_values(xt, yt)
    interp.train()

    return interp




if __name__ == "__main__":
    from postprocessing.MultiView.MultiView import MultiView

    xt, yt, xlimits, scaling_factors = get_training_data('test.pkl')
    interp  = get_engine_interp(xt, yt, xlimits, scaling_factors)

    info = {'nx':2,
            'ny':23,
            'user_func':interp.predict_values,
            'resolution':100,
            'plot_size':12,
            'dimension_names':[
                'Mach number',
                'Altitude, 10k ft'],
            'bounds':xlimits.tolist(),
            'X_dimension':0,
            'Y_dimension':1,
            'scatter_points':[xt, yt],
            'dist_range': .1,
            'output_names' : guess_vars,
            }

    # Initialize display parameters and draw GUI
    MultiView(info)