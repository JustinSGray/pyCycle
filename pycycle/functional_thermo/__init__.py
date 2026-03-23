"""
Functional interface for thermodynamic property calculations.

This module provides a common interface for computing thermodynamic properties
using different methods (Tabular, CEA). The interface is designed to be
compatible with JAX automatic differentiation.

The thermo classes work internally in SI units but can accept inputs and
return outputs in different unit systems via the `input_units` parameter:
    - 'SI': SI units (default)
    - 'English': pyCycle English units (degR, psi, Btu/lbm, etc.)

Internal Units (SI):
    T - Temperature (K)
    P - Pressure (Pa)
    h - Enthalpy (J/kg)
    S - Entropy (J/(kg*K))
    Cp - Specific heat at constant pressure (J/(kg*K))
    Cv - Specific heat at constant volume (J/(kg*K))
    gamma - Ratio of specific heats (-)
    rho - Density (kg/m^3)
    R - Specific gas constant (J/(kg*K))
    W - Mass flow rate (kg/s)
    V - Velocity (m/s)
    area - Flow area (m^2)
    MN - Mach number (-)
"""

from .base import ThermoInterface, TotalProps, StaticProps
from .units import UnitConverter, SI_UNITS, ENGLISH_UNITS
from .tabular import ThermoAdd as TabularThermoAdd
from .tabular import ThermoAddOutput as TabularThermoAddOutput
from .cea import JaxCEAThermo
from .cea import ThermoAdd, ThermoAddOutput  # CEA ThermoAdd is the default

# Explicit aliases for clarity
CEAThermoAdd = ThermoAdd
CEAThermoAddOutput = ThermoAddOutput


def create_jax_thermo(thermo_method, thermo_data, composition=None):
    """Factory function to create a JAX thermo object for the given method.

    Parameters
    ----------
    thermo_method : str
        'CEA' or 'TABULAR'
    thermo_data : object
        Thermodynamic data (species_data module for CEA, spec dict for TABULAR).
    composition : dict, optional
        Composition specification. For CEA, elemental dict. Not used for TABULAR.

    Returns
    -------
    object
        A JAX thermo object (JaxCEAThermo or JaxTabularThermo).
    """
    if thermo_method == 'TABULAR':
        from .tabular.jax_tabular import JaxTabularThermo
        from pycycle.constants import AIR_JETA_TAB_SPEC
        spec = thermo_data if thermo_data is not None else AIR_JETA_TAB_SPEC
        return JaxTabularThermo(spec)
    elif thermo_method == 'CEA':
        from .cea.jax_cea import JaxCEAThermo
        return JaxCEAThermo(thermo_data=thermo_data, composition=composition)
    else:
        raise ValueError(f"Unsupported thermo_method: {thermo_method}")


def get_mixed_output_composition(thermo_method, thermo_data, inflow_composition, reactant):
    """Return the output composition dict for a flow that mixes in a reactant.

    This is a standalone factory function that does not require a thermo instance.
    Use this in pyc_setup_output_ports (before the thermo object is created).

    Parameters
    ----------
    thermo_method : str
        'CEA' or 'TABULAR'
    thermo_data : object
        Thermodynamic data.
    inflow_composition : dict
        Inflow composition specification.
    reactant : str or tuple
        Reactant name(s) to mix in.

    Returns
    -------
    dict
        Output composition dict suitable for port setup.
    """
    if thermo_method == 'CEA':
        from .cea.thermo_add import ThermoAdd
        mixer = ThermoAdd(inflow_composition=inflow_composition,
                          mix_mode='reactant',
                          mix_composition=reactant,
                          thermo_data=thermo_data)
        return mixer.mixed_elements
    elif thermo_method == 'TABULAR':
        # For tabular, composition structure doesn't change
        return inflow_composition
    else:
        raise ValueError(f"Unsupported thermo_method: {thermo_method}")


def get_composition_array(thermo_method, thermo_data, composition_dict):
    """Convert a composition dict to the array form used by a thermo method.

    This is a standalone factory function that does not require a thermo instance.

    Parameters
    ----------
    thermo_method : str
        'CEA' or 'TABULAR'
    thermo_data : object
        Thermodynamic data.
    composition_dict : dict
        Composition specification.

    Returns
    -------
    ndarray
        Composition array.
    """
    import numpy as np
    if thermo_method == 'CEA':
        from pycycle.thermo.cea.species_data import Properties
        props = Properties(thermo_data, init_elements=composition_dict)
        return props.b0.copy()
    elif thermo_method == 'TABULAR':
        return np.array(list(composition_dict.values()))
    else:
        raise ValueError(f"Unsupported thermo_method: {thermo_method}")


def create_composition_mixer(thermo_method, thermo_data, inflow_composition, reactant):
    """Create a composition mixer for combining a base flow with a reactant.

    This is a standalone factory function that does not require a thermo instance.

    The returned mixer object has:
    - output_composition: dict for port setup
    - base_b0: numpy array of base composition
    - comp_size: size of the output composition array
    - mix_jax(b0_in, W_in, W_reactant): JAX-traceable mixing function

    Parameters
    ----------
    thermo_method : str
        'CEA' or 'TABULAR'
    thermo_data : object
        Thermodynamic data.
    inflow_composition : dict
        Inflow composition specification.
    reactant : str or tuple
        Reactant name(s) to mix in.

    Returns
    -------
    object
        Mixer object with the interface described above.
    """
    if thermo_method == 'CEA':
        from .cea.jax_cea import CEACompositionMixer
        return CEACompositionMixer(thermo_data, inflow_composition, reactant)
    elif thermo_method == 'TABULAR':
        from .tabular.jax_tabular import TabularCompositionMixer
        return TabularCompositionMixer(inflow_composition)
    else:
        raise ValueError(f"Unsupported thermo_method: {thermo_method}")
