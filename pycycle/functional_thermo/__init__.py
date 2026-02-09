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
