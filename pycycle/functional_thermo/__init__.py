"""
Functional interface for thermodynamic property calculations.

This module provides a common interface for computing thermodynamic properties
using different methods (Tabular, CEA). The interface is designed to be
compatible with JAX automatic differentiation.
"""

from .base import ThermoInterface, TotalProps, StaticProps
from .tabular import TabularThermo
from .cea import CEAThermo, ThermoAdd, ThermoAddOutput
