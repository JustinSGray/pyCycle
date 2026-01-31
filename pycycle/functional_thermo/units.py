"""
Unit conversion utilities for functional thermodynamic interfaces.

This module provides unit conversion support using OpenMDAO's unit library.
The thermo classes internally work in SI units, but can accept and return
values in different unit systems.
"""

from openmdao.utils.units import unit_conversion


# Standard unit systems
SI_UNITS = {
    'T': 'K',
    'P': 'Pa',
    'h': 'J/kg',
    'S': 'J/(kg*K)',
    'Cp': 'J/(kg*K)',
    'Cv': 'J/(kg*K)',
    'R': 'J/(kg*K)',
    'rho': 'kg/m**3',
    'V': 'm/s',
    'Vsonic': 'm/s',
    'area': 'm**2',
    'W': 'kg/s',
}

# pyCycle English units (used by OpenMDAO components)
ENGLISH_UNITS = {
    'T': 'degR',
    'P': 'lbf/inch**2',
    'h': 'Btu/lbm',
    'S': 'Btu/(lbm*degR)',
    'Cp': 'Btu/(lbm*degR)',
    'Cv': 'Btu/(lbm*degR)',
    'R': 'Btu/(lbm*degR)',
    'rho': 'lbm/ft**3',
    'V': 'ft/s',
    'Vsonic': 'ft/s',
    'area': 'inch**2',
    'W': 'lbm/s',
}


class UnitConverter:
    """
    Handles unit conversions between input units and SI units for thermo calculations.

    The thermo classes work internally in SI units. This class pre-computes
    conversion factors for efficient conversion of inputs and outputs.

    Parameters
    ----------
    input_units : str or dict, optional
        Unit system for inputs/outputs. Can be:
        - 'SI': Use SI units (no conversion needed)
        - 'English': Use pyCycle English units
        - dict: Custom unit mapping {property: unit_string}
        Default is 'SI'.
    """

    def __init__(self, input_units='SI'):
        self.input_units = input_units

        # Determine the unit mapping
        if input_units == 'SI':
            self._units = SI_UNITS
            self._is_si = True
        elif input_units == 'English':
            self._units = ENGLISH_UNITS
            self._is_si = False
        elif isinstance(input_units, dict):
            # Custom units - merge with SI defaults
            self._units = SI_UNITS.copy()
            self._units.update(input_units)
            self._is_si = False
        else:
            raise ValueError(f"Unknown unit system: {input_units}. "
                           f"Use 'SI', 'English', or a dict of units.")

        # Pre-compute conversion factors (input_units -> SI and SI -> input_units)
        self._to_si = {}
        self._from_si = {}

        for prop, input_unit in self._units.items():
            si_unit = SI_UNITS[prop]
            if input_unit == si_unit:
                # No conversion needed
                self._to_si[prop] = (1.0, 0.0)
                self._from_si[prop] = (1.0, 0.0)
            else:
                # Get conversion factors using OpenMDAO
                self._to_si[prop] = unit_conversion(input_unit, si_unit)
                self._from_si[prop] = unit_conversion(si_unit, input_unit)

    @property
    def is_si(self):
        """Return True if using SI units (no conversion needed)."""
        return self._is_si

    def to_si(self, prop, value):
        """
        Convert a value from input units to SI.

        Parameters
        ----------
        prop : str
            Property name ('T', 'P', 'h', etc.)
        value : float or array
            Value in input units

        Returns
        -------
        float or array
            Value in SI units
        """
        if self._is_si:
            return value
        factor, offset = self._to_si[prop]
        return (value + offset) * factor

    def from_si(self, prop, value):
        """
        Convert a value from SI to input units.

        Parameters
        ----------
        prop : str
            Property name ('T', 'P', 'h', etc.)
        value : float or array
            Value in SI units

        Returns
        -------
        float or array
            Value in input units
        """
        if self._is_si:
            return value
        factor, offset = self._from_si[prop]
        return (value + offset) * factor

    def get_factor_to_si(self, prop):
        """
        Get the multiplicative factor to convert from input units to SI.

        Note: For temperature, this only returns the scale factor, not the offset.
        Use to_si() for full conversion including offsets.

        Parameters
        ----------
        prop : str
            Property name

        Returns
        -------
        float
            Conversion factor (multiply input by this to get SI)
        """
        return self._to_si[prop][0]

    def get_factor_from_si(self, prop):
        """
        Get the multiplicative factor to convert from SI to input units.

        Note: For temperature, this only returns the scale factor, not the offset.
        Use from_si() for full conversion including offsets.

        Parameters
        ----------
        prop : str
            Property name

        Returns
        -------
        float
            Conversion factor (multiply SI by this to get input units)
        """
        return self._from_si[prop][0]

    def input_unit(self, prop):
        """Get the input unit string for a property."""
        return self._units[prop]

    def si_unit(self, prop):
        """Get the SI unit string for a property."""
        return SI_UNITS[prop]
