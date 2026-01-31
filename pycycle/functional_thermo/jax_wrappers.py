"""
Named index classes for accessing JAX thermo property arrays.

This module provides index classes for accessing properties returned by
JaxTabularThermo methods, enabling readable code like `props[TotalPropsIdx.h]`.
"""


class TotalPropsIdx:
    """Named indices for total property arrays returned by JaxTabularThermo.props_TP."""
    h = 0
    S = 1
    gamma = 2
    Cp = 3
    Cv = 4
    rho = 5
    R = 6

    @classmethod
    def count(cls):
        return 7


class StaticPropsIdx:
    """Named indices for static property arrays returned by JaxTabularThermo.static_from_*."""
    Ts = 0
    Ps = 1
    hs = 2
    rhos = 3
    MN = 4
    V = 5
    Vsonic = 6
    area = 7
    gamma = 8
    Cp = 9
    Cv = 10
    S = 11
    R = 12

    @classmethod
    def count(cls):
        return 13
