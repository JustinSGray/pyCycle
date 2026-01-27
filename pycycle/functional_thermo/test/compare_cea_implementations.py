"""
Comparison of OpenMDAO CEA implementation vs functional CEAThermo.

This script runs both implementations at identical conditions and compares results.
"""

import numpy as np
import openmdao.api as om

from pycycle.thermo.thermo import Thermo
from pycycle.thermo.cea import species_data
from pycycle import constants

from pycycle.functional_thermo import CEAThermo


def run_openmdao_cea(T, P, composition=None, spec=None):
    """Run OpenMDAO CEA implementation and return properties."""
    if composition is None:
        composition = constants.CEA_AIR_COMPOSITION
    if spec is None:
        spec = species_data.janaf

    p = om.Problem()
    p.model.add_subsystem('thermo', Thermo(mode='total_TP',
                                            method='CEA',
                                            thermo_kwargs={'composition': composition,
                                                          'spec': spec}),
                          promotes=['*'])
    p.setup(check=False)
    p.set_solver_print(level=-1)
    p.final_setup()

    # Convert units: T in K -> degK (same), P in Pa -> bar
    p.set_val('T', T, units='degK')
    p.set_val('P', P / 100000.0, units='bar')

    p.run_model()

    # Extract results (convert to SI units)
    h = p.get_val('flow:h', units='J/kg')[0]
    S = p.get_val('flow:S', units='J/(kg*degK)')[0]
    gamma = p['flow:gamma'][0]
    Cp = p.get_val('flow:Cp', units='J/(kg*degK)')[0]
    Cv = p.get_val('flow:Cv', units='J/(kg*degK)')[0]
    rho = p.get_val('flow:rho', units='kg/m**3')[0]
    R_gas = p.get_val('flow:R', units='J/(kg*degK)')[0]

    return {
        'h': h,
        'S': S,
        'gamma': gamma,
        'Cp': Cp,
        'Cv': Cv,
        'rho': rho,
        'R': R_gas
    }


def run_functional_cea(T, P, composition=None, thermo_data=None):
    """Run functional CEAThermo implementation and return properties."""
    if composition is None:
        composition = constants.CEA_AIR_COMPOSITION
    if thermo_data is None:
        thermo_data = species_data.janaf

    thermo = CEAThermo(composition=composition, thermo_data=thermo_data)
    props = thermo.props_TP(T, P)

    return {
        'h': props.h,
        'S': props.S,
        'gamma': props.gamma,
        'Cp': props.Cp,
        'Cv': props.Cv,
        'rho': props.rho,
        'R': props.R
    }


def compare_results(omdao_results, func_results, label=""):
    """Compare results and print differences."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'Property':<12} {'OpenMDAO':<18} {'Functional':<18} {'Rel Error %':<12} {'Status':<8}")
    print("-" * 70)

    all_pass = True
    for prop in ['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R']:
        omdao_val = omdao_results[prop]
        func_val = func_results[prop]

        if abs(omdao_val) > 1e-10:
            rel_err = abs((func_val - omdao_val) / omdao_val) * 100
        else:
            rel_err = abs(func_val - omdao_val) * 100

        # Different tolerances for different properties
        if prop == 'gamma':
            tol = 1.0  # 1% for gamma
        elif prop in ['h', 'S']:
            tol = 2.0  # 2% for h and S
        else:
            tol = 5.0  # 5% for others

        status = "PASS" if rel_err < tol else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{prop:<12} {omdao_val:<18.6g} {func_val:<18.6g} {rel_err:<12.4f} {status:<8}")

    return all_pass


def run_comparison_suite():
    """Run full comparison suite."""
    print("\n" + "=" * 70)
    print("  CEA Implementation Comparison: OpenMDAO vs Functional")
    print("=" * 70)

    # Test conditions
    test_cases = [
        # (T, P, description)
        (500.0, 101325.0, "Low temperature (500K), 1 atm"),
        (1000.0, 101325.0, "Mid temperature (1000K), 1 atm"),
        (1500.0, 101325.0, "High temperature (1500K), 1 atm"),
        (2000.0, 101325.0, "Very high temperature (2000K), 1 atm"),
        (1500.0, 300000.0, "High temperature (1500K), 3 atm"),
        (1500.0, 50000.0, "High temperature (1500K), 0.5 atm"),
        (800.0, 200000.0, "Mid temperature (800K), 2 atm"),
    ]

    results_summary = []

    for T, P, desc in test_cases:
        try:
            omdao = run_openmdao_cea(T, P)
            func = run_functional_cea(T, P)
            passed = compare_results(omdao, func, f"{desc} (T={T}K, P={P/1000:.1f}kPa)")
            results_summary.append((desc, T, P, passed))
        except Exception as e:
            print(f"\nERROR at T={T}K, P={P}Pa: {e}")
            results_summary.append((desc, T, P, False))

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    total = len(results_summary)
    passed = sum(1 for r in results_summary if r[3])
    print(f"\nTotal test cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if passed == total:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print("\n*** SOME TESTS FAILED ***")
        print("\nFailed cases:")
        for desc, T, P, status in results_summary:
            if not status:
                print(f"  - {desc}")

    return passed == total


def run_static_comparison():
    """Compare static property calculations."""
    print("\n" + "=" * 70)
    print("  Static Property Comparison")
    print("=" * 70)

    Tt = 1500.0  # K
    Pt = 300000.0  # Pa
    W = 10.0  # kg/s
    MN = 0.5

    # OpenMDAO static calculation
    p = om.Problem()
    total_TP = Thermo(mode='total_TP',
                      method='CEA',
                      thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION,
                                    'spec': species_data.janaf})
    p.model.add_subsystem('set_total_TP', total_TP)

    static_MN = Thermo(mode='static_MN',
                       method='CEA',
                       thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION,
                                     'spec': species_data.janaf})
    p.model.add_subsystem('set_static_MN', static_MN)

    p.model.connect('set_total_TP.flow:S', 'set_static_MN.S')
    p.model.connect('set_total_TP.flow:h', 'set_static_MN.ht')
    p.model.connect('set_total_TP.flow:gamma', 'set_static_MN.guess:gamt')
    p.model.connect('set_total_TP.flow:P', 'set_static_MN.guess:Pt')

    p.set_solver_print(level=-1)
    p.setup(check=False)

    p.set_val('set_total_TP.T', Tt, units='degK')
    p.set_val('set_total_TP.P', Pt / 100000.0, units='bar')
    p.set_val('set_static_MN.MN', MN)
    p.set_val('set_static_MN.W', W, units='kg/s')

    p.run_model()

    omdao_static = {
        'Ts': p.get_val('set_static_MN.flow:T', units='degK')[0],
        'Ps': p.get_val('set_static_MN.flow:P', units='Pa')[0],
        'MN': p['set_static_MN.flow:MN'][0],
        'V': p.get_val('set_static_MN.flow:V', units='m/s')[0],
        'area': p.get_val('set_static_MN.flow:area', units='m**2')[0],
        'rhos': p.get_val('set_static_MN.flow:rho', units='kg/m**3')[0],
    }

    # Functional static calculation
    thermo = CEAThermo()
    func_static_props = thermo.static_from_MN(Tt, Pt, MN, W)

    func_static = {
        'Ts': func_static_props.Ts,
        'Ps': func_static_props.Ps,
        'MN': func_static_props.MN,
        'V': func_static_props.V,
        'area': func_static_props.area,
        'rhos': func_static_props.rhos,
    }

    print(f"\nTest conditions: Tt={Tt}K, Pt={Pt/1000:.1f}kPa, MN={MN}, W={W}kg/s")
    print(f"\n{'Property':<12} {'OpenMDAO':<18} {'Functional':<18} {'Rel Error %':<12}")
    print("-" * 60)

    for prop in ['Ts', 'Ps', 'MN', 'V', 'area', 'rhos']:
        omdao_val = omdao_static[prop]
        func_val = func_static[prop]

        if abs(omdao_val) > 1e-10:
            rel_err = abs((func_val - omdao_val) / omdao_val) * 100
        else:
            rel_err = abs(func_val - omdao_val) * 100

        print(f"{prop:<12} {omdao_val:<18.6g} {func_val:<18.6g} {rel_err:<12.4f}")


def run_inverse_comparison():
    """Compare inverse calculations (T_from_hP, T_from_SP)."""
    print("\n" + "=" * 70)
    print("  Inverse Calculation Comparison (T_from_hP, T_from_SP)")
    print("=" * 70)

    test_temps = [500.0, 800.0, 1200.0, 1500.0]
    P = 101325.0

    thermo = CEAThermo()

    print(f"\nT_from_hP round-trip test (P = {P/1000:.1f} kPa):")
    print(f"{'T_original':<15} {'h (J/kg)':<18} {'T_recovered':<15} {'Error (K)':<12}")
    print("-" * 60)

    for T_original in test_temps:
        h = thermo.h(T_original, P)
        T_recovered = thermo.T_from_hP(h, P)
        error = abs(T_recovered - T_original)
        print(f"{T_original:<15.1f} {h:<18.2f} {T_recovered:<15.2f} {error:<12.4f}")

    print(f"\nT_from_SP round-trip test (P = {P/1000:.1f} kPa):")
    print(f"{'T_original':<15} {'S (J/kg/K)':<18} {'T_recovered':<15} {'Error (K)':<12}")
    print("-" * 60)

    for T_original in test_temps:
        S = thermo.S(T_original, P)
        T_recovered = thermo.T_from_SP(S, P)
        error = abs(T_recovered - T_original)
        print(f"{T_original:<15.1f} {S:<18.2f} {T_recovered:<15.2f} {error:<12.4f}")


def run_derivative_comparison():
    """Compare derivative calculations."""
    print("\n" + "=" * 70)
    print("  Derivative Comparison (JVP vs Finite Difference)")
    print("=" * 70)

    T = 1500.0
    P = 300000.0
    eps = 1.0  # Use larger eps for CEA due to equilibrium changes

    thermo = CEAThermo()

    # Get finite difference derivatives
    props_base = thermo.props_TP(T, P)
    props_T_plus = thermo.props_TP(T + eps, P)
    props_T_minus = thermo.props_TP(T - eps, P)

    fd_derivs = {
        'h': (props_T_plus.h - props_T_minus.h) / (2 * eps),
        'S': (props_T_plus.S - props_T_minus.S) / (2 * eps),
        'gamma': (props_T_plus.gamma - props_T_minus.gamma) / (2 * eps),
        'Cp': (props_T_plus.Cp - props_T_minus.Cp) / (2 * eps),
    }

    # Get analytical derivatives
    thermo.linearize(T, P)
    jvp = thermo.jvp(T_dot=1.0, P_dot=0.0)

    print(f"\nTest conditions: T={T}K, P={P/1000:.1f}kPa")
    print(f"Finite difference step: eps={eps}K")
    print(f"\n{'d/dT of':<12} {'FD':<18} {'Analytical':<18} {'Ratio':<12}")
    print("-" * 60)

    for prop in ['h', 'S', 'gamma', 'Cp']:
        fd_val = fd_derivs[prop]
        ana_val = jvp[prop]

        if abs(fd_val) > 1e-10:
            ratio = ana_val / fd_val
        else:
            ratio = float('nan')

        print(f"{prop:<12} {fd_val:<18.6g} {ana_val:<18.6g} {ratio:<12.4f}")


if __name__ == "__main__":
    # Run all comparisons
    run_comparison_suite()
    run_static_comparison()
    run_inverse_comparison()
    run_derivative_comparison()
