"""
JaxElement - Base class for pyCycle elements using JAX automatic differentiation.

This provides a clean separation between physics (compute_physics) and OpenMDAO interface.
Derivatives are computed automatically using JAX's autodiff.
"""

import numpy as np
import jax
import jax.numpy as jnp

import openmdao.api as om

from pycycle.constants import ALLOWED_THERMOS, AIR_JETA_TAB_SPEC

# =============================================================================
# Detailed timing stats for compute_partials breakdown
# =============================================================================
import time

_jax_element_timing_stats = {
    # JaxElement compute() breakdown
    'compute_calls': 0,
    'compute_time': 0.0,
    # JaxElement compute_partials breakdown
    'jacobian_compute_calls': 0,
    'jacobian_compute_time': 0.0,
    'jacobian_assign_calls': 0,
    'jacobian_assign_time': 0.0,
    'jvp_calls': 0,
    'jvp_time': 0.0,
    'jit_cache_hits': 0,
    'jit_cache_misses': 0,
}

def reset_timing_stats():
    """Reset all JaxElement timing statistics."""
    for key in _jax_element_timing_stats:
        _jax_element_timing_stats[key] = 0.0 if 'time' in key else 0


def get_timing_stats():
    """Return a copy of the timing statistics dictionary."""
    return dict(_jax_element_timing_stats)


def clear_jit_cache():
    """Clear the class-level JIT function cache.

    This forces recompilation on the next compute_partials call.
    Useful for testing or when thermo objects have changed.
    """
    JaxElement._jit_jvp_cache.clear()

def print_timing_stats():
    """Print detailed JaxElement timing statistics."""
    stats = _jax_element_timing_stats
    print("\n=== JaxElement Timing Statistics ===")

    # Forward compute stats
    print(f"  Forward compute (compute_physics):")
    print(f"    calls: {stats['compute_calls']}")
    print(f"    total time: {stats['compute_time']*1000:.3f} ms")
    if stats['compute_calls'] > 0:
        print(f"    avg time: {stats['compute_time']*1000/stats['compute_calls']:.3f} ms")

    # Jacobian stats
    print(f"  Jacobian computation (JAX JVP):")
    print(f"    calls: {stats['jacobian_compute_calls']}")
    print(f"    total time: {stats['jacobian_compute_time']*1000:.3f} ms")
    if stats['jacobian_compute_calls'] > 0:
        print(f"    avg time: {stats['jacobian_compute_time']*1000/stats['jacobian_compute_calls']:.3f} ms")
    print(f"    individual jvp calls: {stats['jvp_calls']}")
    print(f"    individual jvp total time: {stats['jvp_time']*1000:.3f} ms")
    if stats['jvp_calls'] > 0:
        print(f"    individual jvp avg time: {stats['jvp_time']*1000/stats['jvp_calls']:.3f} ms")

    print(f"  Jacobian assignment to partials:")
    print(f"    calls: {stats['jacobian_assign_calls']}")
    print(f"    total time: {stats['jacobian_assign_time']*1000:.3f} ms")
    if stats['jacobian_assign_calls'] > 0:
        print(f"    avg time: {stats['jacobian_assign_time']*1000/stats['jacobian_assign_calls']:.3f} ms")

    # JIT cache stats
    print(f"  JIT function cache:")
    print(f"    cache hits: {stats['jit_cache_hits']}")
    print(f"    cache misses (compilations): {stats['jit_cache_misses']}")
    total_lookups = stats['jit_cache_hits'] + stats['jit_cache_misses']
    if total_lookups > 0:
        print(f"    hit rate: {100*stats['jit_cache_hits']/total_lookups:.1f}%")

    # Summary
    compute_time = stats['compute_time']
    partials_time = stats['jacobian_compute_time'] + stats['jacobian_assign_time']
    total_time = compute_time + partials_time
    print(f"  --- Summary ---")
    print(f"    Forward compute time: {compute_time*1000:.3f} ms")
    print(f"    Partials time: {partials_time*1000:.3f} ms")
    print(f"    Total tracked time: {total_time*1000:.3f} ms")
    if total_time > 0:
        print(f"    forward/partials ratio: {compute_time/partials_time:.2f}x" if partials_time > 0 else "    (no partials computed)")
    print("=====================================\n")


# =============================================================================
# Flow Property Definitions (shared by add_flow_input/add_flow_output)
# =============================================================================
# Format: (property_name, default_value, units)
_TOTAL_PROPS = [
    ('h', 124.0, 'Btu/lbm'), ('T', 518., 'degR'), ('P', 1., 'lbf/inch**2'),
    ('rho', 1.0, 'lbm/ft**3'), ('gamma', 1.4, None), ('Cp', 1.0, 'Btu/(lbm*degR)'),
    ('Cv', 1.0, 'Btu/(lbm*degR)'), ('S', 1.0, 'Btu/(lbm*degR)'), ('R', 1.0, 'Btu/(lbm*degR)'),
]

_STATIC_PROPS = [
    ('h', 124.0, 'Btu/lbm'), ('T', 518., 'degR'), ('P', 1.0, 'lbf/inch**2'),
    ('rho', 1.0, 'lbm/ft**3'), ('gamma', 1.4, None), ('Cp', 1.0, 'Btu/(lbm*degR)'),
    ('Cv', 1.0, 'Btu/(lbm*degR)'), ('S', 0.0, 'Btu/(lbm*degR)'), ('R', 1.0, 'Btu/(lbm*degR)'),
    ('V', 1.0, 'ft/s'), ('Vsonic', 1.0, 'ft/s'), ('MN', 0.5, None),
    ('area', 1.0, 'inch**2'), ('Wc', 1.0, 'lbm/s'),
]

# Primal output mappings: (om_property_suffix, primal_name)
_TOTAL_PRIMAL_OUTPUTS = [
    ('P', 'Pt_out'), ('T', 'Tt_out'), ('h', 'ht_out'),
    ('S', 'S_out'), ('gamma', 'gamma_out'), ('Cp', 'Cp_out'),
    ('Cv', 'Cv_out'), ('rho', 'rho_out'), ('R', 'R_out'),
]

_STATIC_PRIMAL_OUTPUTS = [
    ('h', 'hs_out'), ('T', 'Ts_out'), ('P', 'Ps_out'),
    ('rho', 'rhos_out'), ('gamma', 'gammas_out'), ('Cp', 'Cps_out'),
    ('Cv', 'Cvs_out'), ('S', 'Ss_out'), ('R', 'Rs_out'),
    ('V', 'V_out'), ('Vsonic', 'Vsonic_out'), ('MN', 'MN_out'),
    ('area', 'area_out'), ('Wc', 'Wc_out'),
]


class JaxElement(om.ExplicitComponent):
    """
    Base class for pyCycle elements using JAX for automatic differentiation.

    Subclasses implement `compute_physics` which is a pure JAX-traceable function.
    The base class automatically handles all partial derivative computation.

    JaxThermo objects are shared across instances with the same configuration
    (thermo_method, thermo_data) to avoid redundant JAX tracing. The linearization
    cache is kept per-instance to allow different operating points.

    JIT-compiled JVP functions are cached at the class level to avoid redundant
    JAX tracing/compilation across instances with the same configuration.

    Example
    -------
    class MyDuct(JaxElement):

        def setup(self):
            self.add_input('Fl_I:tot:P', val=1.0, units='psi')
            self.add_input('dPqP', val=0.0)
            self.add_output('Fl_O:tot:P', val=1.0, units='psi')

        def compute_physics(self, Pt_in, ht_in, dPqP):
            '''Pure JAX computation.'''
            Pt_out = Pt_in * (1.0 - dPqP)
            return (Pt_out,)  # Tuple of outputs
    """

    # Class-level cache for shared JaxThermo objects
    # Key: (thermo_method, id(thermo_data))
    # Value: JaxThermo instance
    _shared_thermos = {}

    # Class-level cache for JIT-compiled JVP functions
    # Key: configuration tuple from _get_jit_config_key()
    # Value: JIT-compiled jvp_wrapper function
    _jit_jvp_cache = {}

    def initialize(self):

        self._jax_thermo = None
        self._primal_input_names = []  # Maps primal arg name -> OpenMDAO input name
        self._primal_output_names = []  # Maps primal return index -> OpenMDAO output name
        self._cached_args = None
        self._jit_jacfwd_fn = None  # Cached JIT-compiled full Jacobian function

        # For compatibility with Cycle's flow graph
        self.Fl_I_data = {}
        self.Fl_O_data = {}
        self.options.declare('design', default=True,
                             desc='Switch between on-design and off-design calculation.')
        self.options.declare('thermo_data', default=None, recordable=False,
                             desc='Thermodynamic data specific to this element')
        self.options.declare('thermo_method', default='CEA', values=ALLOWED_THERMOS,
                             desc='Method for computing thermodynamic properties')

    @property
    def jax_thermo(self):
        """
        Lazy-initialized JAX thermo object, accessible during compute_physics.

        JaxThermo objects are shared across instances with the same configuration
        to avoid redundant JAX tracing. The linearization cache is kept per-instance.
        """
        if self._jax_thermo is None:
            self._jax_thermo = self._get_shared_jax_thermo()
        return self._jax_thermo

    def _get_shared_jax_thermo(self):
        """
        Get or create a shared JaxTabularThermo for this configuration.

        Uses class-level cache keyed by (thermo_method, thermo_data).
        """
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        key = (thermo_method, id(thermo_data))

        if key not in JaxElement._shared_thermos:
            # Create new JaxTabularThermo and cache at class level
            spec = self._get_thermo_spec()
            from pycycle.functional_thermo.jax_tabular import JaxTabularThermo
            JaxElement._shared_thermos[key] = JaxTabularThermo(spec)

        return JaxElement._shared_thermos[key]

    def _get_thermo_spec(self):
        """
        Get the tabular thermo spec dict based on options.

        Returns
        -------
        dict
            Tabular thermo specification dictionary.

        Raises
        ------
        ValueError
            If thermo_method is not 'TABULAR'.
        """
        method = self.options['thermo_method']
        if method != 'TABULAR':
            raise ValueError(f"JaxElement only supports TABULAR thermo_method, got {method}")

        thermo_data = self.options['thermo_data']
        if thermo_data is None:
            return AIR_JETA_TAB_SPEC
        return thermo_data

    def pyc_setup_output_ports(self):
        """Override in subclass to set up output port data for Cycle's flow graph."""
        pass

    def copy_flow(self, src_port, output_port):
        """Copy flow data from source port to output port."""
        if src_port in self.Fl_I_data:
            self.Fl_O_data[output_port] = self.Fl_I_data[src_port]

    def init_output_flow(self, port_name, port_data):
        """Initialize an output port with specific port data."""
        self.Fl_O_data[port_name] = port_data

    # =========================================================================
    # Primal Input/Output Registration
    # =========================================================================

    def add_primal_input(self, om_name, primal_name):
        """
        Register an OpenMDAO input as an argument to compute_physics.

        Parameters
        ----------
        om_name : str
            The OpenMDAO input variable name (e.g., 'Fl_I:tot:P')
        primal_name : str
            The argument name used in compute_physics (e.g., 'Pt_in')
        """
        self._primal_input_names.append((primal_name, om_name))

    def add_primal_output(self, om_name, primal_name):
        """
        Register an OpenMDAO output as a return value from compute_physics.

        Parameters
        ----------
        om_name : str
            The OpenMDAO output variable name (e.g., 'Fl_O:tot:P')
        primal_name : str
            Descriptive name for this output position (e.g., 'Pt_out')
        """
        self._primal_output_names.append((primal_name, om_name))

    def setup_partials(self):
        """Declare all partials between primal inputs and outputs."""
        # Use wildcard declaration - OpenMDAO will handle sparsity detection
        self.declare_partials('*', '*')

    def add_flow_total_primal_outputs(self, fl_name='Fl_O'):
        """
        Register standard total flow properties as primal outputs.

        Registers: P, T, h, S, gamma, Cp, Cv, rho, R (in that order).
        Does NOT include W (mass flow) - register that separately if needed.
        """
        for prop, primal_name in _TOTAL_PRIMAL_OUTPUTS:
            self.add_primal_output(f'{fl_name}:tot:{prop}', primal_name)

    def add_flow_static_primal_outputs(self, fl_name='Fl_O'):
        """
        Register standard static flow properties as primal outputs.

        Registers: h, T, P, rho, gamma, Cp, Cv, S, R, V, Vsonic, MN, area, Wc
        """
        for prop, primal_name in _STATIC_PRIMAL_OUTPUTS:
            self.add_primal_output(f'{fl_name}:stat:{prop}', primal_name)

    def compute_physics(self, *args):
        """
        Pure JAX-traceable computation. Override in subclass.

        Parameters
        ----------
        *args : floats
            Input values in order of add_primal_input calls

        Returns
        -------
        tuple
            Output values in order of add_primal_output calls
        """
        raise NotImplementedError("Subclass must implement compute_physics")

    def _get_jit_config_key(self):
        """
        Return a hashable key for JIT function caching.

        This key identifies the configuration that affects compute_physics behavior.
        Instances with the same key can share the same JIT-compiled JVP function,
        avoiding redundant JAX tracing.

        Subclasses should override to include any options that affect compute_physics.

        Returns
        -------
        tuple
            Hashable configuration key
        """
        return (
            type(self).__name__,
            self.options.get('design', True),
            id(self._jax_thermo) if self._jax_thermo is not None else None,
        )

    def _is_array_primal(self, om_name):
        """
        Check if an OpenMDAO variable should be treated as an array primal.

        Override in subclass to mark specific inputs as array-valued (not scalar).
        By default, 'composition' variables are treated as arrays.

        Parameters
        ----------
        om_name : str
            OpenMDAO variable name

        Returns
        -------
        bool
            True if this variable should be treated as an array in compute_physics
        """
        return 'composition' in om_name

    def compute(self, inputs, outputs):
        """Extract inputs, call compute_physics, assign outputs."""
        # Extract inputs in registered order, handling array primals
        args = []
        for primal_name, om_name in self._primal_input_names:
            if self._is_array_primal(om_name):
                args.append(jnp.array(inputs[om_name]))
            else:
                args.append(float(inputs[om_name][0]))

        # Call pure computation with timing
        t_start = time.perf_counter()
        result = self.compute_physics(*args)
        _jax_element_timing_stats['compute_calls'] += 1
        _jax_element_timing_stats['compute_time'] += (time.perf_counter() - t_start)

        # Validate output count matches registration
        expected = len(self._primal_output_names)
        actual = len(result)
        if actual != expected:
            raise ValueError(
                f"{type(self).__name__}.compute_physics returned {actual} outputs, "
                f"but {expected} were registered via add_primal_output. "
                f"Ensure compute_physics return order matches add_primal_output call order."
            )

        # Assign outputs in registered order
        for i, (primal_name, om_name) in enumerate(self._primal_output_names):
            outputs[om_name] = float(result[i])

        # Pass through composition and FAR (not part of JAX computation)
        if hasattr(self, '_passthrough_vars'):
            for src, dst in self._passthrough_vars:
                outputs[dst] = inputs[src]

        # Cache args for partials
        self._cached_args = args

    def compute_partials(self, inputs, partials):
        """Compute partial derivatives using JAX autodiff."""
        args = self._cached_args

        # Compute Jacobian (subclass can override compute_jacobian for efficiency)
        t_start = time.perf_counter()
        if hasattr(self, 'compute_jacobian'):
            jacs = self.compute_jacobian(args)
        else:
            jacs = self._compute_jacobian_fwd(args)
        _jax_element_timing_stats['jacobian_compute_calls'] += 1
        _jax_element_timing_stats['jacobian_compute_time'] += (time.perf_counter() - t_start)

        # Assign to partials dict, handling array primals
        t_start = time.perf_counter()
        for j, (_, in_name) in enumerate(self._primal_input_names):
            for i, (_, out_name) in enumerate(self._primal_output_names):
                deriv = jacs[j][i]
                if self._is_array_primal(in_name):
                    if np.isscalar(deriv):
                        partials[out_name, in_name] = np.array([[deriv]])
                    else:
                        partials[out_name, in_name] = np.array(deriv).reshape(1, -1)
                else:
                    partials[out_name, in_name] = deriv

        # Handle passthrough variable partials (identity derivatives)
        if hasattr(self, '_passthrough_vars'):
            for src, dst in self._passthrough_vars:
                src_val = inputs[src]
                if np.isscalar(src_val) or len(src_val) == 1:
                    partials[dst, src] = 1.0
                else:
                    partials[dst, src] = np.eye(len(src_val))
        _jax_element_timing_stats['jacobian_assign_calls'] += 1
        _jax_element_timing_stats['jacobian_assign_time'] += (time.perf_counter() - t_start)

    def _compute_jacobian_fwd(self, args):
        """Compute Jacobian using forward-mode with cached JIT-compiled JVP calls.

        JIT-compiled functions are cached at the class level, keyed by configuration.
        This allows instances with the same config to share the compiled function,
        avoiding redundant JAX tracing which is expensive.
        """
        n_inputs = len(args)
        n_outputs = len(self._primal_output_names)
        jacs = [np.zeros(n_outputs) for _ in range(n_inputs)]
        args_tuple = tuple(args)

        # Get config key for class-level cache lookup
        config_key = self._get_jit_config_key()

        # Get or create JIT-compiled JVP function for this configuration
        if config_key not in JaxElement._jit_jvp_cache:
            # First instance with this config creates the JIT function
            # The function captures self.compute_physics which uses the shared jax_thermo
            _jax_element_timing_stats['jit_cache_misses'] += 1
            compute_physics = self.compute_physics
            def jvp_wrapper(args_tuple, tangents_tuple):
                _, jvp_out = jax.jvp(compute_physics, args_tuple, tangents_tuple)
                return jnp.array(jvp_out)
            # JIT compile and cache at class level
            JaxElement._jit_jvp_cache[config_key] = jax.jit(jvp_wrapper)
        else:
            _jax_element_timing_stats['jit_cache_hits'] += 1

        jit_fn = JaxElement._jit_jvp_cache[config_key]

        for j in range(n_inputs):
            # Create tangent with same structure as args (handle arrays)
            tangents = []
            for i, arg in enumerate(args):
                if i == j:
                    if hasattr(arg, 'shape') and len(arg.shape) > 0:
                        tangents.append(jnp.ones_like(arg))
                    else:
                        tangents.append(1.0)
                else:
                    if hasattr(arg, 'shape') and len(arg.shape) > 0:
                        tangents.append(jnp.zeros_like(arg))
                    else:
                        tangents.append(0.0)

            t_jvp = time.perf_counter()
            jvp_out = jit_fn(args_tuple, tuple(tangents))
            _jax_element_timing_stats['jvp_calls'] += 1
            _jax_element_timing_stats['jvp_time'] += (time.perf_counter() - t_jvp)

            jacs[j][:] = np.array(jvp_out)

        return jacs

    # =========================================================================
    # Flow Input/Output Helpers (for pyCycle flow connections)
    # =========================================================================

    def add_flow_input(self, fl_name='Fl_I'):
        """
        Add a complete set of flow input variables for a flow port.

        These are needed for pyCycle's pyc_connect_flow to work.
        Use add_primal_input separately to register which ones are used in compute_physics.
        """
        # Total properties
        for prop, val, units in _TOTAL_PROPS:
            self.add_input(f'{fl_name}:tot:{prop}', val=val, units=units)
        self.add_input(f'{fl_name}:tot:composition', shape_by_conn=True)

        # Static properties
        for prop, val, units in _STATIC_PROPS:
            self.add_input(f'{fl_name}:stat:{prop}', val=val, units=units)
        self.add_input(f'{fl_name}:stat:composition', shape_by_conn=True)
        self.add_input(f'{fl_name}:stat:W', val=1.0, units='lbm/s')
        self.add_input(f'{fl_name}:FAR', val=0.0)

    def add_flow_output(self, fl_name='Fl_O', statics=True, fl_src='Fl_I'):
        """
        Add a complete set of flow output variables for a flow port.

        Parameters
        ----------
        fl_name : str
            Name prefix for the flow port (e.g., 'Fl_O')
        statics : bool
            If True, include static property outputs.
        fl_src : str
            Source flow port for composition passthrough (e.g., 'Fl_I')
        """
        # Total properties
        for prop, val, units in _TOTAL_PROPS:
            kwargs = {'val': val, 'units': units}
            if prop == 'P':
                kwargs['lower'] = 1e-4
            self.add_output(f'{fl_name}:tot:{prop}', **kwargs)
        self.add_output(f'{fl_name}:tot:composition', shape_by_conn=True,
                        copy_shape=f'{fl_src}:tot:composition')

        # Static properties
        if statics:
            for prop, val, units in _STATIC_PROPS:
                self.add_output(f'{fl_name}:stat:{prop}', val=val, units=units)

        # Always output mass flow and FAR
        self.add_output(f'{fl_name}:stat:W', val=1.0, units='lbm/s')
        self.add_output(f'{fl_name}:FAR', val=0.0)

        # Track passthrough variables for compute() (not part of JAX computation)
        if not hasattr(self, '_passthrough_vars'):
            self._passthrough_vars = []
        self._passthrough_vars.append((f'{fl_src}:tot:composition', f'{fl_name}:tot:composition'))
        self._passthrough_vars.append((f'{fl_src}:FAR', f'{fl_name}:FAR'))

