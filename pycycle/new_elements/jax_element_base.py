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
    """Clear all class-level JIT function caches.

    This forces recompilation on the next compute/compute_partials call.
    Useful for testing or when thermo objects have changed.
    """
    JaxElement._jit_jvp_cache.clear()
    JaxElement._jit_compute_cache.clear()

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


class JaxElement(om.ExplicitComponent):
    """
    Base class for pyCycle elements using JAX for automatic differentiation.

    Subclasses implement `compute_physics` which is a pure JAX-traceable function
    that takes a flat input vector and returns a flat output vector.

    The base class automatically handles:
    - Packing OpenMDAO inputs into the input vector
    - Unpacking the output vector to OpenMDAO outputs
    - Computing partial derivatives via JAX autodiff

    Example
    -------
    class MyDuct(JaxElement):

        def setup(self):
            self.add_input('Fl_I:tot:P', val=1.0, units='psi')
            self.add_input('dPqP', val=0.0)
            self.add_output('Fl_O:tot:P', val=1.0, units='psi')

            # Register which inputs/outputs are used in compute_physics
            self.add_primal_input('Fl_I:tot:P')
            self.add_primal_input('dPqP')
            self.add_primal_output('Fl_O:tot:P')

            self.setup_partials()

        def compute_physics(self, inputs):
            '''Pure JAX computation with vector I/O.'''
            Pt_in = self.inp(inputs, 'Fl_I:tot:P')
            dPqP = self.inp(inputs, 'dPqP')

            Pt_out = Pt_in * (1.0 - dPqP)

            return jnp.array([Pt_out])
    """

    # Class-level cache for shared JaxThermo objects
    # Key: (thermo_method, id(thermo_data))
    # Value: JaxThermo instance
    _shared_thermos = {}

    # Class-level cache for JIT-compiled JVP functions
    # Key: configuration tuple from _get_jit_config_key()
    # Value: JIT-compiled jvp_wrapper function
    _jit_jvp_cache = {}

    # Class-level cache for JIT-compiled compute_physics functions
    # Key: configuration tuple from _get_jit_config_key()
    # Value: (jit_fn, thermo_type) where thermo_type is 'CEA' or other
    _jit_compute_cache = {}

    def initialize(self):

        self._jax_thermo = None

        # Track order of add_input/add_output calls
        self._input_order = []   # [om_name, ...] in add_input call order
        self._output_order = []  # [om_name, ...] in add_output call order

        # Primal markers: which inputs/outputs are used in compute_physics
        # size can be: None (scalar), int (fixed array), or 'dynamic' (shape_by_conn)
        self._primal_input_set = {}   # om_name -> size
        self._primal_output_set = {}  # om_name -> size

        # Built by setup_partials(): primal I/O in add_input/add_output order
        # These use declared sizes (may include 'dynamic' placeholders)
        self._primal_inputs = []   # [(om_name, size), ...] filtered and ordered
        self._primal_outputs = []  # [(om_name, size), ...] filtered and ordered

        # Runtime-resolved primal lists (with actual sizes for dynamic vars)
        self._runtime_primal_inputs = None   # [(om_name, size), ...] with resolved sizes
        self._runtime_primal_outputs = None  # [(om_name, size), ...] with resolved sizes

        # Index mappings built by setup_partials() or _rebuild_mappings()
        self._input_idx = {}      # om_name -> index (for scalars)
        self._input_slices = {}   # om_name -> slice (for arrays)
        self._output_idx = {}     # om_name -> index (for scalars)
        self._output_slices = {}  # om_name -> slice (for arrays)
        self._n_primal_inputs = 0
        self._n_primal_outputs = 0

        # Flag to indicate if runtime rebuilding is needed
        self._needs_runtime_rebuild = False

        # Cached input vector for partials computation
        self._cached_input_vec = None

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

    def _get_cea_composition(self):
        """
        Get the CEA composition for this element.

        Checks Fl_I_data first (for flow-through elements like Duct, Compressor),
        then Fl_O_data (for flow-start elements like FlowStart),
        then falls back to the default CEA_AIR_COMPOSITION.

        Returns
        -------
        dict
            Elemental composition dict (e.g., {'N': 0.054, 'O': 0.014, ...})
        """
        if 'Fl_I' in self.Fl_I_data:
            return self.Fl_I_data['Fl_I']
        if 'Fl_O' in self.Fl_O_data:
            return self.Fl_O_data['Fl_O']
        from pycycle.constants import THERMO_DEFAULT_COMPOSITIONS
        return THERMO_DEFAULT_COMPOSITIONS['CEA']

    def _get_shared_jax_thermo(self):
        """
        Get or create a shared JaxThermo for this configuration.

        Uses class-level cache to share JIT-compiled thermo objects across
        elements with the same configuration, avoiding redundant JAX tracing.

        For TABULAR: keyed by (thermo_method, thermo_data_id)
        For CEA: keyed by (thermo_method, thermo_data_id, composition)
            because composition (b0, aij) is baked into JIT-compiled functions.
        """
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']

        if thermo_method == 'TABULAR':
            key = (thermo_method, id(thermo_data))
            if key not in JaxElement._shared_thermos:
                spec = self._get_thermo_spec()
                from pycycle.functional_thermo.tabular.jax_tabular import JaxTabularThermo
                JaxElement._shared_thermos[key] = JaxTabularThermo(spec)

        elif thermo_method == 'CEA':
            composition = self._get_cea_composition()
            # Include the element set in the cache key so that elements with
            # different species sets (e.g., air vs air+fuel) get separate thermos.
            comp_key = tuple(sorted(composition.keys()))
            key = (thermo_method, id(thermo_data), comp_key)
            if key not in JaxElement._shared_thermos:
                from pycycle.functional_thermo.cea.jax_cea import JaxCEAThermo
                JaxElement._shared_thermos[key] = JaxCEAThermo(
                    thermo_data=thermo_data, composition=composition)

        else:
            raise ValueError(f"Unsupported thermo_method: {thermo_method}")

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
            raise ValueError(f"_get_thermo_spec only supports TABULAR thermo_method, got {method}")

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
    # OpenMDAO add_input/add_output overrides to track creation order
    # =========================================================================

    @staticmethod
    def _infer_size(kwargs):
        """
        Infer the primal size for an input/output from its kwargs.

        Returns
        -------
        size : None, int, or 'dynamic'
            None for scalars, int for fixed-size arrays, 'dynamic' for shape_by_conn.
        """
        if kwargs.get('shape_by_conn', False):
            return 'dynamic'
        # Check explicit shape kwarg
        shape = kwargs.get('shape', None)
        if shape is not None:
            total = int(np.prod(shape))
            return total if total > 1 else None
        # Check val kwarg
        val = kwargs.get('val', None)
        if val is not None and np.ndim(val) > 0 and np.size(val) > 1:
            return int(np.size(val))
        return None

    def add_input(self, name, **kwargs):
        """
        Override to track input creation order and auto-register as primal.

        All inputs are automatically registered as primal (included in the
        compute_physics input vector) unless primal=False is passed.
        Inputs with shape_by_conn=True are registered with size='dynamic'.
        Array inputs (val with ndim>0 or explicit shape) are registered with
        their size.
        """
        primal = kwargs.pop('primal', True)
        size = self._infer_size(kwargs)
        super().add_input(name, **kwargs)
        self._input_order.append(name)
        if primal:
            self._primal_input_set[name] = size

    def add_output(self, name, **kwargs):
        """
        Override to track output creation order and auto-register as primal.

        All outputs are automatically registered as primal (included in the
        compute_physics output vector) unless primal=False is passed.
        Array outputs (val with ndim>0 or explicit shape) are registered with
        their size.
        """
        primal = kwargs.pop('primal', True)
        size = self._infer_size(kwargs)
        super().add_output(name, **kwargs)
        self._output_order.append(name)
        if primal:
            self._primal_output_set[name] = size

    # =========================================================================
    # Primal Input/Output Registration
    # =========================================================================

    def add_primal_input(self, om_name, size=None):
        """
        Mark an OpenMDAO input to be included in the compute_physics input vector.

        The position in the input vector is determined by the order of add_input
        calls, not the order of add_primal_input calls.

        Parameters
        ----------
        om_name : str
            The OpenMDAO input variable name (e.g., 'Fl_I:tot:P')
        size : int, optional
            Size of the input if it's an array. None for scalars.
        """
        self._primal_input_set[om_name] = size

    def add_primal_output(self, om_name, size=None):
        """
        Mark an OpenMDAO output to be included in the compute_physics output vector.

        The position in the output vector is determined by the order of add_output
        calls, not the order of add_primal_output calls.

        Parameters
        ----------
        om_name : str
            The OpenMDAO output variable name (e.g., 'Fl_O:tot:P')
        size : int, optional
            Size of the output if it's an array. None for scalars.
        """
        self._primal_output_set[om_name] = size

    def setup_partials(self):
        """
        Build index mappings and declare partials.

        Must be called at the end of subclass setup() after all
        add_primal_input/add_primal_output calls.

        The primal input/output order follows the add_input/add_output call order,
        filtered to only include inputs/outputs marked as primal.
        """
        # Build _primal_inputs in add_input order (filtered to primal only)
        self._primal_inputs = []
        for om_name in self._input_order:
            if om_name in self._primal_input_set:
                size = self._primal_input_set[om_name]
                self._primal_inputs.append((om_name, size))

        # Build _primal_outputs in add_output order (filtered to primal only)
        self._primal_outputs = []
        for om_name in self._output_order:
            if om_name in self._primal_output_set:
                size = self._primal_output_set[om_name]
                self._primal_outputs.append((om_name, size))

        # Check if any primal has dynamic size (shape_by_conn)
        has_dynamic = any(
            size == 'dynamic' for _, size in self._primal_inputs
        ) or any(
            size == 'dynamic' for _, size in self._primal_outputs
        )

        if has_dynamic:
            # Defer mapping construction to runtime when actual sizes are known
            self._needs_runtime_rebuild = True
            # Set placeholder values - will be rebuilt in compute()
            self._n_primal_inputs = 0
            self._n_primal_outputs = 0
        else:
            # Build mappings now with known sizes
            self._build_index_mappings(self._primal_inputs, self._primal_outputs)

        # Declare partials - use wildcard for simplicity
        self.declare_partials('*', '*')

    def _build_index_mappings(self, primal_inputs, primal_outputs):
        """
        Build index mappings for the given primal input/output lists.

        Parameters
        ----------
        primal_inputs : list of (om_name, size) tuples
            Primal inputs with resolved sizes (no 'dynamic' entries)
        primal_outputs : list of (om_name, size) tuples
            Primal outputs with resolved sizes (no 'dynamic' entries)
        """
        # Clear existing mappings
        self._input_idx = {}
        self._input_slices = {}
        self._output_idx = {}
        self._output_slices = {}

        # Build input index mappings
        # size=None means scalar, size>=1 means array (even size=1)
        offset = 0
        for om_name, size in primal_inputs:
            if size is not None:
                self._input_slices[om_name] = slice(offset, offset + size)
                offset += size
            else:
                self._input_idx[om_name] = offset
                offset += 1
        self._n_primal_inputs = offset

        # Build output index mappings
        offset = 0
        for om_name, size in primal_outputs:
            if size is not None:
                self._output_slices[om_name] = slice(offset, offset + size)
                offset += size
            else:
                self._output_idx[om_name] = offset
                offset += 1
        self._n_primal_outputs = offset

        # Store runtime-resolved lists
        self._runtime_primal_inputs = primal_inputs
        self._runtime_primal_outputs = primal_outputs

    def _resolve_dynamic_sizes(self, inputs):
        """
        Resolve 'dynamic' sizes to actual sizes from OpenMDAO inputs.

        Parameters
        ----------
        inputs : dict-like
            OpenMDAO inputs dictionary

        Returns
        -------
        tuple
            (resolved_primal_inputs, resolved_primal_outputs)
        """
        resolved_inputs = []
        for om_name, size in self._primal_inputs:
            if size == 'dynamic':
                # Get actual size from OpenMDAO input
                # Always treat shape_by_conn variables as arrays, even if size is 1
                actual_size = np.size(inputs[om_name])
                resolved_inputs.append((om_name, actual_size))
            else:
                resolved_inputs.append((om_name, size))

        resolved_outputs = []
        for om_name, size in self._primal_outputs:
            if size == 'dynamic':
                # For outputs, we need to infer from corresponding input
                # This is element-specific; for now assume same as declared
                resolved_outputs.append((om_name, size))
            else:
                resolved_outputs.append((om_name, size))

        return resolved_inputs, resolved_outputs

    # =========================================================================
    # Input/Output Accessors for compute_physics
    # =========================================================================

    def inp(self, inputs, om_name):
        """
        Get an input value from the input vector by OpenMDAO name.

        The dictionary lookup happens at JAX trace time, not runtime,
        so there is zero performance cost in the compiled function.

        Parameters
        ----------
        inputs : jnp.ndarray
            The input vector passed to compute_physics
        om_name : str
            The OpenMDAO input name (e.g., 'Fl_I:tot:P')

        Returns
        -------
        float or jnp.ndarray
            The input value (scalar or array slice)
        """
        if om_name in self._input_idx:
            return inputs[self._input_idx[om_name]]
        elif om_name in self._input_slices:
            return inputs[self._input_slices[om_name]]
        else:
            raise KeyError(f"Input '{om_name}' not registered as primal input. "
                          f"Registered inputs: {[name for name, _ in self._primal_inputs]}")

    # =========================================================================
    # Flow Registration Helpers
    # =========================================================================

    def add_flow_total_primal_outputs(self, fl_name='Fl_O'):
        """
        Register standard total flow properties as primal outputs.

        Registers: P, T, h, S, gamma, Cp, Cv, rho, R (in that order).
        Does NOT include W (mass flow) - register that separately if needed.
        """
        for prop, _, _ in _TOTAL_PROPS:
            self.add_primal_output(f'{fl_name}:tot:{prop}')

    def add_flow_static_primal_outputs(self, fl_name='Fl_O'):
        """
        Register standard static flow properties as primal outputs.

        Registers: h, T, P, rho, gamma, Cp, Cv, S, R, V, Vsonic, MN, area, Wc
        """
        for prop, _, _ in _STATIC_PROPS:
            self.add_primal_output(f'{fl_name}:stat:{prop}')

    def compute_physics(self, inputs):
        """
        Pure JAX-traceable computation. Override in subclass.

        Parameters
        ----------
        inputs : jnp.ndarray
            Flat input vector containing all primal inputs.
            Use self.inp(inputs, 'name') to access values.

        Returns
        -------
        jnp.ndarray
            Flat output vector in order of add_primal_output calls.
        """
        raise NotImplementedError("Subclass must implement compute_physics")

    def _get_jit_config_key(self):
        """
        Return a hashable key for JIT function caching.

        This key identifies the configuration that affects compute_physics behavior.
        Instances with the same key can share the same JIT-compiled JVP function,
        avoiding redundant JAX tracing.

        Automatically scans all declared options using OpenMDAO's options system.
        For recordable options, the value is included directly.
        For non-recordable options (like thermo_data objects), id() is used.

        Returns
        -------
        tuple
            Hashable configuration key
        """
        # Start with class name to distinguish different element types
        key_parts = [type(self).__name__]

        # Include all options - iterate through the internal dict to access metadata
        for name, meta in self.options._dict.items():
            value = self.options[name]
            recordable = meta.get('recordable', True)

            if recordable:
                # For recordable options, include the value directly
                # Must be hashable (bool, int, float, str, tuple, None, etc.)
                try:
                    hash(value)
                    key_parts.append((name, value))
                except TypeError:
                    # Value not hashable, use id
                    key_parts.append((name, id(value)))
            else:
                # For non-recordable options (objects), use id
                key_parts.append((name, id(value)))

        # Include I/O structure (input/output names and sizes affect tracing)
        # Use runtime-resolved lists if available (handles dynamic sizes)
        primal_inputs = self._runtime_primal_inputs or self._primal_inputs
        primal_outputs = self._runtime_primal_outputs or self._primal_outputs
        key_parts.append(('_primal_inputs', tuple(primal_inputs)))
        key_parts.append(('_primal_outputs', tuple(primal_outputs)))

        # Include thermo object identity (may be initialized lazily)
        key_parts.append(('_jax_thermo', id(self._jax_thermo) if self._jax_thermo is not None else None))

        return tuple(key_parts)

    def _make_jit_compute_wrapper(self):
        """
        Create a JIT-compiled wrapper around compute_physics.

        For CEA thermo, threads warm-start caches through the JIT boundary
        as explicit function arguments. For other thermos (e.g. TABULAR)
        or elements without thermo, just JIT-compiles compute_physics directly.

        Returns
        -------
        tuple
            (jit_fn, thermo_type) where thermo_type indicates the calling convention.
        """
        compute_physics = self.compute_physics

        # Ensure thermo is initialized before JIT tracing starts.
        # If thermo_data is None, the element doesn't use thermo.
        if self.options['thermo_data'] is not None:
            thermo = self.jax_thermo  # Trigger lazy init outside JIT scope
        else:
            thermo = self._jax_thermo  # May be None for non-thermo elements

        if thermo is not None:
            from pycycle.functional_thermo.cea.jax_cea import JaxCEAThermo
            if isinstance(thermo, JaxCEAThermo):
                def wrapper(input_vec, cached_n, cached_pi, cached_MN):
                    # Pre-load caches as traced values
                    thermo._cached_n = cached_n
                    thermo._cached_pi = cached_pi
                    thermo._cached_MN = cached_MN
                    output_vec = compute_physics(input_vec)
                    # Return updated caches
                    return output_vec, thermo._cached_n, thermo._cached_pi, thermo._cached_MN

                return jax.jit(wrapper), 'CEA'

        return jax.jit(compute_physics), 'OTHER'

    def compute(self, inputs, outputs):
        """Pack inputs, call JIT-compiled compute_physics, unpack outputs."""
        # Handle dynamic sizes on first call
        if self._needs_runtime_rebuild:
            resolved_inputs, resolved_outputs = self._resolve_dynamic_sizes(inputs)
            self._build_index_mappings(resolved_inputs, resolved_outputs)
            self._needs_runtime_rebuild = False

        # Get the runtime-resolved primal lists (or original if no dynamic sizes)
        primal_inputs = self._runtime_primal_inputs or self._primal_inputs
        primal_outputs = self._runtime_primal_outputs or self._primal_outputs

        # Pack OpenMDAO inputs into flat vector
        # Use dtype from inputs to support complex step derivatives
        sample_name = primal_inputs[0][0]
        input_vec = np.zeros(self._n_primal_inputs, dtype=inputs[sample_name].dtype)
        for om_name, size in primal_inputs:
            if om_name in self._input_slices:
                input_vec[self._input_slices[om_name]] = inputs[om_name]
            else:
                input_vec[self._input_idx[om_name]] = inputs[om_name][0]

        input_vec = jnp.array(input_vec)

        # Detect complex step (used by OpenMDAO check_partials with method='cs')
        is_complex = np.issubdtype(inputs[sample_name].dtype, np.complexfloating)

        if is_complex:
            # Bypass JIT for complex-step derivative checks.
            # JAX JIT re-traces for complex dtypes, which causes issues:
            # - CEA wrapper has side effects (cache assignment) that leak tracers
            # - Re-tracing is expensive and unnecessary for CS validation
            # Call compute_physics directly instead.
            t_start = time.perf_counter()
            output_vec = self.compute_physics(input_vec)
        else:
            # Get or create JIT-compiled compute function
            config_key = self._get_jit_config_key()
            if config_key not in JaxElement._jit_compute_cache:
                JaxElement._jit_compute_cache[config_key] = self._make_jit_compute_wrapper()

            jit_fn, thermo_type = JaxElement._jit_compute_cache[config_key]

            # Call JIT-compiled computation with timing
            t_start = time.perf_counter()
            if thermo_type == 'CEA':
                thermo = self.jax_thermo
                output_vec, new_n, new_pi, new_MN = jit_fn(
                    input_vec, thermo._cached_n, thermo._cached_pi, thermo._cached_MN)
                thermo._cached_n = new_n
                thermo._cached_pi = new_pi
                thermo._cached_MN = new_MN
            else:
                output_vec = jit_fn(input_vec)
        _jax_element_timing_stats['compute_calls'] += 1
        _jax_element_timing_stats['compute_time'] += (time.perf_counter() - t_start)

        # Validate output size
        expected = self._n_primal_outputs
        actual = len(output_vec)
        if actual != expected:
            raise ValueError(
                f"{type(self).__name__}.compute_physics returned {actual} outputs, "
                f"but {expected} were registered via add_primal_output."
            )

        # Unpack to OpenMDAO outputs
        output_vec = np.asarray(output_vec)
        for om_name, size in primal_outputs:
            if om_name in self._output_slices:
                outputs[om_name] = output_vec[self._output_slices[om_name]]
            else:
                outputs[om_name] = output_vec[self._output_idx[om_name]]

        # Pass through composition and FAR (not part of JAX computation)
        if hasattr(self, '_passthrough_vars'):
            for src, dst in self._passthrough_vars:
                outputs[dst] = inputs[src]

        # Cache input vector for partials
        self._cached_input_vec = input_vec

    def compute_partials(self, inputs, partials):
        """Compute partial derivatives using JAX autodiff."""
        input_vec = self._cached_input_vec

        # Get the runtime-resolved primal lists (or original if no dynamic sizes)
        primal_inputs = self._runtime_primal_inputs or self._primal_inputs
        primal_outputs = self._runtime_primal_outputs or self._primal_outputs

        # Compute full Jacobian
        t_start = time.perf_counter()
        jac = self._compute_jacobian(input_vec)
        _jax_element_timing_stats['jacobian_compute_calls'] += 1
        _jax_element_timing_stats['jacobian_compute_time'] += (time.perf_counter() - t_start)

        # Convert JAX array to numpy once (avoid repeated conversions in loop)
        jac_np = np.asarray(jac)

        # Assign to partials dict
        t_start = time.perf_counter()
        for out_name, out_size in primal_outputs:
            for in_name, in_size in primal_inputs:
                # Get row/column indices from mappings
                if out_name in self._output_slices:
                    out_slice = self._output_slices[out_name]
                    is_out_array = True
                else:
                    out_slice = self._output_idx[out_name]
                    is_out_array = False

                if in_name in self._input_slices:
                    in_slice = self._input_slices[in_name]
                    is_in_array = True
                else:
                    in_slice = self._input_idx[in_name]
                    is_in_array = False

                # Extract submatrix from numpy Jacobian (fast indexing)
                sub_jac = jac_np[out_slice, in_slice]

                # Handle scalar vs array shapes for OpenMDAO
                if not is_out_array and not is_in_array:
                    # scalar -> scalar
                    partials[out_name, in_name] = float(sub_jac)
                elif not is_out_array and is_in_array:
                    # array -> scalar: row vector
                    partials[out_name, in_name] = sub_jac.reshape(1, -1)
                elif is_out_array and not is_in_array:
                    # scalar -> array: column vector
                    partials[out_name, in_name] = sub_jac.reshape(-1, 1)
                else:
                    # array -> array: matrix
                    partials[out_name, in_name] = sub_jac

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

    def _compute_jacobian(self, input_vec):
        """
        Compute full Jacobian matrix using JAX.

        Uses forward-mode AD (jacfwd) which is efficient when
        n_inputs <= n_outputs.

        For CEA thermo, uses the cache-threading wrapper with argnums=0
        so that jacfwd differentiates only w.r.t. the input vector while
        the warm-start caches are treated as constants. This prevents
        tracer leaks from CEA thermo's cache side effects.

        Parameters
        ----------
        input_vec : jnp.ndarray
            The input vector

        Returns
        -------
        jnp.ndarray
            Jacobian matrix of shape (n_outputs, n_inputs)
        """
        config_key = self._get_jit_config_key()

        if config_key not in JaxElement._jit_jvp_cache:
            _jax_element_timing_stats['jit_cache_misses'] += 1

            # Create JIT-compiled Jacobian function
            compute_physics = self.compute_physics
            jac_fn = jax.jacfwd(compute_physics)
            JaxElement._jit_jvp_cache[config_key] = jax.jit(jac_fn)
        else:
            _jax_element_timing_stats['jit_cache_hits'] += 1

        jit_jac_fn = JaxElement._jit_jvp_cache[config_key]

        t_jvp = time.perf_counter()
        jac = jit_jac_fn(input_vec)
        _jax_element_timing_stats['jvp_calls'] += 1
        _jax_element_timing_stats['jvp_time'] += (time.perf_counter() - t_jvp)

        return jac

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

    def add_flow_output(self, fl_name='Fl_O', statics=True, fl_src='Fl_I',
                        passthrough_composition=True, passthrough_FAR=True):
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
        passthrough_composition : bool
            If True (default), composition is passed through from fl_src input
            to fl_name output outside of JAX. If False, composition is treated
            as a primal output that must be computed by compute_physics.
        passthrough_FAR : bool
            If True (default), FAR is passed through from fl_src input to
            fl_name output outside of JAX. If False, FAR is treated as a
            primal output that must be computed by compute_physics.
        """
        if not hasattr(self, '_passthrough_vars'):
            self._passthrough_vars = []

        # Total properties
        for prop, val, units in _TOTAL_PROPS:
            kwargs = {'val': val, 'units': units}
            if prop == 'P':
                kwargs['lower'] = 1e-4
            self.add_output(f'{fl_name}:tot:{prop}', **kwargs)

        if passthrough_composition:
            self.add_output(f'{fl_name}:tot:composition', shape_by_conn=True,
                            copy_shape=f'{fl_src}:tot:composition', primal=False)
            self._passthrough_vars.append(
                (f'{fl_src}:tot:composition', f'{fl_name}:tot:composition'))
        else:
            self.add_output(f'{fl_name}:tot:composition', shape_by_conn=True,
                            copy_shape=f'{fl_src}:tot:composition')

        # Static properties
        if statics:
            for prop, val, units in _STATIC_PROPS:
                self.add_output(f'{fl_name}:stat:{prop}', val=val, units=units)

        # Always output mass flow and FAR
        self.add_output(f'{fl_name}:stat:W', val=1.0, units='lbm/s')

        if passthrough_FAR:
            self.add_output(f'{fl_name}:FAR', val=0.0, primal=False)
            self._passthrough_vars.append(
                (f'{fl_src}:FAR', f'{fl_name}:FAR'))
        else:
            self.add_output(f'{fl_name}:FAR', val=0.0)
