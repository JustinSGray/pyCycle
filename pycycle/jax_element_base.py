"""
JaxElement - Base class for pyCycle elements using JAX automatic differentiation.

This provides a clean separation between physics (compute_physics) and OpenMDAO interface.
Derivatives are computed automatically using JAX's autodiff.
"""

import numpy as np
import jax
import jax.numpy as jnp

import openmdao.api as om

from pycycle.constants import (ALLOWED_THERMOS, CEA_AIR_COMPOSITION,
                               TAB_AIR_FUEL_COMPOSITION, AIR_JETA_TAB_SPEC)

# =============================================================================
# Optional timing stats (no-op by default, can be enabled for profiling)
# =============================================================================
def reset_timing_stats():
    """No-op stub for backward compatibility."""
    pass

def print_timing_stats():
    """No-op stub for backward compatibility."""
    pass


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

    def initialize(self):

        self._jax_thermo = None
        self._thermo_cache = {}  # Instance-level cache for linearized derivatives
        self._primal_input_names = []  # Maps primal arg name -> OpenMDAO input name
        self._primal_output_names = []  # Maps primal return index -> OpenMDAO output name
        self._cached_args = None

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
        Get or create a shared JaxThermo for this configuration.

        Uses class-level cache keyed by (thermo_method, thermo_data).
        """
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        key = (thermo_method, id(thermo_data))

        if key not in JaxElement._shared_thermos:
            # Create new JaxThermo and cache at class level
            thermo = self._create_thermo()
            from pycycle.functional_thermo.jax_wrappers import JaxThermo
            JaxElement._shared_thermos[key] = JaxThermo(thermo)

        return JaxElement._shared_thermos[key]

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

    # =========================================================================
    # Hooks for subclass customization
    # =========================================================================

    def _pre_linearize(self, inputs, outputs):
        """
        Pre-linearize thermo at current operating point for faster JAX JVP.

        Uses flow port configuration from `_flow_in_port` and `_flow_out_port`
        attributes. Set these in setup() to enable automatic pre-linearization.

        Parameters
        ----------
        inputs : dict-like
            OpenMDAO inputs dict
        outputs : dict-like
            OpenMDAO outputs dict (cached from last compute)
        """
        # Skip if no jax_thermo or no flow ports configured
        if self._jax_thermo is None:
            return
        if not hasattr(self, '_flow_out_port') or self._flow_out_port is None:
            return

        fl_out = self._flow_out_port
        fl_in = getattr(self, '_flow_in_port', 'Fl_I')
        design = self.options['design']
        statics = self.options['statics'] if 'statics' in self.options else False

        # Get MN_or_area if statics are enabled
        if statics:
            if design:
                MN_or_area = float(inputs['MN'][0]) if 'MN' in inputs else None
            else:
                MN_or_area = float(inputs['area'][0]) if 'area' in inputs else None
        else:
            MN_or_area = None

        # Bind instance cache to shared thermo
        self._jax_thermo.set_cache(self._thermo_cache)

        # Extract props from outputs (computed in forward pass)
        # to avoid duplicate table lookups in linearize()
        from pycycle.functional_thermo.base import TotalProps, StaticProps
        props = TotalProps(
            h=float(outputs[f'{fl_out}:tot:h'][0]),
            S=float(outputs[f'{fl_out}:tot:S'][0]),
            gamma=float(outputs[f'{fl_out}:tot:gamma'][0]),
            Cp=float(outputs[f'{fl_out}:tot:Cp'][0]),
            Cv=float(outputs[f'{fl_out}:tot:Cv'][0]),
            rho=float(outputs[f'{fl_out}:tot:rho'][0]),
            R=float(outputs[f'{fl_out}:tot:R'][0]),
        )

        # Extract static props if available
        static_props = None
        if statics and f'{fl_out}:stat:T' in outputs:
            static_props = StaticProps(
                Ts=float(outputs[f'{fl_out}:stat:T'][0]),
                Ps=float(outputs[f'{fl_out}:stat:P'][0]),
                hs=float(outputs[f'{fl_out}:stat:h'][0]),
                rhos=float(outputs[f'{fl_out}:stat:rho'][0]),
                MN=float(outputs[f'{fl_out}:stat:MN'][0]),
                V=float(outputs[f'{fl_out}:stat:V'][0]),
                Vsonic=float(outputs[f'{fl_out}:stat:Vsonic'][0]),
                area=float(outputs[f'{fl_out}:stat:area'][0]),
                gamma=float(outputs[f'{fl_out}:stat:gamma'][0]),
                Cp=float(outputs[f'{fl_out}:stat:Cp'][0]),
                Cv=float(outputs[f'{fl_out}:stat:Cv'][0]),
                S=float(outputs[f'{fl_out}:stat:S'][0]),
                R=float(outputs[f'{fl_out}:stat:R'][0]),
            )

        # Pre-linearize at current operating point
        # Note: outputs dict contains arrays, extract scalar values
        self._jax_thermo.linearize_at(
            ht=float(outputs[f'{fl_out}:tot:h'][0]),
            Pt=float(outputs[f'{fl_out}:tot:P'][0]),
            W=float(inputs[f'{fl_in}:stat:W'][0]),
            MN_or_area=MN_or_area,
            is_design=design,
            statics=statics,
            composition=inputs[f'{fl_in}:tot:composition'],
            T=float(outputs[f'{fl_out}:tot:T'][0]),
            props=props,
            static_props=static_props
        )

    def _post_linearize(self):
        """
        Hook called after computing Jacobian. Override for cleanup.

        This is called at the end of compute_partials() to allow subclasses
        to clear caches or perform other cleanup.

        Default implementation clears thermo caches if jax_thermo is in use.
        """
        if self._jax_thermo is not None:
            self._jax_thermo.clear_cache()
            self._thermo_cache.clear()

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

        # Call pure computation
        result = self.compute_physics(*args)

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
        # Pre-linearization hook (for thermo caching)
        # Use OpenMDAO's internal _outputs dict for current output values
        # JSG: _outputs being used for speed here. But might be risky. Check with OpenMDAO devs. 
        self._pre_linearize(inputs, self._outputs)

        args = self._cached_args

        # Compute Jacobian (subclass can override compute_jacobian for efficiency)
        if hasattr(self, 'compute_jacobian'):
            jacs = self.compute_jacobian(args)
        else:
            jacs = self._compute_jacobian_fwd(args)

        # Assign to partials dict, handling array primals
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

        # Post-linearization hook (cleanup)
        self._post_linearize()

    def _compute_jacobian_fwd(self, args):
        """Compute Jacobian using forward-mode with sequential JVP calls."""
        n_inputs = len(args)
        n_outputs = len(self._primal_output_names)
        jacs = [np.zeros(n_outputs) for _ in range(n_inputs)]
        args_tuple = tuple(args)

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

            _, jvp_out = jax.jvp(self.compute_physics, args_tuple, tuple(tangents))
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

    # =========================================================================
    # Thermo Creation Helpers
    # =========================================================================

    def _create_thermo(self, fl_name='Fl_I', composition=None):
        """Create a functional thermo object based on options."""
        method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']

        if method == 'CEA':
            from pycycle.functional_thermo import CEAThermo
            from pycycle.thermo.cea import species_data

            if composition is None:
                composition = self.Fl_I_data.get(fl_name, CEA_AIR_COMPOSITION)
            if thermo_data is None:
                thermo_data = species_data.janaf

            return CEAThermo(composition=composition,
                             thermo_data=thermo_data,
                             input_units='English')

        elif method == 'TABULAR':
            from pycycle.functional_thermo import TabularThermo

            if composition is None:
                composition = self.Fl_I_data.get(fl_name, TAB_AIR_FUEL_COMPOSITION)
            FAR = composition.get('FAR', 0.0) if isinstance(composition, dict) else 0.0
            spec = thermo_data if thermo_data else AIR_JETA_TAB_SPEC

            return TabularThermo(FAR=FAR, spec=spec, input_units='English')

        else:
            raise ValueError(f"Unknown thermo_method: {method}")

    def _create_jax_thermo(self, fl_name='Fl_I', composition=None):
        """Create a JAX-wrapped functional thermo object."""
        from pycycle.functional_thermo.jax_wrappers import JaxThermo

        thermo = self._create_thermo(fl_name=fl_name, composition=composition)
        return JaxThermo(thermo)
