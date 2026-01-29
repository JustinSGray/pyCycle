# OpenMDAO Component Interface Reference

This document describes the key interfaces for building libraries on top of OpenMDAO, based on patterns observed in the pyCycle codebase.

## Overview

OpenMDAO has three primary building blocks:
- **Component**: The computational unit (has inputs, outputs, and computes something)
- **Group**: A container that holds components and/or other groups
- **Problem**: The top-level container that wraps the model and provides the execution interface

## 1. Components

Components are the fundamental computational units. There are two types:
- `om.ExplicitComponent`: Output is computed directly from inputs (`y = f(x)`)
- `om.ImplicitComponent`: Defines residuals that must be driven to zero (`R(x, y) = 0`)

### Key Methods

| Method | Purpose | When Called |
|--------|---------|-------------|
| `initialize()` | Declare options (configuration parameters) | During instantiation, before setup |
| `setup()` | Declare inputs/outputs and their metadata | During `prob.setup()` |
| `compute()` | Calculate outputs from inputs | During `prob.run_model()` |
| `compute_partials()` | Calculate analytical derivatives | During optimization or `check_partials()` |

### Options vs Inputs vs Outputs

| Type | Defined In | When Set | Can Change During Run | Purpose |
|------|------------|----------|----------------------|---------|
| **Options** | `initialize()` | Instantiation time | No (fixed at setup) | Configuration, structure, behavior switches |
| **Inputs** | `setup()` | After `prob.setup()` | Yes | Runtime data flowing into component |
| **Outputs** | `setup()` | Computed | Yes (by component) | Runtime data flowing out of component |

### Example: ExplicitComponent

```python
import openmdao.api as om

class MachPressureLossMap(om.ExplicitComponent):
    """
    Calculates pressure loss across the duct as a function of Mach number.
    """

    def initialize(self):
        """
        Declare OPTIONS - configuration parameters set at instantiation time.
        Options are fixed after setup and cannot change during model execution.
        Use options for:
          - Switching between modes (design vs off-design)
          - Configuring component structure (number of ports, etc.)
          - Setting constants that affect what inputs/outputs exist
        """
        self.options.declare('design', default=True,
                             desc='Switch between on-design and off-design calculation.')
        self.options.declare('expMN', default=0.0,
                             desc='MN exponent for loss calculations')

    def setup(self):
        """
        Declare INPUTS and OUTPUTS.
        - Inputs: values that flow into this component from elsewhere
        - Outputs: values computed by this component

        Can use options to conditionally create different I/O structures.
        """
        design = self.options['design']

        # Inputs - data coming into the component
        self.add_input('MN_in', val=0.0,
                       desc='Mach number entering duct')

        if design:
            # In design mode: dPqP is input, s_dPqP is computed
            self.add_input('dPqP', val=0.0,
                           desc='Pressure differential as a fraction of incoming pressure')
            self.add_output('s_dPqP', val=0.0,
                            desc='Pressure loss scalar')
            self.declare_partials('s_dPqP', ['dPqP', 'MN_in'])
        else:
            # In off-design mode: s_dPqP is input, dPqP is computed
            self.add_input('s_dPqP', val=0.0,
                           desc='Pressure loss scalar')
            self.add_output('dPqP', val=0.0,
                            desc='Pressure differential as a fraction of incoming pressure')
            self.declare_partials('dPqP', ['s_dPqP', 'MN_in'])

    def compute(self, inputs, outputs):
        """
        Calculate outputs from inputs.
        - Access options: self.options['option_name']
        - Read inputs: inputs['input_name']
        - Write outputs: outputs['output_name'] = value
        """
        design = self.options['design']
        expMN = self.options['expMN']

        if design:
            outputs['s_dPqP'] = inputs['dPqP'] / inputs['MN_in']**expMN
        else:
            outputs['dPqP'] = inputs['s_dPqP'] * inputs['MN_in']**expMN

    def compute_partials(self, inputs, J):
        """
        Calculate analytical partial derivatives.
        J['output', 'input'] = derivative of output w.r.t. input
        """
        design = self.options['design']
        expMN = self.options['expMN']

        if design:
            J['s_dPqP', 'dPqP'] = 1.0 / inputs['MN_in']**expMN
            J['s_dPqP', 'MN_in'] = -expMN * inputs['dPqP'] * inputs['MN_in']**(-expMN-1.0)
        else:
            J['dPqP', 's_dPqP'] = inputs['MN_in']**expMN
            J['dPqP', 'MN_in'] = expMN * inputs['s_dPqP'] * inputs['MN_in']**(expMN-1.0)

    # Alternative: compute_jacvec_product for matrix-free derivatives
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Compute Jacobian-vector product (alternative to compute_partials).
        Use when forming full Jacobian is expensive or impossible.

        mode='fwd': d_inputs -> d_outputs  (forward mode)
        mode='rev': d_outputs -> d_inputs  (reverse mode)
        """
        if mode == 'fwd':
            if 'MN_in' in d_inputs:
                d_outputs['s_dPqP'] += self.dsdMN * d_inputs['MN_in']
        elif mode == 'rev':
            if 'MN_in' in d_inputs:
                d_inputs['MN_in'] += self.dsdMN * d_outputs['s_dPqP']
```

### add_input() Parameters

```python
self.add_input(
    name,                    # str: Variable name in component's namespace
    val=1.0,                 # float/array: Initial value (default 1.0, also sets shape)
    shape=None,              # int/tuple: Explicit shape (not required if val is array)
    units=None,              # str: Physical units (e.g., 'lbm/s', 'degR', 'lbf/inch**2')
    desc='',                 # str: Description string
    tags=None,               # str/list: User-defined tags for filtering
    shape_by_conn=False,     # bool: If True, shape determined by connected output
    copy_shape=None,         # str: Match shape of named variable in this component
    distributed=None,        # bool: Whether distributed across MPI processes
)
```

### add_output() Parameters

```python
self.add_output(
    name,                    # str: Variable name in component's namespace
    val=1.0,                 # float/array: Initial value (default 1.0, also sets shape)
    shape=None,              # int/tuple: Explicit shape (not required if val is array)
    units=None,              # str: Physical units
    desc='',                 # str: Description string
    lower=None,              # float: Lower bound (for optimization, in user-defined units)
    upper=None,              # float: Upper bound (for optimization, in user-defined units)
    ref=1.0,                 # float: Scaling reference value
    ref0=0.0,                # float: Scaling reference offset
    res_ref=None,            # float: Residual scaling (defaults to ref)
    tags=None,               # str/list: User-defined tags for filtering
    distributed=None,        # bool: Whether distributed across MPI processes
)
```

### options.declare() Parameters

```python
self.options.declare(
    name,                    # str: Option identifier
    default=UNDEFINED,       # object: Default value (must satisfy constraints below)
    values=None,             # set/list/tuple: Allowed values (enum-like constraint)
    types=None,              # type/tuple: Allowed types
    desc='',                 # str: Description
    upper=None,              # float: Maximum allowable value
    lower=None,              # float: Minimum allowable value
    check_valid=None,        # callable: Custom validation function(name, value)
    allow_none=False,        # bool: If True, None is valid regardless of types/values
    recordable=True,         # bool: Whether to include in case recording
    set_function=None,       # callable: Pre-process value before setting
    deprecation=None,        # str/tuple: Deprecation warning message
)
```

### ImplicitComponent

For components defined by residual equations rather than explicit functions:

```python
class MyImplicitComp(om.ImplicitComponent):
    """
    Implicit components solve: R(inputs, outputs) = 0
    Outputs are NOT computed directly; instead residuals are computed
    and a solver finds outputs that drive residuals to zero.
    """

    def initialize(self):
        # Same as ExplicitComponent
        self.options.declare('option_name', default=value)

    def setup(self):
        # Same as ExplicitComponent
        self.add_input('x', val=1.0)
        self.add_output('y', val=1.0)
        self.declare_partials('y', ['x', 'y'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute residuals given inputs and outputs.
        The solver will adjust outputs until residuals are zero.
        """
        x = inputs['x']
        y = outputs['y']
        residuals['y'] = y**2 - x  # Solving y = sqrt(x)

    def solve_nonlinear(self, inputs, outputs):
        """
        (Optional) Provide a direct solution if known.
        If not provided, the framework uses iterative solvers.
        """
        outputs['y'] = inputs['x'] ** 0.5

    def linearize(self, inputs, outputs, jacobian):
        """
        Compute partial derivatives of residuals.
        jacobian['residual', 'variable'] = d(residual)/d(variable)
        """
        jacobian['y', 'x'] = -1.0
        jacobian['y', 'y'] = 2.0 * outputs['y']
```

**Key Differences from ExplicitComponent:**
- Use `apply_nonlinear()` instead of `compute()` to define residuals
- Outputs are implicit (solved for), not directly computed
- Use `linearize()` instead of `compute_partials()` for derivatives
- Optional `solve_nonlinear()` provides direct solution if available

### Dynamic Input/Output Creation

Options can control how many inputs/outputs are created:

```python
class Shaft(om.ExplicitComponent):
    """Calculates power balance for shaft with variable number of ports."""

    def initialize(self):
        self.options.declare('num_ports', default=2,
                             desc="number shaft connections to make")

    def setup(self):
        num_ports = self.options['num_ports']

        # Fixed inputs/outputs
        self.add_input('Nmech', val=1000.0, units="rpm")
        self.add_output('pwr_net', val=1.0, units='hp')

        # Dynamic inputs created in a loop based on option
        self.trq_vars = []
        for i in range(num_ports):
            trq_var_name = 'trq_{:d}'.format(i)
            self.add_input(trq_var_name, val=0., units='ft*lbf')
            self.trq_vars.append(trq_var_name)
            self.declare_partials(['pwr_net'], trq_var_name)

    def compute(self, inputs, outputs):
        # Iterate over dynamic inputs
        total_trq = 0
        for trq_var in self.trq_vars:
            total_trq += inputs[trq_var]
        outputs['pwr_net'] = total_trq * inputs['Nmech'] * self.convert
```

---

## 2. Groups

Groups are containers that hold subsystems (components or other groups) and define how they connect.

### Key Methods

| Method | Purpose |
|--------|---------|
| `initialize()` | Declare group-level options |
| `setup()` | Add subsystems, make connections, configure solvers |
| `configure()` | Modify child settings after hierarchy is built (cannot add subsystems) |
| `add_subsystem()` | Add a component or group to this group |
| `connect()` | Connect output of one subsystem to input of another |
| `set_input_defaults()` | Set default values for promoted inputs |

**Note on setup() vs configure():**
- `setup()`: Called to build the group structure. Add subsystems and connections here.
- `configure()`: Called after all subsystems are instantiated. Use to modify child settings but cannot add new subsystems.

### Example: Simple Group

```python
import openmdao.api as om

class Duct(om.Group):
    """
    A duct element that models pressure loss.
    """

    def initialize(self):
        """Declare group-level options that configure structure."""
        self.options.declare('design', default=True,
                             desc='Switch between on-design and off-design.')
        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')

    def setup(self):
        design = self.options['design']
        statics = self.options['statics']

        # Add subsystems with add_subsystem()
        # Syntax: add_subsystem(name, instance, promotes_inputs=[], promotes_outputs=[])

        self.add_subsystem('flow_in', FlowIn(),
                           promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        if design:
            self.add_subsystem('dP_map', MachPressureLossMap(design=True),
                               promotes_inputs=['dPqP', ('MN_in', 'Fl_I:stat:MN')],
                               promotes_outputs=['s_dPqP'])
        else:
            self.add_subsystem('dP_map', MachPressureLossMap(design=False),
                               promotes_inputs=['s_dPqP', ('MN_in', 'Fl_I:stat:MN')],
                               promotes_outputs=['dPqP'])

        self.add_subsystem('p_loss', PressureLoss(),
                           promotes_inputs=['dPqP', ('Pt_in', 'Fl_I:tot:P')])

        # Connect subsystems with connect()
        # Syntax: connect('source_subsys.output', 'target_subsys.input')
        self.connect('p_loss.Pt_out', 'real_flow.P')

        if statics:
            self.add_subsystem('out_stat', Thermo(mode='static_Ps'),
                               promotes_outputs=['Fl_O:stat:*'])
            self.connect('real_flow.flow:S', 'out_stat.S')
```

### add_subsystem() Parameters

```python
self.add_subsystem(
    name,                      # str: Subsystem name (used for dot-path access)
    subsys,                    # System: Component or Group instance
    promotes=None,             # iter: Variables to promote (both inputs and outputs)
    promotes_inputs=None,      # iter: Inputs to promote (or ['*'] for all)
    promotes_outputs=None,     # iter: Outputs to promote (or ['*'] for all)
    min_procs=1,               # int: Minimum MPI processes for this subsystem
    max_procs=None,            # int: Maximum MPI processes for this subsystem
    proc_weight=1.0,           # float: Weight for MPI process allocation
    proc_group=None,           # str: Processor group identifier
)

# Promotion with renaming:
# ('local_name', 'promoted_name') - renames variable when promoting
promotes_inputs=['dPqP', ('MN_in', 'Fl_I:stat:MN')]
# 'dPqP' keeps its name, 'MN_in' becomes 'Fl_I:stat:MN' at group level
```

### connect() Parameters

```python
self.connect(
    src_name,                  # str: Source variable name (output)
    tgt_name,                  # str/list: Target variable name(s) (input)
    src_indices=None,          # int/list/ndarray: Indices of source to connect
    flat_src_indices=None,     # bool: If True, interpret indices as flattened array
)
```

### Variable Promotion

Promotion exposes a subsystem's variable at the group level:

```python
# Without promotion - must use full path
self.connect('comp1.output', 'comp2.input')
prob['group.comp1.output']

# With promotion - variable accessible at group level
self.add_subsystem('comp1', MyComp(), promotes_outputs=['output'])
prob['group.output']  # No need for comp1 in path
```

### Promotion patterns:
- `promotes=['*']` - promote all inputs and outputs
- `promotes_inputs=['a', 'b']` - promote specific inputs
- `promotes_outputs=['x']` - promote specific outputs
- `promotes_inputs=[('local', 'renamed')]` - promote with renaming

### set_input_defaults()

Set default values for promoted inputs that share the same name:

```python
self.set_input_defaults(
    name,                      # str: Promoted input name
    val=UNDEFINED,             # object: Default value
    units=None,                # str: Units for the input
    src_shape=None,            # int/tuple: Assumed shape of connected source
)

# Example: Set defaults before setup
cycle.set_input_defaults('MN', 0.5)
cycle.set_input_defaults('P', 17., units='psi')
```

---

## 3. Cycles (Custom Group Pattern)

pyCycle defines a `Cycle` class that extends `om.Group` with domain-specific functionality:

```python
class Cycle(om.Group):
    """
    Base class for thermodynamic cycles that propagates options
    to child elements and manages flow connections.
    """

    def initialize(self):
        self.options.declare('design', default=True,
                             desc='Switch between on-design and off-design.')
        self.options.declare('thermo_method', values=['CEA', 'TABULAR'], default='CEA')
        self.options.declare('thermo_data', default=species_data.janaf)

    def add_subsystem(self, name, subsys, **kwargs):
        """Override to propagate cycle-level options to elements."""
        if isinstance(subsys, Element):
            if 'thermo_method' in subsys.options:
                subsys.options['thermo_method'] = self.options['thermo_method']
        return super().add_subsystem(name, subsys, **kwargs)

    def setup(self):
        # Push cycle-level options to all children
        for child_name, child in self._children.items():
            for opt in ['thermo_method', 'thermo_data', 'design']:
                if opt in child.options:
                    child.options[opt] = self.options[opt]

    def pyc_connect_flow(self, source, target, connect_w=True):
        """Helper to connect multiple flow variables at once."""
        for var in ['tot:P', 'tot:T', 'tot:h', 'tot:S', ...]:
            self.connect(f'{source}:{var}', f'{target}:{var}')
```

### Example: Using a Cycle

```python
class Turbojet(pyc.Cycle):

    def setup(self):
        design = self.options['design']

        # Add engine elements
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('comp', pyc.Compressor(map_data=pyc.AXI5),
                           promotes_inputs=['Nmech'])
        self.add_subsystem('burner', pyc.Combustor(fuel_type='Jet-A(g)'))
        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269),
                           promotes_inputs=['Nmech'])
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2),
                           promotes_inputs=['Nmech'])

        # Connect flow stations (custom helper method)
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')

        # Connect individual variables
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')

        # Add balance components for implicit relationships
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf', rhs_name='Fn_target')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('perf.Fn', 'balance.lhs:W')

        # Configure nonlinear solver
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['maxiter'] = 15
        self.linear_solver = om.DirectSolver()

        super().setup()  # Call parent setup
```

---

## 4. Problems

The Problem is the top-level interface for setting up and running models.

### Basic Usage

```python
import openmdao.api as om

# Create problem and assign model
prob = om.Problem()
prob.model = MyGroup()

# Setup (initializes everything, calls all setup() methods)
prob.setup(check=False, force_alloc_complex=True)

# Set input values (two equivalent syntaxes)
prob.set_val('inlet.Fl_I:stat:W', 100.0, units='lbm/s')  # With units
prob['inlet.Fl_I:stat:W'] = 100.0                        # Without units

# Run the model
prob.run_model()

# Access output values
thrust = prob['perf.Fn']
mass_flow = prob.get_val('inlet.Fl_O:stat:W', units='kg/s')  # With unit conversion
```

### Key Problem Methods

| Method | Purpose |
|--------|---------|
| `setup()` | Initialize the model structure |
| `set_val(name, val, units=None, indices=None)` | Set input value with optional units |
| `get_val(name, units=None, indices=None)` | Get value with optional unit conversion |
| `run_model()` | Execute the model |
| `check_partials()` | Verify analytical derivatives against finite difference |
| `set_solver_print(level, depth)` | Control solver output verbosity |

### Problem Method Signatures

```python
# Setup the problem
prob.setup(
    check=None,                    # bool: Run setup checks
    logger=None,                   # Logger instance
    mode='auto',                   # str: 'fwd', 'rev', or 'auto' for derivatives
    force_alloc_complex=False,     # bool: Allocate complex arrays for cs derivatives
    derivatives=True,              # bool: Whether to compute derivatives
)

# Set values
prob.set_val(
    name,                          # str: Variable path
    val=None,                      # value to set
    units=None,                    # str: Units (converts if different from model)
    indices=None,                  # indices to set (for partial assignment)
)

# Get values
prob.get_val(
    name,                          # str: Variable path
    units=None,                    # str: Units (converts if different from model)
    indices=None,                  # indices to get
    get_remote=False,              # bool: Get values from remote MPI procs
)

# Run the model
prob.run_model(
    case_prefix=None,              # str: Prefix for case recording
    reset_iter_counts=True,        # bool: Reset iteration counters
)

# Check partial derivatives
prob.check_partials(
    out_stream=sys.stdout,         # Output stream for results
    includes=None,                 # list: Components to include
    excludes=None,                 # list: Components to exclude
    compact_print=False,           # bool: Compact output format
    method='fd',                   # str: 'fd' or 'cs' (complex step)
    step=None,                     # float: Step size for finite diff
    form='forward',                # str: 'forward', 'backward', 'central'
    show_only_incorrect=False,     # bool: Only show failing partials
)
```

### Setting Values: set_val() vs Bracket Notation

```python
# set_val() - preferred when units matter
prob.set_val('fc.alt', 35000, units='ft')
prob.set_val('fc.MN', 0.8)

# Bracket notation - simpler but no unit handling
prob['fc.alt'] = 35000  # Must be in model's native units
prob['fc.MN'] = 0.8

# set_input_defaults() - set defaults BEFORE setup (on the model, not prob)
prob.model.set_input_defaults('Nmech', 8000.0, units='rpm')
```

### Testing Components

```python
import unittest
from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

class TestMyComponent(unittest.TestCase):

    def test_values(self):
        prob = Problem()
        prob.model.add_subsystem('comp', MyComponent(), promotes=['*'])
        prob.setup(check=False, force_alloc_complex=True)

        # Set inputs
        prob.set_val('input1', 10.0)
        prob.set_val('input2', 20.0)

        # Run
        prob.run_model()

        # Check outputs
        assert_near_equal(prob['output1'], expected_value, tolerance=1e-6)

    def test_partials(self):
        prob = Problem()
        prob.model.add_subsystem('comp', MyComponent(), promotes=['*'])
        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        # Check derivatives against complex-step
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)
```

---

## 5. Summary: Options vs Inputs vs Outputs

### Options (Configuration)
- **Declared in**: `initialize()` with `self.options.declare()`
- **Set at**: Instantiation time (constructor)
- **Accessed via**: `self.options['name']`
- **Mutable after setup**: No
- **Use for**: Structural configuration, mode switches, number of ports

```python
# Declaration
def initialize(self):
    self.options.declare('design', default=True)
    self.options.declare('num_ports', default=2)

# Usage at instantiation
comp = MyComponent(design=False, num_ports=3)
```

### Inputs (Runtime Data In)
- **Declared in**: `setup()` with `self.add_input()`
- **Set at**: After `prob.setup()`, before or between `run_model()` calls
- **Accessed via**: `inputs['name']` in compute methods, `prob['path.name']` externally
- **Mutable after setup**: Yes
- **Use for**: Data flowing into component, values that change during analysis

```python
# Declaration
def setup(self):
    self.add_input('pressure', val=14.696, units='psi')

# Setting value
prob['comp.pressure'] = 15.0
prob.set_val('comp.pressure', 15.0, units='psi')
```

### Outputs (Runtime Data Out)
- **Declared in**: `setup()` with `self.add_output()`
- **Set at**: Computed by the component in `compute()`
- **Accessed via**: `outputs['name']` in compute methods, `prob['path.name']` externally
- **Mutable after setup**: Yes (by component)
- **Use for**: Computed results, data flowing to downstream components

```python
# Declaration
def setup(self):
    self.add_output('thrust', val=0.0, units='lbf', lower=0.0)

# In compute
def compute(self, inputs, outputs):
    outputs['thrust'] = calculated_value

# Reading value externally
result = prob['comp.thrust']
```

---

## 6. Naming Conventions in pyCycle

pyCycle uses specific naming patterns for flow variables:

```
Fl_I:tot:P    - Flow In, total (stagnation) pressure
Fl_I:stat:W   - Flow In, static property, mass flow rate
Fl_O:tot:T    - Flow Out, total temperature
Fl_O:stat:MN  - Flow Out, static Mach number

# Pattern: [Port]:[Property Type]:[Variable]
# Port: Fl_I (input), Fl_O (output)
# Type: tot (total/stagnation), stat (static)
# Variable: P, T, h, S, W, MN, V, etc.
```
