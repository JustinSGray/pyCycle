"""
Tests for JaxElement base class.

Uses simple test components (no thermo) to verify the core JaxElement
machinery: auto-registration, packing/unpacking, JAX autodiff, passthrough,
scalar and array I/O, dynamic sizing, and primal opt-out.
"""

import unittest

import numpy as np
import jax.numpy as jnp

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.new_elements.jax_element_base import JaxElement, clear_jit_cache


# ============================================================================
# Test Components
# ============================================================================

class ScalarMath(JaxElement):
    """
    Simple element with two scalar inputs and two scalar outputs.
    y1 = a * x1 + x2
    y2 = x1 * x2
    """

    def initialize(self):
        super().initialize()
        self.options.declare('a', default=2.0)

    def setup(self):
        self.add_input('x1', val=1.0)
        self.add_input('x2', val=1.0)
        self.add_output('y1', val=1.0)
        self.add_output('y2', val=1.0)
        self.setup_partials()

    def compute_physics(self, inputs):
        a = self.options['a']
        x1 = self.inp(inputs, 'x1')
        x2 = self.inp(inputs, 'x2')
        y1 = a * x1 + x2
        y2 = x1 * x2
        return jnp.array([y1, y2])


class ArrayMath(JaxElement):
    """
    Element with a mix of scalar and array I/O.
    Takes a scalar 'scale' and array 'vec' (size 3), produces
    scalar 'total' and array 'scaled' (size 3).
    total = sum(vec) * scale
    scaled = vec * scale
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        self.add_input('scale', val=1.0)
        self.add_input('vec', val=np.ones(3))
        self.add_output('total', val=1.0)
        self.add_output('scaled', val=np.ones(3))
        self.setup_partials()

    def compute_physics(self, inputs):
        scale = self.inp(inputs, 'scale')
        vec = self.inp(inputs, 'vec')
        total = jnp.sum(vec) * scale
        scaled = vec * scale
        return jnp.concatenate([jnp.array([total]), scaled])


class WithNonPrimal(JaxElement):
    """
    Element that has some inputs/outputs marked primal=False.
    Only x1 and y1 participate in JAX; x_extra and y_extra do not.
    y1 = x1 ** 2
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        self.add_input('x1', val=2.0)
        self.add_input('x_extra', val=99.0, primal=False)
        self.add_output('y1', val=1.0)
        self.add_output('y_extra', val=0.0, primal=False)
        self.setup_partials()

    def compute_physics(self, inputs):
        x1 = self.inp(inputs, 'x1')
        return jnp.array([x1 ** 2])


class WithPassthrough(JaxElement):
    """
    Element with a passthrough variable (simulates composition/FAR pattern).
    Primal: x -> y = 2*x
    Passthrough: pt_in -> pt_out (identity)
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        self.add_input('x', val=3.0)
        self.add_input('pt_in', val=np.array([10.0, 20.0, 30.0]), primal=False)
        self.add_output('y', val=1.0)
        self.add_output('pt_out', val=np.ones(3), primal=False)

        self._passthrough_vars = [('pt_in', 'pt_out')]

        self.setup_partials()

    def compute_physics(self, inputs):
        x = self.inp(inputs, 'x')
        return jnp.array([2.0 * x])


class WithScalarPassthrough(JaxElement):
    """
    Element with a scalar passthrough variable.
    Primal: x -> y = 3*x
    Passthrough: flag_in -> flag_out (scalar identity)
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        self.add_input('x', val=2.0)
        self.add_input('flag_in', val=0.5, primal=False)
        self.add_output('y', val=1.0)
        self.add_output('flag_out', val=0.0, primal=False)

        self._passthrough_vars = [('flag_in', 'flag_out')]

        self.setup_partials()

    def compute_physics(self, inputs):
        x = self.inp(inputs, 'x')
        return jnp.array([3.0 * x])


class ThreeInputsOneOutput(JaxElement):
    """
    Element where only some inputs affect the output.
    y = x1 + x3  (x2 is unused but still a primal input)
    Tests that unused primal inputs get zero derivatives.
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        self.add_input('x1', val=1.0)
        self.add_input('x2', val=1.0)  # unused in physics
        self.add_input('x3', val=1.0)
        self.add_output('y', val=1.0)
        self.setup_partials()

    def compute_physics(self, inputs):
        x1 = self.inp(inputs, 'x1')
        x3 = self.inp(inputs, 'x3')
        return jnp.array([x1 + x3])


class MultiArrayElement(JaxElement):
    """
    Element with multiple array I/O of different sizes.
    a (size 2) and b (size 3) -> c (size 2) and d (size 3)
    c = a * sum(b)
    d = b + a[0]
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        self.add_input('a', val=np.ones(2))
        self.add_input('b', val=np.ones(3))
        self.add_output('c', val=np.ones(2))
        self.add_output('d', val=np.ones(3))
        self.setup_partials()

    def compute_physics(self, inputs):
        a = self.inp(inputs, 'a')
        b = self.inp(inputs, 'b')
        c = a * jnp.sum(b)
        d = b + a[0]
        return jnp.concatenate([c, d])


class MixedScalarArrayPassthrough(JaxElement):
    """
    Element combining scalar primal, array primal, scalar passthrough,
    and array passthrough in one component.
    Primal: scale (scalar), vec (array 3) -> total (scalar), scaled (array 3)
    Passthrough: meta_in (scalar) -> meta_out, comp_in (array 2) -> comp_out
    """

    def initialize(self):
        super().initialize()

    def setup(self):
        # Primal inputs
        self.add_input('scale', val=2.0)
        self.add_input('vec', val=np.array([1.0, 2.0, 3.0]))
        # Non-primal passthrough inputs
        self.add_input('meta_in', val=0.5, primal=False)
        self.add_input('comp_in', val=np.array([10.0, 20.0]), primal=False)

        # Primal outputs
        self.add_output('total', val=1.0)
        self.add_output('scaled', val=np.ones(3))
        # Non-primal passthrough outputs
        self.add_output('meta_out', val=0.0, primal=False)
        self.add_output('comp_out', val=np.ones(2), primal=False)

        self._passthrough_vars = [('meta_in', 'meta_out'), ('comp_in', 'comp_out')]
        self.setup_partials()

    def compute_physics(self, inputs):
        scale = self.inp(inputs, 'scale')
        vec = self.inp(inputs, 'vec')
        total = jnp.sum(vec) * scale
        scaled = vec * scale
        return jnp.concatenate([jnp.array([total]), scaled])


class OptionControlled(JaxElement):
    """
    Element whose I/O structure changes based on options.
    mode='add': y = x1 + x2
    mode='mul': y = x1 * x2, extra_out = x1 - x2
    """

    def initialize(self):
        super().initialize()
        self.options.declare('mode', default='add', values=['add', 'mul'])

    def setup(self):
        mode = self.options['mode']
        self.add_input('x1', val=2.0)
        self.add_input('x2', val=3.0)
        self.add_output('y', val=1.0)
        if mode == 'mul':
            self.add_output('extra_out', val=1.0)
        self.setup_partials()

    def compute_physics(self, inputs):
        mode = self.options['mode']
        x1 = self.inp(inputs, 'x1')
        x2 = self.inp(inputs, 'x2')
        if mode == 'add':
            return jnp.array([x1 + x2])
        else:
            return jnp.array([x1 * x2, x1 - x2])


# ============================================================================
# Tests
# ============================================================================

class TestScalarMath(unittest.TestCase):
    """Test basic scalar I/O with auto-registration."""

    def setUp(self):
        clear_jit_cache()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('comp', ScalarMath(a=3.0), promotes=['*'])
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_compute(self):
        self.prob['x1'] = 4.0
        self.prob['x2'] = 5.0
        self.prob.run_model()
        # y1 = 3*4 + 5 = 17, y2 = 4*5 = 20
        assert_near_equal(self.prob['y1'], 17.0)
        assert_near_equal(self.prob['y2'], 20.0)

    def test_partials(self):
        self.prob['x1'] = 4.0
        self.prob['x2'] = 5.0
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)

    def test_different_values(self):
        """Verify compute works with different input values."""
        self.prob['x1'] = -2.0
        self.prob['x2'] = 7.0
        self.prob.run_model()
        # y1 = 3*(-2) + 7 = 1, y2 = (-2)*7 = -14
        assert_near_equal(self.prob['y1'], 1.0)
        assert_near_equal(self.prob['y2'], -14.0)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestArrayMath(unittest.TestCase):
    """Test mixed scalar + array I/O."""

    def setUp(self):
        clear_jit_cache()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('comp', ArrayMath(), promotes=['*'])
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_compute(self):
        self.prob['scale'] = 2.0
        self.prob['vec'] = [3.0, 4.0, 5.0]
        self.prob.run_model()
        # total = (3+4+5)*2 = 24, scaled = [6, 8, 10]
        assert_near_equal(self.prob['total'], 24.0)
        assert_near_equal(self.prob['scaled'], [6.0, 8.0, 10.0])

    def test_partials(self):
        self.prob['scale'] = 2.0
        self.prob['vec'] = [3.0, 4.0, 5.0]
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestMultiArrayElement(unittest.TestCase):
    """Test multiple arrays of different sizes."""

    def setUp(self):
        clear_jit_cache()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('comp', MultiArrayElement(), promotes=['*'])
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_compute(self):
        self.prob['a'] = [2.0, 3.0]
        self.prob['b'] = [1.0, 2.0, 3.0]
        self.prob.run_model()
        # c = [2, 3] * (1+2+3) = [12, 18]
        # d = [1, 2, 3] + 2.0 = [3, 4, 5]
        assert_near_equal(self.prob['c'], [12.0, 18.0])
        assert_near_equal(self.prob['d'], [3.0, 4.0, 5.0])

    def test_partials(self):
        self.prob['a'] = [2.0, 3.0]
        self.prob['b'] = [1.0, 2.0, 3.0]
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestNonPrimal(unittest.TestCase):
    """Test primal=False opt-out from auto-registration."""

    def setUp(self):
        clear_jit_cache()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('comp', WithNonPrimal(), promotes=['*'])
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_compute(self):
        self.prob['x1'] = 5.0
        self.prob['x_extra'] = 123.0  # should not affect y1
        self.prob.run_model()
        assert_near_equal(self.prob['y1'], 25.0)

    def test_non_primal_not_in_jax(self):
        """Verify non-primal variables are excluded from JAX computation."""
        comp = self.prob.model.comp
        # x_extra should not be in the primal input set
        self.assertNotIn('x_extra', comp._primal_input_set)
        # y_extra should not be in the primal output set
        self.assertNotIn('y_extra', comp._primal_output_set)
        # Primal vector should only have x1 (1 element)
        self.assertEqual(comp._n_primal_inputs, 1)
        # Primal output should only have y1 (1 element)
        self.assertEqual(comp._n_primal_outputs, 1)

    def test_partials(self):
        self.prob['x1'] = 5.0
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestPassthrough(unittest.TestCase):
    """Test passthrough variable handling (array passthrough)."""

    def setUp(self):
        clear_jit_cache()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('comp', WithPassthrough(), promotes=['*'])
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_compute(self):
        self.prob['x'] = 7.0
        self.prob['pt_in'] = [10.0, 20.0, 30.0]
        self.prob.run_model()
        # y = 2*7 = 14
        assert_near_equal(self.prob['y'], 14.0)
        # pt_out should be identity copy of pt_in
        assert_near_equal(self.prob['pt_out'], [10.0, 20.0, 30.0])

    def test_partials(self):
        self.prob['x'] = 7.0
        self.prob['pt_in'] = [10.0, 20.0, 30.0]
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestScalarPassthrough(unittest.TestCase):
    """Test passthrough with a scalar variable."""

    def setUp(self):
        clear_jit_cache()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('comp', WithScalarPassthrough(), promotes=['*'])
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_compute(self):
        self.prob['x'] = 4.0
        self.prob['flag_in'] = 0.75
        self.prob.run_model()
        assert_near_equal(self.prob['y'], 12.0)
        assert_near_equal(self.prob['flag_out'], 0.75)

    def test_partials(self):
        self.prob['x'] = 4.0
        self.prob['flag_in'] = 0.75
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestMixedScalarArrayPassthrough(unittest.TestCase):
    """Test a component with primal scalars, primal arrays, scalar passthrough,
    and array passthrough all together."""

    def setUp(self):
        clear_jit_cache()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('comp', MixedScalarArrayPassthrough(), promotes=['*'])
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_compute(self):
        self.prob['scale'] = 3.0
        self.prob['vec'] = [2.0, 4.0, 6.0]
        self.prob['meta_in'] = 0.99
        self.prob['comp_in'] = [100.0, 200.0]
        self.prob.run_model()
        # total = (2+4+6)*3 = 36
        assert_near_equal(self.prob['total'], 36.0)
        # scaled = [6, 12, 18]
        assert_near_equal(self.prob['scaled'], [6.0, 12.0, 18.0])
        # passthroughs
        assert_near_equal(self.prob['meta_out'], 0.99)
        assert_near_equal(self.prob['comp_out'], [100.0, 200.0])

    def test_partials(self):
        self.prob['scale'] = 3.0
        self.prob['vec'] = [2.0, 4.0, 6.0]
        self.prob['meta_in'] = 0.99
        self.prob['comp_in'] = [100.0, 200.0]
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestUnusedPrimalInput(unittest.TestCase):
    """Test that unused primal inputs produce zero derivative columns."""

    def setUp(self):
        clear_jit_cache()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('comp', ThreeInputsOneOutput(), promotes=['*'])
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_compute(self):
        self.prob['x1'] = 3.0
        self.prob['x2'] = 999.0  # unused
        self.prob['x3'] = 7.0
        self.prob.run_model()
        assert_near_equal(self.prob['y'], 10.0)

    def test_partials(self):
        self.prob['x1'] = 3.0
        self.prob['x2'] = 999.0
        self.prob['x3'] = 7.0
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)

    def test_zero_derivative_for_unused(self):
        """Verify dy/dx2 == 0 since x2 is unused."""
        self.prob['x1'] = 3.0
        self.prob['x2'] = 999.0
        self.prob['x3'] = 7.0
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, method='cs')
        jac = data['comp']['y', 'x2']['J_fwd']
        assert_near_equal(jac.flatten(), 0.0, tolerance=1e-15)


class TestOptionControlled(unittest.TestCase):
    """Test that options change I/O structure and correct derivatives."""

    def test_add_mode(self):
        clear_jit_cache()
        prob = om.Problem()
        prob.model.add_subsystem('comp', OptionControlled(mode='add'), promotes=['*'])
        prob.setup(check=False, force_alloc_complex=True)
        prob['x1'] = 3.0
        prob['x2'] = 4.0
        prob.run_model()
        assert_near_equal(prob['y'], 7.0)
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)

    def test_mul_mode(self):
        clear_jit_cache()
        prob = om.Problem()
        prob.model.add_subsystem('comp', OptionControlled(mode='mul'), promotes=['*'])
        prob.setup(check=False, force_alloc_complex=True)
        prob['x1'] = 3.0
        prob['x2'] = 4.0
        prob.run_model()
        # y = 3*4 = 12, extra_out = 3-4 = -1
        assert_near_equal(prob['y'], 12.0)
        assert_near_equal(prob['extra_out'], -1.0)
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


class TestAutoRegistration(unittest.TestCase):
    """Test that add_input/add_output auto-register as primal by default."""

    def test_all_inputs_registered(self):
        clear_jit_cache()
        prob = om.Problem()
        comp = ScalarMath()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False)
        # Both x1 and x2 should be primal
        self.assertIn('x1', comp._primal_input_set)
        self.assertIn('x2', comp._primal_input_set)
        self.assertEqual(comp._n_primal_inputs, 2)

    def test_all_outputs_registered(self):
        clear_jit_cache()
        prob = om.Problem()
        comp = ScalarMath()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False)
        # Both y1 and y2 should be primal
        self.assertIn('y1', comp._primal_output_set)
        self.assertIn('y2', comp._primal_output_set)
        self.assertEqual(comp._n_primal_outputs, 2)

    def test_primal_false_excludes(self):
        clear_jit_cache()
        prob = om.Problem()
        comp = WithNonPrimal()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False)
        self.assertIn('x1', comp._primal_input_set)
        self.assertNotIn('x_extra', comp._primal_input_set)
        self.assertIn('y1', comp._primal_output_set)
        self.assertNotIn('y_extra', comp._primal_output_set)

    def test_array_input_auto_sized(self):
        """Array inputs with explicit val should get size from val."""
        clear_jit_cache()
        prob = om.Problem()
        comp = ArrayMath()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False)
        # 'vec' has val=np.ones(3), should be registered as primal
        # size=None since it's not shape_by_conn (size inferred at runtime)
        self.assertIn('vec', comp._primal_input_set)
        # After setup_partials, vec should have size 3 in the index map
        self.assertIn('vec', comp._input_slices)
        vec_slice = comp._input_slices['vec']
        self.assertEqual(vec_slice.stop - vec_slice.start, 3)


class TestIndexMappings(unittest.TestCase):
    """Test that index mappings are built correctly."""

    def test_scalar_indices(self):
        clear_jit_cache()
        prob = om.Problem()
        comp = ScalarMath()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False)
        # x1 added first, x2 second
        self.assertEqual(comp._input_idx['x1'], 0)
        self.assertEqual(comp._input_idx['x2'], 1)
        self.assertEqual(comp._output_idx['y1'], 0)
        self.assertEqual(comp._output_idx['y2'], 1)

    def test_mixed_scalar_array_indices(self):
        clear_jit_cache()
        prob = om.Problem()
        comp = ArrayMath()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False)
        # scale (scalar) at index 0, vec (array 3) at indices 1-3
        self.assertEqual(comp._input_idx['scale'], 0)
        self.assertEqual(comp._input_slices['vec'], slice(1, 4))
        # total (scalar) at index 0, scaled (array 3) at indices 1-3
        self.assertEqual(comp._output_idx['total'], 0)
        self.assertEqual(comp._output_slices['scaled'], slice(1, 4))
        self.assertEqual(comp._n_primal_inputs, 4)
        self.assertEqual(comp._n_primal_outputs, 4)

    def test_non_primal_not_in_mappings(self):
        clear_jit_cache()
        prob = om.Problem()
        comp = WithNonPrimal()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False)
        self.assertNotIn('x_extra', comp._input_idx)
        self.assertNotIn('x_extra', comp._input_slices)
        self.assertNotIn('y_extra', comp._output_idx)
        self.assertNotIn('y_extra', comp._output_slices)


class TestConnectedModel(unittest.TestCase):
    """Test JaxElement in a connected group to verify OpenMDAO integration."""

    def test_two_connected_components(self):
        """Chain two JaxElements: y = a*x + 1, z = a*y + 1."""
        clear_jit_cache()
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp1', ScalarMath(a=2.0))
        model.add_subsystem('comp2', ScalarMath(a=3.0))
        model.connect('comp1.y1', 'comp2.x1')
        model.connect('comp1.y2', 'comp2.x2')

        prob.setup(check=False, force_alloc_complex=True)
        prob['comp1.x1'] = 2.0
        prob['comp1.x2'] = 3.0
        prob.run_model()

        # comp1: y1 = 2*2+3 = 7, y2 = 2*3 = 6
        # comp2: y1 = 3*7+6 = 27, y2 = 7*6 = 42
        assert_near_equal(prob['comp2.y1'], 27.0)
        assert_near_equal(prob['comp2.y2'], 42.0)

        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)

    def test_passthrough_in_chain(self):
        """Verify passthrough vars work when connected downstream."""
        clear_jit_cache()
        prob = om.Problem()
        model = prob.model

        # Source component produces pt_out as passthrough
        model.add_subsystem('src', WithPassthrough())
        # Sink just reads it
        model.add_subsystem('sink', ScalarMath(a=1.0))
        # Connect the passthrough output to a scalar input
        # (just checking it produces valid values)

        prob.setup(check=False, force_alloc_complex=True)
        prob['src.x'] = 5.0
        prob['src.pt_in'] = [1.0, 2.0, 3.0]
        prob.run_model()

        assert_near_equal(prob['src.y'], 10.0)
        assert_near_equal(prob['src.pt_out'], [1.0, 2.0, 3.0])


class TestOutputSizeValidation(unittest.TestCase):
    """Test that compute_physics output size mismatch is caught."""

    def test_wrong_output_count(self):

        class BadElement(JaxElement):
            def initialize(self):
                super().initialize()

            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y1', val=1.0)
                self.add_output('y2', val=1.0)
                self.setup_partials()

            def compute_physics(self, inputs):
                x = self.inp(inputs, 'x')
                # Returns 1 output but 2 are registered
                return jnp.array([x])

        clear_jit_cache()
        prob = om.Problem()
        prob.model.add_subsystem('comp', BadElement(), promotes=['*'])
        prob.setup(check=False, force_alloc_complex=True)

        with self.assertRaises(ValueError) as ctx:
            prob.run_model()
        self.assertIn('returned 1 outputs', str(ctx.exception))
        self.assertIn('2 were registered', str(ctx.exception))


class TestInpAccessorErrors(unittest.TestCase):
    """Test that inp() raises helpful errors for unregistered inputs."""

    def test_unregistered_input(self):
        clear_jit_cache()
        prob = om.Problem()
        comp = WithNonPrimal()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False, force_alloc_complex=True)

        # x_extra is not primal, so inp() should raise KeyError
        dummy_vec = jnp.zeros(comp._n_primal_inputs)
        with self.assertRaises(KeyError) as ctx:
            comp.inp(dummy_vec, 'x_extra')
        self.assertIn('not registered as primal input', str(ctx.exception))


class TestAddPrimalInputOverride(unittest.TestCase):
    """Test that add_primal_input can override auto-registered size."""

    def test_override_to_dynamic(self):
        """Verify add_primal_input can change size after auto-registration."""

        class OverrideElement(JaxElement):
            def initialize(self):
                super().initialize()

            def setup(self):
                self.add_input('x', val=1.0)
                # Auto-registered as scalar (size=None)
                # Override to array size
                self.add_primal_input('x', size=3)
                self.add_output('y', val=1.0)
                self.setup_partials()

            def compute_physics(self, inputs):
                x = self.inp(inputs, 'x')
                return jnp.array([jnp.sum(x)])

        clear_jit_cache()
        prob = om.Problem()
        comp = OverrideElement()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup(check=False)
        # size should be overridden to 3
        self.assertEqual(comp._primal_input_set['x'], 3)


if __name__ == '__main__':
    unittest.main()
