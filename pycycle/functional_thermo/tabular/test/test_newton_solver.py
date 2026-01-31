"""
Tests for the generic NewtonSolver class.
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from pycycle.functional_thermo.tabular.tabular import NewtonSolver


class NewtonSolver1DTestCase(unittest.TestCase):
    """Test 1D Newton solver functionality."""

    def test_simple_quadratic(self):
        """Test solving x^2 - 4 = 0, solution x = 2."""
        def residual_and_jac(x):
            return x**2 - 4.0, 2.0 * x

        solver = NewtonSolver(max_iter=20, tol=1e-12)
        x, converged, n_iter = solver.solve(residual_and_jac, 1.0)

        self.assertTrue(converged)
        assert_allclose(x, 2.0, rtol=1e-6)
        self.assertLess(n_iter, 10)

    def test_cubic_root(self):
        """Test solving x^3 - 2 = 0, solution x = 2^(1/3)."""
        def residual_and_jac(x):
            return x**3 - 2.0, 3.0 * x**2

        solver = NewtonSolver(max_iter=20, tol=1e-12)
        x, converged, n_iter = solver.solve(residual_and_jac, 1.0)

        self.assertTrue(converged)
        assert_allclose(x, 2.0**(1.0/3.0), rtol=1e-10)

    def test_returns_scalar_for_scalar_input(self):
        """Verify that scalar input returns scalar output."""
        def residual_and_jac(x):
            return x - 5.0, 1.0

        solver = NewtonSolver()
        x, converged, n_iter = solver.solve(residual_and_jac, 0.0)

        self.assertTrue(np.isscalar(x))
        self.assertEqual(x, 5.0)

    def test_negative_solution(self):
        """Test solving x^2 - 4 = 0 starting from negative, solution x = -2."""
        def residual_and_jac(x):
            return x**2 - 4.0, 2.0 * x

        solver = NewtonSolver(max_iter=20, tol=1e-12)
        x, converged, n_iter = solver.solve(residual_and_jac, -1.0)

        self.assertTrue(converged)
        assert_allclose(x, -2.0, rtol=1e-6)


class NewtonSolver2DTestCase(unittest.TestCase):
    """Test 2D Newton solver functionality."""

    def test_linear_system(self):
        """Test solving a simple 2D linear system."""
        # x + y = 3
        # x - y = 1
        # Solution: x = 2, y = 1
        def residual_and_jac(x):
            R = np.array([x[0] + x[1] - 3.0, x[0] - x[1] - 1.0])
            J = np.array([[1.0, 1.0], [1.0, -1.0]])
            return R, J

        solver = NewtonSolver(max_iter=10, tol=1e-12)
        x, converged, n_iter = solver.solve(residual_and_jac, np.array([0.0, 0.0]))

        self.assertTrue(converged)
        assert_allclose(x, [2.0, 1.0], rtol=1e-10)
        # Linear system converges in 2 iterations (1 to solve, 1 to verify convergence)
        self.assertLessEqual(n_iter, 2)

    def test_nonlinear_system(self):
        """Test solving a 2D nonlinear system."""
        # x^2 + y^2 = 5
        # x * y = 2
        # Solutions include (2, 1) and (1, 2)
        def residual_and_jac(x):
            R = np.array([x[0]**2 + x[1]**2 - 5.0, x[0] * x[1] - 2.0])
            J = np.array([
                [2.0 * x[0], 2.0 * x[1]],
                [x[1], x[0]]
            ])
            return R, J

        # Start closer to the (2, 1) solution
        solver = NewtonSolver(max_iter=20, tol=1e-10)
        x, converged, n_iter = solver.solve(residual_and_jac, np.array([2.5, 0.8]))

        self.assertTrue(converged)
        # Check that it's a valid solution
        assert_allclose(x[0]**2 + x[1]**2, 5.0, rtol=1e-8)
        assert_allclose(x[0] * x[1], 2.0, rtol=1e-8)

    def test_returns_array_for_array_input(self):
        """Verify that array input returns array output."""
        def residual_and_jac(x):
            R = np.array([x[0] - 1.0, x[1] - 2.0])
            J = np.eye(2)
            return R, J

        solver = NewtonSolver()
        x, converged, n_iter = solver.solve(residual_and_jac, np.array([0.0, 0.0]))

        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (2,))


class NewtonSolverNDTestCase(unittest.TestCase):
    """Test N-D Newton solver functionality."""

    def test_3d_linear_system(self):
        """Test solving a 3D linear system."""
        # x + y + z = 6
        # x - y + z = 2
        # x + y - z = 0
        # Solution: x = 1, y = 2, z = 3
        def residual_and_jac(x):
            R = np.array([
                x[0] + x[1] + x[2] - 6.0,
                x[0] - x[1] + x[2] - 2.0,
                x[0] + x[1] - x[2] - 0.0
            ])
            J = np.array([
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0]
            ])
            return R, J

        solver = NewtonSolver(max_iter=10, tol=1e-12)
        x, converged, n_iter = solver.solve(residual_and_jac, np.zeros(3))

        self.assertTrue(converged)
        assert_allclose(x, [1.0, 2.0, 3.0], rtol=1e-10)


class NewtonSolverBoundsTestCase(unittest.TestCase):
    """Test bounds handling in NewtonSolver."""

    def test_lower_bound_active(self):
        """Test that lower bound is enforced."""
        # Solve x - 5 = 0, but with lower bound at 10
        def residual_and_jac(x):
            return x - 5.0, 1.0

        solver = NewtonSolver(max_iter=20, tol=1e-12, bounds=(10.0, 100.0))
        x, converged, n_iter = solver.solve(residual_and_jac, 50.0)

        # Should hit the lower bound
        self.assertEqual(x, 10.0)

    def test_upper_bound_active(self):
        """Test that upper bound is enforced."""
        # Solve x - 100 = 0, but with upper bound at 50
        def residual_and_jac(x):
            return x - 100.0, 1.0

        solver = NewtonSolver(max_iter=20, tol=1e-12, bounds=(0.0, 50.0))
        x, converged, n_iter = solver.solve(residual_and_jac, 25.0)

        # Should hit the upper bound
        self.assertEqual(x, 50.0)

    def test_2d_bounds(self):
        """Test bounds in 2D problem."""
        # Solve [x - 10, y - 10] = 0, but bounded to [0, 5]
        def residual_and_jac(x):
            R = np.array([x[0] - 10.0, x[1] - 10.0])
            J = np.eye(2)
            return R, J

        solver = NewtonSolver(max_iter=20, tol=1e-12, bounds=(0.0, 5.0))
        x, converged, n_iter = solver.solve(residual_and_jac, np.array([2.0, 2.0]))

        # Should hit upper bounds
        assert_allclose(x, [5.0, 5.0])

    def test_asymmetric_bounds(self):
        """Test different bounds for each variable."""
        def residual_and_jac(x):
            R = np.array([x[0] - 100.0, x[1] + 100.0])
            J = np.eye(2)
            return R, J

        # x bounded [0, 10], y bounded [-5, 5]
        solver = NewtonSolver(max_iter=20, bounds=([0.0, -5.0], [10.0, 5.0]))
        x, converged, n_iter = solver.solve(residual_and_jac, np.array([5.0, 0.0]))

        assert_allclose(x, [10.0, -5.0])


class NewtonSolverConvergenceModeTestCase(unittest.TestCase):
    """Test different convergence modes."""

    def test_absolute_convergence(self):
        """Test absolute convergence mode."""
        def residual_and_jac(x):
            return x - 1000.0, 1.0

        solver = NewtonSolver(convergence_mode='absolute', tol=1e-8)
        x, converged, n_iter = solver.solve(residual_and_jac, 0.0)

        self.assertTrue(converged)
        assert_allclose(x, 1000.0, atol=1e-8)

    def test_relative_convergence(self):
        """Test relative convergence mode."""
        def residual_and_jac(x):
            return x - 1000.0, 1.0

        solver = NewtonSolver(convergence_mode='relative', tol=1e-10, abs_tol=1e-6)
        x, converged, n_iter = solver.solve(residual_and_jac, 0.0, ref_value=1000.0)

        self.assertTrue(converged)
        assert_allclose(x, 1000.0, rtol=1e-10)

    def test_component_convergence(self):
        """Test component-wise convergence mode."""
        def residual_and_jac(x):
            R = np.array([x[0] - 1.0, x[1] - 1000.0])
            J = np.eye(2)
            return R, J

        solver = NewtonSolver(convergence_mode='component', tol=1e-8)
        x, converged, n_iter = solver.solve(residual_and_jac, np.zeros(2))

        self.assertTrue(converged)
        # Each component should be within tolerance
        self.assertLess(abs(x[0] - 1.0), 1e-8)
        self.assertLess(abs(x[1] - 1000.0), 1e-8)


class NewtonSolverLinesearchTestCase(unittest.TestCase):
    """Test Armijo-Goldstein line search functionality."""

    def test_linesearch_prevents_divergence(self):
        """Test that line search prevents divergence on arctan problem."""
        # arctan(x) = 0, solution x = 0
        # Newton without line search can diverge from far starting points
        def residual_and_jac(x):
            return np.arctan(x), 1.0 / (1.0 + x**2)

        # Without line search - may diverge
        solver_no_ls = NewtonSolver(max_iter=50, tol=1e-10, linesearch=False)
        x_no_ls, converged_no_ls, _ = solver_no_ls.solve(residual_and_jac, 10.0)

        # With line search - should converge
        solver_ls = NewtonSolver(max_iter=50, tol=1e-10, linesearch=True)
        x_ls, converged_ls, _ = solver_ls.solve(residual_and_jac, 10.0)

        self.assertTrue(converged_ls)
        assert_allclose(x_ls, 0.0, atol=1e-6)
        # The no-linesearch version likely diverged
        self.assertFalse(converged_no_ls)

    def test_linesearch_converges_standard_problem(self):
        """Test that line search still works on well-behaved problems."""
        def residual_and_jac(x):
            return x**2 - 4.0, 2.0 * x

        solver = NewtonSolver(max_iter=20, tol=1e-12, linesearch=True)
        x, converged, n_iter = solver.solve(residual_and_jac, 1.0)

        self.assertTrue(converged)
        assert_allclose(x, 2.0, rtol=1e-6)

    def test_linesearch_2d_problem(self):
        """Test line search on 2D problem."""
        # Rosenbrock-like: [10*(y - x^2), 1 - x]
        # Solution: (1, 1)
        def residual_and_jac(x):
            R = np.array([10.0 * (x[1] - x[0]**2), 1.0 - x[0]])
            J = np.array([
                [-20.0 * x[0], 10.0],
                [-1.0, 0.0]
            ])
            return R, J

        solver = NewtonSolver(max_iter=50, tol=1e-10, linesearch=True)
        x, converged, n_iter = solver.solve(residual_and_jac, np.array([-2.0, 5.0]))

        self.assertTrue(converged)
        assert_allclose(x, [1.0, 1.0], rtol=1e-8)

    def test_linesearch_parameters(self):
        """Test that line search parameters are respected."""
        def residual_and_jac(x):
            return np.arctan(x), 1.0 / (1.0 + x**2)

        # More aggressive backtracking (smaller rho)
        solver = NewtonSolver(
            max_iter=100, tol=1e-10,
            linesearch=True, ls_c=0.1, ls_rho=0.25, ls_maxiter=10
        )
        x, converged, n_iter = solver.solve(residual_and_jac, 10.0)

        self.assertTrue(converged)
        assert_allclose(x, 0.0, atol=1e-6)


class NewtonSolverEdgeCaseTestCase(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_singular_jacobian(self):
        """Test handling of singular Jacobian."""
        def residual_and_jac(x):
            # Jacobian is zero at x = 0, but we want to solve x^2 - 1 = 0
            # Starting at x = 0 gives singular Jacobian
            return x**2 - 1.0, 2.0 * x

        solver = NewtonSolver(max_iter=20, tol=1e-12)
        x, converged, n_iter = solver.solve(residual_and_jac, 0.0)

        # Should fail gracefully (not crash) due to singular Jacobian at x=0
        self.assertFalse(converged)

    def test_stagnation_detection(self):
        """Test that stagnation is detected and reported as converged."""
        def residual_and_jac(x):
            # Linear function where we can precisely control when stagnation happens
            # f(x) = x - 1, solution x = 1
            # Newton solves this in one step, but we'll use a large stagnation_tol
            # to force early termination
            return x - 1.0, 1.0

        # Large stagnation tolerance to force stagnation detection
        # The step will be exactly |1 - x0|, so starting from 1.0 + 1e-10 gives step 1e-10
        solver = NewtonSolver(max_iter=10, tol=1e-100, stagnation_tol=1e-8)
        x, converged, n_iter = solver.solve(residual_and_jac, 1.0 + 1e-10)

        # Should detect stagnation (step of 1e-10 < stagnation_tol of 1e-8)
        self.assertTrue(converged)
        assert_allclose(x, 1.0, atol=1e-8)

    def test_max_iterations_reached(self):
        """Test that max iterations is respected."""
        def residual_and_jac(x):
            # Slow convergence
            return np.sin(x), np.cos(x)

        solver = NewtonSolver(max_iter=3, tol=1e-15)
        x, converged, n_iter = solver.solve(residual_and_jac, 3.0)

        self.assertEqual(n_iter, 3)
        # May or may not have converged in 3 iterations

    def test_already_converged(self):
        """Test when initial guess is already the solution."""
        def residual_and_jac(x):
            return x - 5.0, 1.0

        solver = NewtonSolver(max_iter=20, tol=1e-12)
        x, converged, n_iter = solver.solve(residual_and_jac, 5.0)

        self.assertTrue(converged)
        self.assertEqual(n_iter, 1)
        self.assertEqual(x, 5.0)


class NewtonSolverReferenceValueTestCase(unittest.TestCase):
    """Test reference value handling for relative convergence."""

    def test_ref_value_scalar(self):
        """Test relative convergence with scalar reference."""
        def residual_and_jac(x):
            return x - 1e6, 1.0

        solver = NewtonSolver(convergence_mode='relative', tol=1e-10)
        x, converged, n_iter = solver.solve(residual_and_jac, 0.0, ref_value=1e6)

        self.assertTrue(converged)
        # Relative error should be < tol
        self.assertLess(abs(x - 1e6) / 1e6, 1e-10)

    def test_ref_value_array(self):
        """Test relative convergence with array reference."""
        def residual_and_jac(x):
            R = np.array([x[0] - 1e3, x[1] - 1e6])
            J = np.eye(2)
            return R, J

        solver = NewtonSolver(convergence_mode='relative', tol=1e-10)
        x, converged, n_iter = solver.solve(
            residual_and_jac,
            np.zeros(2),
            ref_value=np.array([1e3, 1e6])
        )

        self.assertTrue(converged)


if __name__ == '__main__':
    unittest.main()
