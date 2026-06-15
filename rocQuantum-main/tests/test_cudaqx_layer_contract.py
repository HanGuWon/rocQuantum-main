"""Contracts for the experimental CUDA-QX-style helper layer."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestVqeSolverContract(unittest.TestCase):
    def test_solver_capabilities_expose_experimental_cudaqx_boundary(self):
        import rocquantum.solvers as solvers
        from rocquantum.solvers import capabilities, solver_capabilities

        capability_data = solver_capabilities()

        self.assertEqual(capability_data["status"], "experimental_partial")
        self.assertEqual(capabilities(), capability_data)
        self.assertIn("VQE_Solver.evaluate_energy", capability_data["entry_points"])
        self.assertIn(
            "VQE one-shot energy evaluation through rocq.observe()",
            capability_data["supported_features"],
        )
        self.assertIn(
            "finite-real parameter, energy, backend, gradient-method, optimizer-result, optimizer-interface, and optimizer-option validation",
            capability_data["supported_features"],
        )
        self.assertIn(
            "GPU-resident native adjoint differentiation",
            capability_data["unsupported_features"],
        )
        self.assertIn("scipy", capability_data["optional_dependencies"])
        self.assertIn("CUDA-QX", capability_data["comparison_target"])
        self.assertIn("self-hosted ROCm CI", capability_data["performance_note"])
        self.assertIn("solver_capabilities", solvers.__all__)

    def test_objective_uses_canonical_observe(self):
        import rocq
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import VQE_Solver

        @rocq.kernel
        def ansatz(theta):
            q = rocq.qvec(1)
            rocq.rx(theta, q[0])

        solver = VQE_Solver(backend="state_vector")
        hamiltonian = PauliOperator("Z0")
        with mock.patch("rocquantum.solvers.vqe_solver.observe", return_value=-0.25) as patched_observe:
            energy = solver._objective_function(np.array([0.125]), hamiltonian, ansatz, 1)

        self.assertEqual(energy, -0.25)
        patched_observe.assert_called_once_with(ansatz, hamiltonian, 0.125, backend="state_vector")
        self.assertEqual(len(solver._intermediate_results), 1)

    def test_objective_accepts_canonical_sparse_operator(self):
        import rocq
        from rocq.operator import SparseHamiltonianOperator
        from rocquantum.solvers.vqe_solver import VQE_Solver

        @rocq.kernel
        def ansatz(theta):
            q = rocq.qvec(1)
            rocq.rx(theta, q[0])

        solver = VQE_Solver(backend="state_vector")
        hamiltonian = SparseHamiltonianOperator(
            data=np.array([1.0, -1.0], dtype=np.complex128),
            indices=np.array([0, 1], dtype=np.int64),
            indptr=np.array([0, 1, 2], dtype=np.int64),
            shape=(2, 2),
        )
        with mock.patch("rocquantum.solvers.vqe_solver.observe", return_value=-0.5) as patched_observe:
            energy = solver._objective_function(np.array([0.125]), hamiltonian, ansatz, 1)

        self.assertEqual(energy, -0.5)
        patched_observe.assert_called_once_with(ansatz, hamiltonian, 0.125, backend="state_vector")

    def test_public_energy_evaluation_uses_canonical_observe_once(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import VQE_Solver

        def ansatz(theta):
            return None

        solver = VQE_Solver(backend="state_vector")
        hamiltonian = PauliOperator("Z0")
        with mock.patch(
            "rocquantum.solvers.vqe_solver.observe",
            return_value=-0.25 + 1.0e-12j,
        ) as patched_observe:
            energy = solver.evaluate_energy(
                hamiltonian,
                ansatz,
                1,
                parameters=[0.125],
            )

        self.assertEqual(energy, -0.25)
        self.assertEqual(solver._intermediate_results, [])
        patched_observe.assert_called_once_with(ansatz, hamiltonian, 0.125, backend="state_vector")

        with mock.patch(
            "rocquantum.solvers.vqe_solver.observe",
            return_value=-0.25,
        ):
            recorded_energy = solver.evaluate_energy(
                hamiltonian,
                ansatz,
                1,
                parameters=[0.25],
                record_intermediate=True,
            )
        self.assertEqual(recorded_energy, -0.25)
        self.assertEqual(len(solver._intermediate_results), 1)

    def test_parameter_shift_gradient_is_available(self):
        import rocq
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import VQE_Solver

        @rocq.kernel
        def ansatz(theta):
            q = rocq.qvec(1)
            rocq.rx(theta, q[0])

        solver = VQE_Solver()
        hamiltonian = PauliOperator("Z0")
        with mock.patch("rocquantum.solvers.vqe_solver.observe", side_effect=[1.0, 0.0]):
            gradient = solver.estimate_gradient(
                np.array([0.25]),
                hamiltonian,
                ansatz,
                1,
                method="parameter_shift",
            )

        np.testing.assert_allclose(gradient, np.array([0.5]))

    def test_parameter_shift_gradient_accepts_scalar_parameter(self):
        import rocq
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import VQE_Solver

        @rocq.kernel
        def ansatz(theta):
            q = rocq.qvec(1)
            rocq.rx(theta, q[0])

        solver = VQE_Solver()
        hamiltonian = PauliOperator("Z0")
        with mock.patch("rocquantum.solvers.vqe_solver.observe", side_effect=[1.0, 0.0]):
            gradient = solver.estimate_gradient(
                0.25,
                hamiltonian,
                ansatz,
                1,
                method="parameter_shift",
            )

        np.testing.assert_allclose(gradient, np.array([0.5]))

    def test_gradient_estimation_does_not_pollute_optimizer_trace(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import VQE_Solver

        def ansatz(theta):
            return None

        solver = VQE_Solver()
        solver._intermediate_results.append(
            {"parameters": np.array([0.1]), "energy": -0.25}
        )
        hamiltonian = PauliOperator("Z0")

        with mock.patch("rocquantum.solvers.vqe_solver.observe", side_effect=[1.0, 0.0]) as patched_observe:
            gradient = solver.estimate_gradient(
                np.array([0.25]),
                hamiltonian,
                ansatz,
                1,
                method="parameter_shift",
            )

        np.testing.assert_allclose(gradient, np.array([0.5]))
        self.assertEqual(len(solver._intermediate_results), 1)
        np.testing.assert_allclose(solver._intermediate_results[0]["parameters"], np.array([0.1]))
        self.assertEqual(solver._intermediate_results[0]["energy"], -0.25)
        self.assertEqual(patched_observe.call_count, 2)

    def test_vqe_rejects_nonfinite_parameters_before_backend_use(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import Optimizer, VQE_Solver

        class RecordingOptimizer(Optimizer):
            def __init__(self):
                self.called = False

            def minimize(self, fun, x0, args=()):
                self.called = True
                return types.SimpleNamespace(fun=0.0, x=np.asarray(x0, dtype=float))

        def ansatz(theta):
            return None

        solver = VQE_Solver()
        hamiltonian = PauliOperator("Z0")
        with mock.patch("rocquantum.solvers.vqe_solver.observe") as patched_observe:
            with self.assertRaisesRegex(ValueError, "parameters must be finite"):
                solver._objective_function(np.array([np.inf]), hamiltonian, ansatz, 1)
        patched_observe.assert_not_called()

        for params in ([True], ["0.25"], True):
            with self.subTest(params=params):
                with mock.patch("rocquantum.solvers.vqe_solver.observe") as patched_observe:
                    with self.assertRaisesRegex(ValueError, "parameters must be finite"):
                        solver._objective_function(params, hamiltonian, ansatz, 1)
                patched_observe.assert_not_called()

        optimizer = RecordingOptimizer()
        with self.assertRaisesRegex(ValueError, "initial_params must be finite"):
            VQE_Solver(optimizer=optimizer).solve(
                hamiltonian,
                ansatz,
                1,
                initial_params=[np.nan],
            )
        self.assertFalse(optimizer.called)

        for initial_params in ([True], ["0.25"]):
            optimizer = RecordingOptimizer()
            with self.subTest(initial_params=initial_params):
                with self.assertRaisesRegex(ValueError, "initial_params must be finite"):
                    VQE_Solver(optimizer=optimizer).solve(
                        hamiltonian,
                        ansatz,
                        1,
                        initial_params=initial_params,
                    )
                self.assertFalse(optimizer.called)

        with mock.patch("rocquantum.solvers.vqe_solver.observe") as patched_observe:
            with self.assertRaisesRegex(ValueError, "parameters must be finite"):
                solver.estimate_gradient(
                    np.array([np.nan]),
                    hamiltonian,
                    ansatz,
                    1,
                )
        patched_observe.assert_not_called()

        with mock.patch("rocquantum.solvers.vqe_solver.observe") as patched_observe:
            with self.assertRaisesRegex(ValueError, "finite_diff step"):
                solver.estimate_gradient(
                    np.array([0.25]),
                    hamiltonian,
                    ansatz,
                    1,
                    method="finite_diff",
                    step=0.0,
                )
        patched_observe.assert_not_called()

        with mock.patch("rocquantum.solvers.vqe_solver.observe") as patched_observe:
            with self.assertRaisesRegex(ValueError, "finite_diff step"):
                solver.estimate_gradient(
                    np.array([0.25]),
                    hamiltonian,
                    ansatz,
                    1,
                    method="finite_diff",
                    step=True,
                )
        patched_observe.assert_not_called()

        for method in (None, True, 3):
            with self.subTest(method=method):
                with mock.patch("rocquantum.solvers.vqe_solver.observe") as patched_observe:
                    with self.assertRaisesRegex(ValueError, "method must be"):
                        solver.estimate_gradient(
                            np.array([0.25]),
                            hamiltonian,
                            ansatz,
                            1,
                            method=method,
                        )
                patched_observe.assert_not_called()

        with mock.patch("rocquantum.solvers.vqe_solver.observe") as patched_observe:
            with self.assertRaisesRegex(ValueError, "num_qubits"):
                solver._objective_function(np.array([0.25]), hamiltonian, ansatz, 0)
        patched_observe.assert_not_called()

    def test_vqe_rejects_non_real_or_nonfinite_energy_results(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import Optimizer, VQE_Solver

        class MissingEnergyOptimizer(Optimizer):
            def minimize(self, fun, x0, args=()):
                return types.SimpleNamespace(x=np.asarray(x0, dtype=float))

        class MissingParameterOptimizer(Optimizer):
            def minimize(self, fun, x0, args=()):
                return types.SimpleNamespace(fun=-0.25)

        class BadEnergyOptimizer(Optimizer):
            def minimize(self, fun, x0, args=()):
                return types.SimpleNamespace(fun=np.nan, x=np.asarray(x0, dtype=float))

        class BadParameterOptimizer(Optimizer):
            def minimize(self, fun, x0, args=()):
                return types.SimpleNamespace(fun=-0.25, x=np.array([np.inf]))

        def ansatz(theta):
            return None

        solver = VQE_Solver()
        hamiltonian = PauliOperator("Z0")

        with mock.patch("rocquantum.solvers.vqe_solver.observe", return_value=1.0j):
            with self.assertRaisesRegex(ValueError, "observed energy must be real"):
                solver.evaluate_energy(hamiltonian, ansatz, 1, parameters=[0.25])

        with mock.patch("rocquantum.solvers.vqe_solver.observe", return_value=np.inf):
            with self.assertRaisesRegex(ValueError, "observed energy must be finite"):
                solver.evaluate_energy(hamiltonian, ansatz, 1, parameters=[0.25])

        with self.assertRaisesRegex(ValueError, "optimizer result must provide 'fun'"):
            VQE_Solver(optimizer=MissingEnergyOptimizer()).solve(
                hamiltonian,
                ansatz,
                1,
                initial_params=[0.25],
            )

        with self.assertRaisesRegex(ValueError, "optimizer result must provide 'x'"):
            VQE_Solver(optimizer=MissingParameterOptimizer()).solve(
                hamiltonian,
                ansatz,
                1,
                initial_params=[0.25],
            )

        with self.assertRaisesRegex(ValueError, "optimizer result energy must be finite"):
            VQE_Solver(optimizer=BadEnergyOptimizer()).solve(
                hamiltonian,
                ansatz,
                1,
                initial_params=[0.25],
            )

        with self.assertRaisesRegex(ValueError, "optimizer result parameters must be finite"):
            VQE_Solver(optimizer=BadParameterOptimizer()).solve(
                hamiltonian,
                ansatz,
                1,
                initial_params=[0.25],
            )

    def test_solve_normalizes_scalar_initial_parameter(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import Optimizer, VQE_Solver

        class RecordingOptimizer(Optimizer):
            def __init__(self):
                self.x0 = None

            def minimize(self, fun, x0, args=()):
                self.x0 = np.asarray(x0, dtype=float)
                return types.SimpleNamespace(fun=-0.25, x=self.x0)

        def ansatz(theta):
            return None

        optimizer = RecordingOptimizer()
        solver = VQE_Solver(optimizer=optimizer)
        result = solver.solve(PauliOperator("Z0"), ansatz, 1, initial_params=0.25)

        np.testing.assert_allclose(optimizer.x0, np.array([0.25]))
        np.testing.assert_allclose(result["optimal_parameters"], np.array([0.25]))

    def test_scipy_optimizer_validates_option_mapping(self):
        from rocquantum.solvers.vqe_solver import SciPyOptimizer

        source_options = {"method": "COBYLA", "tol": 1.0e-4}
        optimizer = SciPyOptimizer(options=source_options)
        source_options["method"] = "BFGS"

        self.assertEqual(optimizer.options, {"method": "COBYLA", "tol": 1.0e-4})

        with self.assertRaisesRegex(ValueError, "options must be a mapping"):
            SciPyOptimizer(options=[("method", "COBYLA")])

        with self.assertRaisesRegex(ValueError, "option keys must be strings"):
            SciPyOptimizer(options={1: "COBYLA"})

    def test_vqe_solver_rejects_invalid_optimizer_object(self):
        from rocquantum.solvers.vqe_solver import VQE_Solver

        invalid_optimizers = (
            object(),
            types.SimpleNamespace(minimize=None),
            "COBYLA",
        )
        for optimizer in invalid_optimizers:
            with self.subTest(optimizer=optimizer):
                with self.assertRaisesRegex(ValueError, "callable minimize"):
                    VQE_Solver(optimizer=optimizer)

    def test_vqe_solver_validates_backend_name_at_construction(self):
        from rocquantum.solvers.vqe_solver import VQE_Solver

        for backend in ("state_vector", "density_matrix", "stabilizer", "tableau", "clifford"):
            with self.subTest(backend=backend):
                self.assertEqual(VQE_Solver(backend=backend).backend, backend)

        invalid_backends = ("gpu", "", None, True)
        for backend in invalid_backends:
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "backend must be one of"):
                    VQE_Solver(backend=backend)

    def test_solve_is_quiet_by_default_with_verbose_opt_in(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import Optimizer, VQE_Solver

        class NoopOptimizer(Optimizer):
            def minimize(self, fun, x0, args=()):
                return types.SimpleNamespace(fun=-0.25, x=np.asarray(x0, dtype=float))

        def ansatz(theta):
            return None

        with mock.patch("builtins.print") as patched_print:
            VQE_Solver(optimizer=NoopOptimizer()).solve(
                PauliOperator("Z0"),
                ansatz,
                1,
                initial_params=[0.25],
            )
        patched_print.assert_not_called()

        with mock.patch("builtins.print") as patched_print:
            VQE_Solver(optimizer=NoopOptimizer(), verbose=True).solve(
                PauliOperator("Z0"),
                ansatz,
                1,
                initial_params=[0.25],
            )
        self.assertEqual(patched_print.call_count, 2)
        self.assertEqual(patched_print.call_args_list[0].args[0], "Starting VQE optimization...")
        self.assertEqual(patched_print.call_args_list[1].args[0], "VQE optimization finished.")

    def test_objective_passes_multi_parameter_vector_to_qaoa_kernel(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.qaoa import make_maxcut_qaoa_kernel
        from rocquantum.solvers.vqe_solver import VQE_Solver

        ansatz = make_maxcut_qaoa_kernel(2, [(0, 1)], layers=1)
        solver = VQE_Solver(backend="state_vector")
        hamiltonian = PauliOperator("Z0 Z1")

        with mock.patch("rocquantum.solvers.vqe_solver.observe", return_value=-0.75) as patched_observe:
            energy = solver._objective_function(np.array([0.3, 0.4]), hamiltonian, ansatz, 2)

        self.assertEqual(energy, -0.75)
        args, kwargs = patched_observe.call_args
        self.assertIs(args[0], ansatz)
        self.assertIs(args[1], hamiltonian)
        np.testing.assert_allclose(args[2], np.array([0.3, 0.4]))
        self.assertEqual(kwargs, {"backend": "state_vector"})

    def test_objective_passes_single_value_vector_to_vector_style_kernel(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import VQE_Solver

        def ansatz(parameters):
            return None

        solver = VQE_Solver(backend="state_vector")
        hamiltonian = PauliOperator("Z0")

        with mock.patch("rocquantum.solvers.vqe_solver.observe", return_value=-0.25) as patched_observe:
            energy = solver._objective_function(np.array([0.125]), hamiltonian, ansatz, 1)

        self.assertEqual(energy, -0.25)
        args, kwargs = patched_observe.call_args
        self.assertIs(args[0], ansatz)
        self.assertIs(args[1], hamiltonian)
        np.testing.assert_allclose(args[2], np.array([0.125]))
        self.assertEqual(kwargs, {"backend": "state_vector"})

    def test_objective_preserves_single_scalar_parameter_kernel(self):
        from rocq.operator import PauliOperator
        from rocquantum.solvers.vqe_solver import VQE_Solver

        def ansatz(theta):
            return None

        solver = VQE_Solver(backend="state_vector")
        hamiltonian = PauliOperator("Z0")

        with mock.patch("rocquantum.solvers.vqe_solver.observe", return_value=-0.25) as patched_observe:
            energy = solver._objective_function(np.array([0.125]), hamiltonian, ansatz, 1)

        self.assertEqual(energy, -0.25)
        patched_observe.assert_called_once_with(ansatz, hamiltonian, 0.125, backend="state_vector")


class TestQaoaHelpers(unittest.TestCase):
    def test_spin_factories_match_cudaq_style_pauli_construction(self):
        import rocq
        from rocq.operator import iter_pauli_terms

        operator = 0.5 - 0.5 * rocq.spin.z(0) * rocq.spin.z(1)

        self.assertEqual(
            iter_pauli_terms(operator),
            [
                (0.5 + 0j, []),
                (-0.5 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )

    def test_spin_factories_preserve_same_qubit_pauli_phases(self):
        import rocq
        from rocq.operator import iter_pauli_terms

        self.assertEqual(
            iter_pauli_terms(rocq.spin.x(0) * rocq.spin.y(0)),
            [(1j, [("Z", 0)])],
        )
        self.assertEqual(
            iter_pauli_terms(rocq.spin.y(0) * rocq.spin.x(0)),
            [(-1j, [("Z", 0)])],
        )

    def test_observable_arithmetic_accepts_numeric_identity_constants(self):
        from rocq.operator import PauliOperator, iter_pauli_terms

        operator = 0.5 - 0.5 * PauliOperator("Z0 Z1")

        self.assertEqual(
            iter_pauli_terms(operator),
            [
                (0.5 + 0j, []),
                (-0.5 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )

    def test_observable_arithmetic_accepts_scalar_division(self):
        import rocq
        from rocq.operator import iter_pauli_terms

        operator = (rocq.spin.i(0) - rocq.spin.z(0) * rocq.spin.z(1)) / 2

        self.assertEqual(
            iter_pauli_terms(operator),
            [
                (0.5 + 0j, []),
                (-0.5 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )

    def test_observable_arithmetic_accepts_chained_pauli_products(self):
        from rocq.operator import PauliOperator, iter_pauli_terms

        operator = 0.5 * PauliOperator("Z0") * PauliOperator("Z1")

        self.assertEqual(
            iter_pauli_terms(operator),
            [
                (0.5 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )

    def test_observable_arithmetic_distributes_sum_products(self):
        from rocq.operator import PauliOperator, iter_pauli_terms

        operator = (0.5 + PauliOperator("Z0")) * PauliOperator("Z1")

        self.assertEqual(
            iter_pauli_terms(operator),
            [
                (0.5 + 0j, [("Z", 1)]),
                (1 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )

    def test_observable_arithmetic_rejects_unsupported_operands(self):
        from rocq.operator import PauliOperator

        class UnsupportedOperand:
            def __add__(self, other):
                return NotImplemented

            def __sub__(self, other):
                return NotImplemented

        operator = PauliOperator("Z0")
        summed = operator + PauliOperator("Z1")
        unsupported = UnsupportedOperand()

        unsupported_cases = (
            lambda: operator * unsupported,
            lambda: operator / unsupported,
            lambda: operator + unsupported,
            lambda: unsupported + operator,
            lambda: operator - unsupported,
            lambda: unsupported - operator,
            lambda: summed + unsupported,
        )

        for expression in unsupported_cases:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(TypeError, "Cannot"):
                    expression()

    def test_maxcut_qaoa_kernel_emits_supported_ops(self):
        from rocquantum.solvers.qaoa import make_maxcut_qaoa_kernel, maxcut_cost_operator
        from rocq.operator import iter_pauli_terms

        kernel = make_maxcut_qaoa_kernel(2, [(0, 1, 2.0)], layers=1)
        mlir = kernel.mlir(np.array([0.3, 0.4]))

        self.assertIn('"quantum.h"', mlir)
        self.assertIn('"quantum.cnot"', mlir)
        self.assertIn('"quantum.rz"', mlir)
        self.assertIn('"quantum.rx"', mlir)

        ops = kernel.build(np.array([0.3, 0.4])).ops
        rz_ops = [op for op in ops if op.name == "rz"]
        rx_ops = [op for op in ops if op.name == "rx"]
        self.assertEqual(len(rz_ops), 1)
        self.assertEqual(len(rx_ops), 2)
        self.assertEqual(rz_ops[0].params["phi"], -0.6)
        self.assertTrue(all(op.params["theta"] == 0.8 for op in rx_ops))

        operator = maxcut_cost_operator(2, [(0, 1, 2.0)])
        self.assertIn("Z0 Z1", operator.to_string())
        self.assertEqual(
            iter_pauli_terms(operator),
            [
                (1 + 0j, []),
                (-1 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )

    def test_maxcut_qaoa_combines_duplicate_undirected_edges(self):
        from rocquantum.solvers.qaoa import make_maxcut_qaoa_kernel, maxcut_cost_operator
        from rocq.operator import iter_pauli_terms

        kernel = make_maxcut_qaoa_kernel(2, [(0, 1, 1.0), (1, 0, 2.0)], layers=1)
        ops = kernel.build(np.array([0.3, 0.4])).ops
        rz_ops = [op for op in ops if op.name == "rz"]

        self.assertEqual(len(rz_ops), 1)
        self.assertAlmostEqual(rz_ops[0].params["phi"], -0.9)

        operator = maxcut_cost_operator(2, [(0, 1, 1.0), (1, 0, 2.0)])
        self.assertEqual(
            iter_pauli_terms(operator),
            [
                (1.5 + 0j, []),
                (-1.5 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )

    def test_maxcut_qaoa_validates_problem_inputs(self):
        from rocquantum.solvers.qaoa import make_maxcut_qaoa_kernel, maxcut_cost_operator, solve_maxcut_qaoa

        with self.assertRaisesRegex(ValueError, "num_qubits must be a positive integer"):
            make_maxcut_qaoa_kernel(2.5, [(0, 1)])
        with self.assertRaisesRegex(ValueError, "layers must be a positive integer"):
            make_maxcut_qaoa_kernel(2, [(0, 1)], layers=1.5)
        with self.assertRaisesRegex(ValueError, "QAOA edges must be an iterable"):
            maxcut_cost_operator(2, None)
        with self.assertRaisesRegex(ValueError, "QAOA edges must be an iterable"):
            solve_maxcut_qaoa(2, "01")
        with self.assertRaisesRegex(ValueError, "QAOA edges must be"):
            maxcut_cost_operator(2, [None])
        with self.assertRaisesRegex(ValueError, "QAOA edges must be"):
            maxcut_cost_operator(2, [(0, 1, 2.0, 3.0)])
        with self.assertRaisesRegex(ValueError, "integer qubit index"):
            maxcut_cost_operator(2, [(0.5, 1)])
        with self.assertRaisesRegex(ValueError, "integer qubit index"):
            maxcut_cost_operator(2, [(False, 1)])
        with self.assertRaisesRegex(ValueError, "finite"):
            maxcut_cost_operator(2, [(0, 1, np.nan)])
        with self.assertRaisesRegex(ValueError, "finite"):
            maxcut_cost_operator(2, [(0, 1, True)])
        with self.assertRaisesRegex(ValueError, "finite"):
            maxcut_cost_operator(2, [(0, 1, "1.0")])
        with self.assertRaisesRegex(ValueError, "distinct valid"):
            maxcut_cost_operator(2, [(1, 1)])
        with self.assertRaisesRegex(ValueError, "distinct valid"):
            solve_maxcut_qaoa(2, [(0, 2)])
        with self.assertRaisesRegex(ValueError, "initial_params must be finite"):
            solve_maxcut_qaoa(2, [(0, 1)], initial_params=[np.nan, 0.1])
        with self.assertRaisesRegex(ValueError, "initial_params must be finite"):
            solve_maxcut_qaoa(2, [(0, 1)], initial_params=[True, 0.1])
        with self.assertRaisesRegex(ValueError, "initial_params must be finite"):
            solve_maxcut_qaoa(2, [(0, 1)], initial_params=["0.1", 0.2])

        kernel = make_maxcut_qaoa_kernel(2, [(0, 1)], layers=1)
        with self.assertRaisesRegex(ValueError, "QAOA must be finite"):
            kernel.build(np.array([np.inf, 0.1]))
        with self.assertRaisesRegex(ValueError, "QAOA must be finite"):
            kernel.build(np.array([True, 0.1], dtype=object))

    def test_solve_maxcut_qaoa_wraps_vqe_solver(self):
        from rocq.operator import iter_pauli_terms
        from rocquantum.solvers import solve_maxcut_qaoa
        from rocquantum.solvers.vqe_solver import Optimizer

        class RecordingOptimizer(Optimizer):
            def __init__(self):
                self.x0 = None
                self.args = None

            def minimize(self, fun, x0, args=()):
                self.x0 = np.asarray(x0, dtype=float).copy()
                self.args = args
                return types.SimpleNamespace(fun=-1.25, x=self.x0 + 0.1)

        optimizer = RecordingOptimizer()
        edges = ((u, v, weight) for u, v, weight in [(0, 1, 1.0), (1, 0, 2.0)])
        result = solve_maxcut_qaoa(
            2,
            edges,
            layers=1,
            initial_params=[0.3, 0.4],
            optimizer=optimizer,
            backend="density_matrix",
        )

        np.testing.assert_allclose(optimizer.x0, np.array([0.3, 0.4]))
        np.testing.assert_allclose(result["optimal_parameters"], np.array([0.4, 0.5]))
        self.assertEqual(result["optimal_energy"], -1.25)
        self.assertEqual(result["optimal_cut_value"], 1.25)
        self.assertEqual(result["normalized_edges"], [(0, 1, 3.0)])
        self.assertEqual(result["layers"], 1)
        self.assertEqual(result["num_qubits"], 2)
        self.assertEqual(result["backend"], "density_matrix")
        self.assertIs(optimizer.args[0], result["optimization_operator"])
        self.assertIs(optimizer.args[1], result["ansatz"])
        self.assertEqual(optimizer.args[2], 2)
        self.assertEqual(result["optimization_direction"], "maximize_cut_value")
        self.assertEqual(
            iter_pauli_terms(result["cost_operator"]),
            [
                (1.5 + 0j, []),
                (-1.5 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )
        self.assertEqual(
            iter_pauli_terms(result["optimization_operator"]),
            [
                (-1.5 + 0j, []),
                (1.5 + 0j, [("Z", 0), ("Z", 1)]),
            ],
        )

    def test_solve_maxcut_qaoa_validates_initial_parameter_count(self):
        from rocquantum.solvers.qaoa import solve_maxcut_qaoa

        with self.assertRaisesRegex(ValueError, "initial_params"):
            solve_maxcut_qaoa(2, [(0, 1)], layers=2, initial_params=[0.1])


class TestQecHelpers(unittest.TestCase):
    def test_qec_capabilities_expose_experimental_cudaqx_boundary(self):
        import rocquantum.qec as qec
        from rocquantum.qec import capabilities, qec_capabilities

        capability_data = qec_capabilities()

        self.assertEqual(capability_data["status"], "experimental_partial")
        self.assertEqual(capabilities(), capability_data)
        self.assertIn("QEC_Experiment.run_single_round", capability_data["entry_points"])
        self.assertIn(
            "three-qubit bit-flip repetition-code single-round sampling",
            capability_data["supported_features"],
        )
        self.assertIn(
            "positive-integer shot/round/num_qubits, ancilla-index, initial-state callable, syndrome, and bool-safe count/bit validation",
            capability_data["supported_features"],
        )
        self.assertIn(
            "mid-circuit measurement and classical feedback",
            capability_data["unsupported_features"],
        )
        self.assertEqual(
            capability_data["supported_code_family"],
            "three-qubit bit-flip repetition code",
        )
        self.assertIn("CUDA-QX", capability_data["comparison_target"])
        self.assertIn("self-hosted ROCm CI", capability_data["performance_note"])
        self.assertIn("qec_capabilities", qec.__all__)

    def test_qec_experiment_samples_canonical_fragments(self):
        from rocq.operator import PauliOperator
        from rocquantum.qec.framework import Decoder, QEC_Experiment, QuantumErrorCode

        class FragmentCode(QuantumErrorCode):
            def __init__(self):
                self.generate_args = None

            def generate_stabilizer_circuits(
                self,
                initial_state_kernel,
                num_qubits,
                backend="state_vector",
            ):
                self.generate_args = (initial_state_kernel, num_qubits, backend)
                return ["fragment-0", "fragment-1"]

            def define_logical_operators(self):
                return {"logical_Z": PauliOperator("Z0")}

        class RecordingDecoder(Decoder):
            def __init__(self):
                self.syndrome = None

            def decode(self, syndrome):
                self.syndrome = list(syndrome)
                return PauliOperator("X0")

        code = FragmentCode()
        decoder = RecordingDecoder()
        experiment = QEC_Experiment(backend="state_vector")

        with mock.patch("builtins.print") as patched_print, mock.patch(
            "rocquantum.qec.framework.rocq.sample",
            side_effect=[{"1": 3, "0": 1}, {"0": 4}],
        ) as patched_sample:
            result = experiment.run_single_round(
                code,
                decoder,
                initial_state_kernel=None,
                num_qubits=5,
                ancilla_qubit_indices=[3, 4],
                shots=4,
            )

        patched_print.assert_not_called()
        self.assertEqual(code.generate_args, (None, 5, "state_vector"))
        self.assertEqual(decoder.syndrome, [1, 0])
        self.assertEqual(result["syndrome"], [1, 0])
        self.assertIn("X0", result["correction_applied"])
        self.assertEqual(result["shots"], 4)
        self.assertEqual(patched_sample.call_count, 2)
        _, args, kwargs = patched_sample.mock_calls[0]
        self.assertEqual(args, ("fragment-0", 4))
        self.assertEqual(kwargs, {"backend": "state_vector", "qubits": [3]})
        _, args, kwargs = patched_sample.mock_calls[1]
        self.assertEqual(args, ("fragment-1", 4))
        self.assertEqual(kwargs, {"backend": "state_vector", "qubits": [4]})

    def test_repetition_code_generator_validates_num_qubits(self):
        from rocquantum.qec.codes.repetition_code import ThreeQubitRepetitionCode

        code = ThreeQubitRepetitionCode()
        circuits = code.generate_stabilizer_circuits(None, np.int64(5))

        self.assertEqual(len(circuits), 2)
        for num_qubits in (True, 5.0, "5"):
            with self.subTest(num_qubits=num_qubits):
                with self.assertRaisesRegex(ValueError, "num_qubits must be a positive integer"):
                    code.generate_stabilizer_circuits(None, num_qubits)
        with self.assertRaisesRegex(ValueError, "at least 5 qubits"):
            code.generate_stabilizer_circuits(None, 4)
        for initial_state_kernel in ("prepare", object(), True):
            with self.subTest(initial_state_kernel=initial_state_kernel):
                with self.assertRaisesRegex(ValueError, "initial_state_kernel must be callable"):
                    code.generate_stabilizer_circuits(initial_state_kernel, 5)

    def test_qec_experiment_validates_ancilla_sample_counts(self):
        from rocquantum.qec.framework import _most_likely_single_bit

        self.assertEqual(_most_likely_single_bit({"0": 1, "1": 2}), 1)
        with self.assertRaisesRegex(ValueError, "No ancilla samples"):
            _most_likely_single_bit({})
        with self.assertRaisesRegex(ValueError, "non-empty binary"):
            _most_likely_single_bit({"": 1})
        with self.assertRaisesRegex(ValueError, "non-empty binary"):
            _most_likely_single_bit({"2": 1})
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            _most_likely_single_bit({"0": 1, "1": -1})
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            _most_likely_single_bit({"0": True})
        with self.assertRaisesRegex(ValueError, "at least one shot"):
            _most_likely_single_bit({"0": 0, "1": 0})

    def test_qec_execution_helpers_require_integer_shots_and_rounds(self):
        from rocquantum.qec.framework import (
            QEC_Experiment,
            _validate_positive_integer,
            run_repetition_code_rounds,
            run_repetition_code_single_round,
        )

        self.assertEqual(_validate_positive_integer(np.int64(2), "shots"), 2)

        experiment = QEC_Experiment()
        with self.assertRaisesRegex(ValueError, "shots must be a positive integer"):
            experiment.run_single_round(None, None, None, 0, [], shots=1.5)
        with self.assertRaisesRegex(ValueError, "num_qubits must be a positive integer"):
            experiment.run_single_round(None, None, None, True, [], shots=1)
        with self.assertRaisesRegex(ValueError, "ancilla_qubit_indices"):
            experiment.run_single_round(None, None, None, 2, True, shots=1)
        with self.assertRaisesRegex(ValueError, "integer qubit indices"):
            experiment.run_single_round(None, None, None, 2, [0.5], shots=1)
        with self.assertRaisesRegex(ValueError, "range for num_qubits"):
            experiment.run_single_round(None, None, None, 2, [2], shots=1)
        with self.assertRaisesRegex(ValueError, "range for num_qubits"):
            experiment.run_single_round(None, None, None, 2, [-1], shots=1)
        with self.assertRaisesRegex(ValueError, "shots must be a positive integer"):
            run_repetition_code_single_round(shots=True)
        with self.assertRaisesRegex(ValueError, "rounds must be a positive integer"):
            run_repetition_code_rounds(rounds=1.5)
        with self.assertRaisesRegex(ValueError, "shots must be positive"):
            run_repetition_code_rounds(shots=0)

    def test_repetition_code_counts_analysis_reports_success_rate(self):
        from rocquantum.qec import analyze_repetition_code_counts

        analysis = analyze_repetition_code_counts(
            {"11": 7, "00": 1},
            initial_bits=[0, 0, 0],
            error_qubit=1,
        )

        self.assertEqual(
            analysis["syndrome_histogram"],
            {"00": 1, "10": 0, "11": 7, "01": 0},
        )
        self.assertEqual(analysis["most_likely_syndrome"], [1, 1])
        self.assertEqual(analysis["most_likely_correction_qubit"], 1)
        self.assertEqual(analysis["most_likely_corrected_data_bits"], [0, 0, 0])
        self.assertAlmostEqual(analysis["logical_success_rate"], 7 / 8)

        numpy_integer_analysis = analyze_repetition_code_counts(
            {"11": np.int64(1)},
            initial_bits=[np.int64(0), np.int64(0), np.int64(0)],
            error_qubit=np.int64(1),
            expected_logical_bit=np.int64(0),
        )
        self.assertEqual(numpy_integer_analysis["initial_data_bits"], [0, 0, 0])
        self.assertEqual(numpy_integer_analysis["observed_data_bits"], [0, 1, 0])

        one_bit_key_analysis = analyze_repetition_code_counts({"1": 2})
        self.assertEqual(
            one_bit_key_analysis["syndrome_histogram"],
            {"00": 0, "10": 2, "11": 0, "01": 0},
        )

    def test_repetition_code_measurement_error_mitigation_scores_syndromes(self):
        from rocquantum.qec import (
            analyze_repetition_code_counts,
            mitigate_repetition_syndrome_counts,
        )

        counts = {"01": 81, "00": 9, "11": 9, "10": 1}
        scores = mitigate_repetition_syndrome_counts(
            counts,
            measurement_error_probability=0.1,
        )

        self.assertAlmostEqual(scores["10"], 100.0)
        self.assertAlmostEqual(scores["00"], 0.0)
        self.assertAlmostEqual(scores["11"], 0.0)
        self.assertAlmostEqual(scores["01"], 0.0)

        analysis = analyze_repetition_code_counts(
            counts,
            initial_bits=[0, 0, 0],
            error_qubit=0,
            measurement_error_probability=0.1,
        )
        self.assertEqual(analysis["most_likely_syndrome"], [1, 0])
        self.assertEqual(analysis["most_likely_correction_qubit"], 0)
        self.assertEqual(analysis["most_likely_syndrome_source"], "measurement_mitigated")
        self.assertEqual(analysis["measurement_error_probability"], 0.1)
        self.assertAlmostEqual(analysis["mitigated_syndrome_scores"]["10"], 100.0)

    def test_repetition_code_counts_analysis_validates_inputs(self):
        from rocquantum.qec import (
            analyze_repetition_code_counts,
            mitigate_repetition_syndrome_counts,
        )

        with self.assertRaisesRegex(ValueError, "binary strings"):
            analyze_repetition_code_counts({"2": 1})
        with self.assertRaisesRegex(ValueError, "non-empty binary strings"):
            analyze_repetition_code_counts({"": 1})
        with self.assertRaisesRegex(ValueError, "at most two bits"):
            analyze_repetition_code_counts({"101": 1})
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            analyze_repetition_code_counts({"00": -1})
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            analyze_repetition_code_counts({"00": True})
        with self.assertRaisesRegex(ValueError, "initial_bits"):
            analyze_repetition_code_counts({"00": 1}, initial_bits=[True, 0, 0])
        with self.assertRaisesRegex(ValueError, "initial_bits"):
            analyze_repetition_code_counts({"00": 1}, initial_bits=True)
        with self.assertRaisesRegex(ValueError, "error_qubit"):
            analyze_repetition_code_counts({"00": 1}, error_qubit=True)
        with self.assertRaisesRegex(ValueError, "expected_logical_bit"):
            analyze_repetition_code_counts({"00": 1}, expected_logical_bit=True)
        with self.assertRaisesRegex(ValueError, "measurement_error_probability"):
            analyze_repetition_code_counts({"00": 1}, measurement_error_probability=True)
        with self.assertRaisesRegex(ValueError, "measurement_error_probability"):
            mitigate_repetition_syndrome_counts({"00": 1}, measurement_error_probability=0.5)

    def test_repetition_code_rounds_analysis_tracks_feed_forward(self):
        from rocquantum.qec import analyze_repetition_code_rounds

        analysis = analyze_repetition_code_rounds(
            [{"01": 5}, {"11": 3, "00": 1}],
            initial_bits=[0, 0, 0],
            error_qubits=[0, 1],
        )

        self.assertEqual(analysis["rounds"], 2)
        self.assertEqual(
            analysis["aggregate_syndrome_histogram"],
            {"00": 1, "10": 5, "11": 3, "01": 0},
        )
        self.assertEqual(analysis["correction_summary"], {"none": 0, "q0": 1, "q1": 1, "q2": 0})
        self.assertEqual(analysis["round_results"][0]["syndrome"], [1, 0])
        self.assertEqual(analysis["round_results"][1]["correction_qubit"], 1)
        self.assertEqual(analysis["final_data_bits"], [0, 0, 0])
        self.assertAlmostEqual(analysis["logical_success_rate"], 8 / 9)

        drifted = analyze_repetition_code_rounds(
            [{"00": 1}, {"00": 1}],
            initial_bits=[0, 0, 0],
            error_qubits=[1, None],
        )
        self.assertEqual(drifted["expected_logical_bit"], 0)
        self.assertEqual(drifted["round_results"][1]["analysis"]["expected_logical_bit"], 0)
        self.assertEqual(drifted["logical_success_rate"], 0.0)

        mitigated = analyze_repetition_code_rounds(
            [{"01": 81, "00": 9, "11": 9, "10": 1}],
            initial_bits=[0, 0, 0],
            error_qubits=[0],
            measurement_error_probability=0.1,
        )
        self.assertEqual(mitigated["measurement_error_probability"], 0.1)
        self.assertEqual(
            mitigated["round_results"][0]["analysis"]["most_likely_syndrome_source"],
            "measurement_mitigated",
        )
        self.assertEqual(mitigated["round_results"][0]["correction_qubit"], 0)

    def test_repetition_code_rounds_analysis_validates_schedule(self):
        from rocquantum.qec import analyze_repetition_code_rounds

        with self.assertRaisesRegex(ValueError, "at least one round"):
            analyze_repetition_code_rounds([])
        with self.assertRaisesRegex(ValueError, "length must match"):
            analyze_repetition_code_rounds([{"00": 1}], error_qubits=[0, 1])
        with self.assertRaisesRegex(ValueError, "count dictionaries"):
            analyze_repetition_code_rounds([[]])
        with self.assertRaisesRegex(ValueError, "error_qubit"):
            analyze_repetition_code_rounds([{"00": 1}], error_qubits=[True])

    def test_repetition_code_single_round_uses_canonical_sample(self):
        from rocquantum.qec.framework import run_repetition_code_single_round

        with mock.patch("rocquantum.qec.framework.rocq.sample", return_value={"01": 5}) as patched_sample:
            result = run_repetition_code_single_round(error_qubit=0, shots=5)

        self.assertEqual(result["syndrome"], [1, 0])
        self.assertIn("X0", result["correction_applied"])
        self.assertEqual(result["analysis"]["syndrome_histogram"], {"00": 0, "10": 5, "11": 0, "01": 0})
        self.assertEqual(result["most_likely_corrected_data_bits"], [0, 0, 0])
        self.assertEqual(result["logical_success_rate"], 1.0)
        patched_sample.assert_called_once()
        _, args, kwargs = patched_sample.mock_calls[0]
        self.assertEqual(kwargs["qubits"], [3, 4])
        self.assertEqual(args[1], 5)

    def test_repetition_code_repeated_rounds_use_canonical_sample(self):
        from rocquantum.qec.framework import run_repetition_code_rounds

        with mock.patch(
            "rocquantum.qec.framework.rocq.sample",
            side_effect=[{"01": 2}, {"11": 2}],
        ) as patched_sample:
            result = run_repetition_code_rounds(error_qubits=[0, 1], rounds=2, shots=2)

        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["aggregate_syndrome_histogram"], {"00": 0, "10": 2, "11": 2, "01": 0})
        self.assertEqual(result["correction_summary"], {"none": 0, "q0": 1, "q1": 1, "q2": 0})
        self.assertEqual(result["final_data_bits"], [0, 0, 0])
        self.assertEqual(result["shots_per_round"], 2)
        self.assertEqual(result["logical_success_rate"], 1.0)
        self.assertEqual(patched_sample.call_count, 2)
        for sample_call in patched_sample.mock_calls:
            _, args, kwargs = sample_call
            self.assertEqual(kwargs["qubits"], [3, 4])
            self.assertEqual(args[1], 2)

    def test_repetition_decoder_uses_canonical_pauli_operator(self):
        from rocq.operator import PauliOperator
        from rocquantum.qec.decoders.repetition_decoder import RepetitionCodeDecoder

        decoder = RepetitionCodeDecoder()
        correction = decoder.decode([0, 1])
        self.assertIsInstance(correction, PauliOperator)
        self.assertIn("X2", correction.to_string())

        invalid_syndromes = ([0], [0, 1, 0], [0, 2], [True, 0], "01", None)
        for syndrome in invalid_syndromes:
            with self.subTest(syndrome=syndrome):
                with self.assertRaisesRegex(ValueError, "syndrome"):
                    decoder.decode(syndrome)


if __name__ == "__main__":
    unittest.main()
