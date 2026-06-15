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

    def test_repetition_code_counts_analysis_validates_inputs(self):
        from rocquantum.qec import analyze_repetition_code_counts

        with self.assertRaisesRegex(ValueError, "binary strings"):
            analyze_repetition_code_counts({"2": 1})
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            analyze_repetition_code_counts({"00": -1})

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

    def test_repetition_code_rounds_analysis_validates_schedule(self):
        from rocquantum.qec import analyze_repetition_code_rounds

        with self.assertRaisesRegex(ValueError, "at least one round"):
            analyze_repetition_code_rounds([])
        with self.assertRaisesRegex(ValueError, "length must match"):
            analyze_repetition_code_rounds([{"00": 1}], error_qubits=[0, 1])
        with self.assertRaisesRegex(ValueError, "count dictionaries"):
            analyze_repetition_code_rounds([[]])

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

        correction = RepetitionCodeDecoder().decode([0, 1])
        self.assertIsInstance(correction, PauliOperator)
        self.assertIn("X2", correction.to_string())


if __name__ == "__main__":
    unittest.main()
