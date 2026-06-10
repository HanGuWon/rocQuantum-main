"""Contracts for the experimental CUDA-QX-style helper layer."""

from __future__ import annotations

import os
import sys
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


class TestQaoaHelpers(unittest.TestCase):
    def test_maxcut_qaoa_kernel_emits_supported_ops(self):
        from rocquantum.solvers.qaoa import make_maxcut_qaoa_kernel, maxcut_cost_operator

        kernel = make_maxcut_qaoa_kernel(2, [(0, 1, 2.0)], layers=1)
        mlir = kernel.mlir(np.array([0.3, 0.4]))

        self.assertIn('"quantum.h"', mlir)
        self.assertIn('"quantum.cnot"', mlir)
        self.assertIn('"quantum.rz"', mlir)
        self.assertIn('"quantum.rx"', mlir)
        self.assertIn("1.2", mlir)

        operator = maxcut_cost_operator(2, [(0, 1, 2.0)])
        self.assertIn("Z0 Z1", operator.to_string())


class TestQecHelpers(unittest.TestCase):
    def test_repetition_code_single_round_uses_canonical_sample(self):
        from rocquantum.qec.framework import run_repetition_code_single_round

        with mock.patch("rocquantum.qec.framework.rocq.sample", return_value={"01": 5}) as patched_sample:
            result = run_repetition_code_single_round(error_qubit=0, shots=5)

        self.assertEqual(result["syndrome"], [1, 0])
        self.assertIn("X0", result["correction_applied"])
        patched_sample.assert_called_once()
        _, args, kwargs = patched_sample.mock_calls[0]
        self.assertEqual(kwargs["qubits"], [3, 4])
        self.assertEqual(args[1], 5)

    def test_repetition_decoder_uses_canonical_pauli_operator(self):
        from rocq.operator import PauliOperator
        from rocquantum.qec.decoders.repetition_decoder import RepetitionCodeDecoder

        correction = RepetitionCodeDecoder().decode([0, 1])
        self.assertIsInstance(correction, PauliOperator)
        self.assertIn("X2", correction.to_string())


if __name__ == "__main__":
    unittest.main()
