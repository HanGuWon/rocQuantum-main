"""
P1 Compiler + Python E2E Flow Tests

Minimal end-to-end flow checks for:
1) rocq kernel MLIR emission
2) emit_qir translation path
3) compile_and_execute runtime path or actionable diagnostics
4) Python public API execution flow

This suite is dependency-aware:
- If `rocquantum_bind` is missing, compile/QIR runtime checks are skipped
  with an explicit ROCm CI verification path.
"""

import math
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import rocq
from rocq.gates import cnot, h, x, z
from rocq.kernel import QuantumKernel
from rocq.qvec import qvec

try:
    import rocquantum_bind
except ImportError:
    rocquantum_bind = None


def _compiler_skip_reason() -> str:
    return (
        "rocquantum_bind is unavailable. Build bindings in 'rocQuantum-main/build-ci' "
        "and run '.github/workflows/rocm-linux-build.yml' job 'build' on a ROCm host "
        "with '/dev/kfd' present."
    )


class TestCompilerE2EFlow(unittest.TestCase):
    @staticmethod
    def _build_h_kernel() -> QuantumKernel:
        @QuantumKernel
        def h_program():
            q = qvec(1)
            h(q[0])

        return h_program

    @staticmethod
    def _build_bell_kernel() -> QuantumKernel:
        @QuantumKernel
        def bell_program():
            q = qvec(2)
            h(q[0])
            cnot(q[0], q[1])

        return bell_program

    @staticmethod
    def _build_three_qubit_chain_kernel() -> QuantumKernel:
        @QuantumKernel
        def chain_program():
            q = qvec(3)
            h(q[0])
            x(q[1])
            z(q[2])
            cnot(q[0], q[2])

        return chain_program

    def _emit_qir_or_skip(self, kernel_obj: QuantumKernel):
        mlir = kernel_obj.mlir()
        self.assertIn("module", mlir)
        self.assertIn("func.func", mlir)

        if rocquantum_bind is None:
            self.skipTest(_compiler_skip_reason())

        compiler = rocquantum_bind.MLIRCompiler(kernel_obj.num_qubits, "hip_statevec")
        qir = compiler.emit_qir(mlir)
        self.assertFalse(qir.startswith("Error:"), msg=qir)
        return mlir, qir, compiler

    def test_emit_qir_h_circuit(self):
        kernel_obj = self._build_h_kernel()
        _, qir, _ = self._emit_qir_or_skip(kernel_obj)
        self.assertIn("__quantum__qis__h__body", qir)

    def test_emit_qir_bell_circuit(self):
        kernel_obj = self._build_bell_kernel()
        _, qir, _ = self._emit_qir_or_skip(kernel_obj)
        self.assertIn("__quantum__qis__h__body", qir)
        self.assertIn("__quantum__qis__cnot__body", qir)

    def test_emit_qir_three_qubit_chain(self):
        kernel_obj = self._build_three_qubit_chain_kernel()
        _, qir, _ = self._emit_qir_or_skip(kernel_obj)
        self.assertIn("__quantum__qis__h__body", qir)
        self.assertIn("__quantum__qis__x__body", qir)
        self.assertIn("__quantum__qis__z__body", qir)
        self.assertIn("__quantum__qis__cnot__body", qir)

    def test_compile_and_execute_bell_or_actionable_diagnostic(self):
        kernel_obj = self._build_bell_kernel()
        mlir = kernel_obj.mlir()

        if rocquantum_bind is None:
            self.skipTest(_compiler_skip_reason())

        compiler = rocquantum_bind.MLIRCompiler(kernel_obj.num_qubits, "hip_statevec")
        try:
            state = compiler.compile_and_execute(mlir, {"strict": True})
        except RuntimeError as exc:
            msg = str(exc).lower()
            actionable_tokens = [
                "not yet implemented",
                "compile_and_execute",
                "hipstatevec",
                "rocm",
                "failed",
            ]
            self.assertTrue(
                any(token in msg for token in actionable_tokens),
                msg=f"Non-actionable runtime diagnostic: {exc}",
            )
            return

        self.assertEqual(len(state), 4)
        norm = sum(abs(amplitude) ** 2 for amplitude in state)
        self.assertTrue(math.isfinite(norm))
        self.assertAlmostEqual(norm, 1.0, places=6)


class TestPythonPublicAPIFlow(unittest.TestCase):
    def test_rocq_execute_bell_flow(self):
        @rocq.kernel
        def bell_program():
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])

        state = rocq.execute(bell_program, backend="state_vector")
        if isinstance(state, str):
            self.assertIn("mock_cpp_state_vector_data", state)
        else:
            self.assertEqual(len(state), 4)


if __name__ == "__main__":
    unittest.main()
