"""
P1 Compiler Pipeline Tests — rocQuantum Stabilization

Tests for MLIR emission and namespace consistency.
No C++ build required; pure-Python validation.

    python -m unittest tests.test_p1_compiler -v
"""

import os
import re
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ===================================================================
# 1. Namespace consistency in .td and .cpp files
# ===================================================================
class TestNamespaceConsistency(unittest.TestCase):
    """Compiler dialect files must use the canonical rocq::mlir:: namespace."""

    def _read(self, *parts):
        path = os.path.join(_PROJECT_ROOT, *parts)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_quantum_ops_td_namespace(self):
        src = self._read("rocqCompiler", "QuantumOps.td")
        self.assertIn('cppNamespace = "rocq::mlir::quantum"', src)

    def test_quantum_dialect_cpp_namespace(self):
        src = self._read("rocqCompiler", "QuantumDialect.cpp")
        self.assertIn("namespace rocq", src)
        self.assertNotIn("namespace roc_q", src)

    def test_simulator_ops_td_namespace(self):
        src = self._read("rocqCompiler", "SimulatorOps.td")
        self.assertIn('cppNamespace = "rocq::mlir::sim"', src)


# ===================================================================
# 2. Z gate lowering present in pass
# ===================================================================
class TestZGateLowering(unittest.TestCase):
    def test_z_op_lowering_present(self):
        path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "passes",
                            "QuantumToSimulatorPass.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("ZOp", src, "Z gate lowering missing from pass")
        self.assertIn('"z"', src)


# ===================================================================
# 3. compile_and_execute no silent empty return
# ===================================================================
class TestCompileAndExecuteContract(unittest.TestCase):
    def test_no_placeholder_return(self):
        path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "MLIRCompiler.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        # After compile_and_execute( there should be no "return {};"
        self.assertNotIn("return {};", src,
                         "compile_and_execute still has placeholder return {};")
        self.assertIn("throw std::runtime_error", src)


# ===================================================================
# 4. kernel.mlir() emits parsable MLIR for supported gates
# ===================================================================
class TestKernelMlirEmission(unittest.TestCase):
    def test_h_gate_emits_mlir(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import h

        @QuantumKernel
        def apply_h():
            q = qvec(1)
            h(q[0])

        mlir_str = apply_h.mlir()
        self.assertIn("module", mlir_str)
        self.assertIn("func.func", mlir_str)
        self.assertIn("quantum.h", mlir_str)
        self.assertIn("quantum.qalloc", mlir_str)

    def test_supported_gates(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import h, x, y, z, cnot

        @QuantumKernel
        def multi_gate():
            q = qvec(2)
            h(q[0])
            x(q[1])
            y(q[0])
            z(q[1])
            cnot(q[0], q[1])

        mlir_str = multi_gate.mlir()
        for gate in ["quantum.h", "quantum.x", "quantum.y", "quantum.z", "quantum.cnot"]:
            self.assertIn(gate, mlir_str, f"{gate} missing from MLIR output")

    def test_unsupported_gate_fails(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import sdg

        @QuantumKernel
        def bad_gate():
            q = qvec(1)
            sdg(q[0])

        with self.assertRaises(NotImplementedError) as ctx:
            bad_gate.mlir()
        self.assertIn("sdg", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
