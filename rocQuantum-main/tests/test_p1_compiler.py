"""
P1 Compiler Pipeline Tests - rocQuantum Stabilization

Behavioral tests for MLIR emission and dialect/lowering consistency.
No C++ build required.

    python -m unittest tests.test_p1_compiler -v
"""

import os
import re
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestNamespaceConsistency(unittest.TestCase):
    """Compiler dialect files must use canonical rocq::mlir namespaces."""

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


class TestLoweringCoverage(unittest.TestCase):
    def test_z_op_lowering_present(self):
        path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "passes", "QuantumToSimulatorPass.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("ZOp", src, "Z gate lowering missing from pass")
        self.assertIn('"z"', src)

    def test_param_gate_lowering_present(self):
        path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "passes", "QuantumToSimulatorPass.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("RxOp", src)
        self.assertIn("RyOp", src)
        self.assertIn("RzOp", src)
        self.assertIn("ApplyParamGateOp", src)


class TestCompileAndExecuteContract(unittest.TestCase):
    def test_no_placeholder_return(self):
        path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "MLIRCompiler.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("return {};", src)
        self.assertIn("throw std::runtime_error", src)


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

    def test_cnot_uses_distinct_qubit_ssa_values(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import cnot, h, x

        @QuantumKernel
        def entangle_nonadjacent():
            q = qvec(3)
            h(q[0])
            x(q[1])
            cnot(q[2], q[0])

        mlir_str = entangle_nonadjacent.mlir()
        self.assertIn("%q0, %q1, %q2", mlir_str)
        self.assertIn('"quantum.cnot"(%q2, %q0)', mlir_str)

    def test_param_gates_emit_angle_attributes(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import rx, ry, rz

        @QuantumKernel
        def param_program():
            q = qvec(3)
            rx(0.125, q[0])
            ry(-0.5, q[1])
            rz(1.25, q[2])

        mlir_str = param_program.mlir()
        self.assertIn('"quantum.rx"(%q0) {angle = 0.125 : f64}', mlir_str)
        self.assertIn('"quantum.ry"(%q1) {angle = -0.5 : f64}', mlir_str)
        self.assertIn('"quantum.rz"(%q2) {angle = 1.25 : f64}', mlir_str)

    def test_emitted_ops_exist_in_quantum_dialect(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import cnot, h, rx, z

        @QuantumKernel
        def dialect_covered():
            q = qvec(2)
            h(q[0])
            z(q[1])
            cnot(q[0], q[1])
            rx(0.75, q[1])

        mlir_str = dialect_covered.mlir()

        td_path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "QuantumOps.td")
        with open(td_path, "r", encoding="utf-8") as f:
            td_src = f.read()

        defined_ops = set(re.findall(r'def\s+\w+\s*:\s*Quantum_Op<"([a-z_]+)"', td_src))
        emitted_ops = set(re.findall(r'"quantum\.([a-z_]+)"', mlir_str))
        self.assertTrue(emitted_ops.issubset(defined_ops))

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
