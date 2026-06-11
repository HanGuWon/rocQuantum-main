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
        self.assertIn("CrxOp", src)
        self.assertIn("CryOp", src)
        self.assertIn("CrzOp", src)
        self.assertIn("ApplyParamGateOp", src)

    def test_extended_core_gate_lowering_present(self):
        path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "passes", "QuantumToSimulatorPass.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        for token in ["SOp", "SdgOp", "TOp", "CzOp", "SwapOp", "CcxOp", "CswapOp"]:
            self.assertIn(token, src)


class TestCompileAndExecuteContract(unittest.TestCase):
    def test_compile_and_execute_dispatches_supported_subset(self):
        path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "MLIRCompiler.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        self.assertNotIn("return {};", src)
        self.assertNotIn("not yet implemented", src)
        self.assertIn("extract_executable_ops", src)
        self.assertIn("backend->initialize(num_qubits)", src)
        self.assertIn("backend->apply_gate", src)
        self.assertIn("backend->apply_parametrized_gate", src)
        self.assertIn("backend->get_state_vector", src)
        self.assertIn("unsupported quantum op", src)
        self.assertIn("quantum.rx", src)
        self.assertIn("quantum.ry", src)
        self.assertIn("quantum.rz", src)
        self.assertIn("quantum.crx", src)
        self.assertIn("quantum.cry", src)
        self.assertIn("quantum.crz", src)
        self.assertIn("quantum.cz", src)
        self.assertIn("quantum.swap", src)
        self.assertIn("quantum.cnot", src)
        self.assertIn("quantum.ccx", src)
        self.assertIn("quantum.cswap", src)

    def test_binding_documents_compile_and_execute_mvp(self):
        path = os.path.join(_PROJECT_ROOT, "bindings.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        self.assertIn("qalloc, H/X/Y/Z/S/Sdg/T, CNOT/CZ/SWAP/CCX/CSWAP, RX/RY/RZ, CRX/CRY/CRZ", src)
        self.assertIn("Unsupported ops raise actionable diagnostics", src)
        self.assertNotIn("Stub API", src)


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
        from rocq.gates import crx, cry, crz, rx, ry, rz

        @QuantumKernel
        def param_program():
            q = qvec(3)
            rx(0.125, q[0])
            ry(-0.5, q[1])
            rz(1.25, q[2])
            crx(0.25, q[0], q[1])
            cry(-0.75, q[1], q[2])
            crz(1.5, q[2], q[0])

        mlir_str = param_program.mlir()
        self.assertIn('"quantum.rx"(%q0) {angle = 0.125 : f64}', mlir_str)
        self.assertIn('"quantum.ry"(%q1) {angle = -0.5 : f64}', mlir_str)
        self.assertIn('"quantum.rz"(%q2) {angle = 1.25 : f64}', mlir_str)
        self.assertIn('"quantum.crx"(%q0, %q1) {angle = 0.25 : f64}', mlir_str)
        self.assertIn('"quantum.cry"(%q1, %q2) {angle = -0.75 : f64}', mlir_str)
        self.assertIn('"quantum.crz"(%q2, %q0) {angle = 1.5 : f64}', mlir_str)

    def test_extended_core_gates_emit_mlir(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import cz, s, sdg, swap, t

        @QuantumKernel
        def extended_core_program():
            q = qvec(2)
            s(q[0])
            sdg(q[1])
            t(q[0])
            cz(q[0], q[1])
            swap(q[0], q[1])

        mlir_str = extended_core_program.mlir()
        self.assertIn('"quantum.s"(%q0)', mlir_str)
        self.assertIn('"quantum.sdg"(%q1)', mlir_str)
        self.assertIn('"quantum.t"(%q0)', mlir_str)
        self.assertIn('"quantum.cz"(%q0, %q1)', mlir_str)
        self.assertIn('"quantum.swap"(%q0, %q1)', mlir_str)

    def test_multi_control_gates_emit_mlir(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import ccx, cswap, fredkin, toffoli

        @QuantumKernel
        def multi_control_program():
            q = qvec(5)
            ccx(q[0], q[1], q[2])
            toffoli(q[1], q[2], q[3])
            cswap(q[0], q[3], q[4])
            fredkin(q[2], q[0], q[4])

        mlir_str = multi_control_program.mlir()
        self.assertIn('"quantum.ccx"(%q0, %q1, %q2)', mlir_str)
        self.assertIn('"quantum.ccx"(%q1, %q2, %q3)', mlir_str)
        self.assertIn('"quantum.cswap"(%q0, %q3, %q4)', mlir_str)
        self.assertIn('"quantum.cswap"(%q2, %q0, %q4)', mlir_str)

    def test_emitted_ops_exist_in_quantum_dialect(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import ccx, cnot, cswap, h, rx, z

        @QuantumKernel
        def dialect_covered():
            q = qvec(4)
            h(q[0])
            z(q[1])
            cnot(q[0], q[1])
            ccx(q[0], q[1], q[2])
            cswap(q[0], q[2], q[3])
            rx(0.75, q[3])

        mlir_str = dialect_covered.mlir()

        td_path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "QuantumOps.td")
        with open(td_path, "r", encoding="utf-8") as f:
            td_src = f.read()

        defined_ops = set(re.findall(r'def\s+\w+\s*:\s*Quantum_Op<"([a-z_]+)"', td_src))
        emitted_ops = set(re.findall(r'"quantum\.([a-z_]+)"', mlir_str))
        self.assertTrue(emitted_ops.issubset(defined_ops))

    def test_unsupported_gate_fails(self):
        from rocq.kernel import QuantumKernel, _KernelBuildContext
        from rocq.qvec import qvec

        @QuantumKernel
        def bad_gate():
            q = qvec(1)
            _KernelBuildContext.add_gate("unsupported_gate", [q[0]])

        with self.assertRaises(NotImplementedError) as ctx:
            bad_gate.mlir()
        self.assertIn("unsupported_gate", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
