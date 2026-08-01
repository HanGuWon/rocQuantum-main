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
    """Compiler dialect files must use canonical collision-free namespaces."""

    def _read(self, *parts):
        path = os.path.join(_PROJECT_ROOT, *parts)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_quantum_ops_td_namespace(self):
        src = self._read("rocqCompiler", "QuantumOps.td")
        self.assertIn('cppNamespace = "::rocq::quantum"', src)

    def test_quantum_dialect_cpp_namespace(self):
        src = self._read("rocqCompiler", "QuantumDialect.cpp")
        self.assertIn("namespace rocq", src)
        self.assertNotIn("namespace roc_q", src)

    def test_simulator_ops_td_namespace(self):
        src = self._read("rocqCompiler", "SimulatorOps.td")
        self.assertIn('cppNamespace = "::rocq::sim"', src)


class TestLoweringCoverage(unittest.TestCase):
    def _read_direct_pass(self):
        path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "passes", "QuantumToQIRPass.cpp")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_qalloc_lowers_to_static_qir_handles(self):
        src = self._read_direct_pass()
        self.assertIn("QallocLowering", src)
        self.assertIn("LLVM::ConstantOp", src)
        self.assertIn("LLVM::IntToPtrOp", src)
        self.assertIn("applyFullConversion", src)

    def test_fixed_and_parametric_qir_gate_coverage(self):
        src = self._read_direct_pass()
        for token in [
            "quantum.h", "quantum.z", "quantum.sdg", "quantum.tdg",
            "quantum.cnot", "quantum.cz", "quantum.swap", "quantum.ccx",
            "quantum.cswap", "quantum.rx", "quantum.ry", "quantum.rz",
            "quantum.p", "quantum.crx", "quantum.cry", "quantum.crz", "quantum.cp",
        ]:
            self.assertIn(token, src)
        self.assertIn('"s__adj"', src)
        self.assertIn('"t__adj"', src)
        self.assertIn('"r1__body"', src)

    def test_mcx_lowers_when_fixed_and_fails_closed_when_wide(self):
        src = self._read_direct_pass()
        self.assertIn("DecomposeMcx", src)
        self.assertIn("operands.size() == 2", src)
        self.assertIn("operands.size() == 3", src)
        self.assertIn("more than two controls requires QIR", src)
        self.assertIn("control-array lowering and runtime support", src)


class TestCompileAndExecuteContract(unittest.TestCase):
    def _read(self, *parts):
        path = os.path.join(_PROJECT_ROOT, *parts)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_compile_and_execute_uses_qir_orc_jit(self):
        compiler = self._read("rocqCompiler", "MLIRCompiler.cpp")
        execution_engine = self._read(
            "rocqCompiler", "QIRExecutionEngine.cpp"
        )
        cmake = self._read("rocqCompiler", "CMakeLists.txt")

        self.assertNotIn("extractExecutableOps", compiler)
        self.assertNotIn("backend->apply_gate", compiler)
        self.assertNotIn("backend->apply_parametrized_gate", compiler)
        self.assertIn("analyzeQuantumModuleForQIR", compiler)
        self.assertIn("buildQIRPipeline", compiler)
        self.assertIn("QIRExecutionEngine::executeStatic", compiler)

        self.assertIn("ExecutionEngine::create", execution_engine)
        self.assertIn("translateModuleToLLVMIR", execution_engine)
        self.assertIn("finalizeAndVerifyQIRModule", execution_engine)
        self.assertIn("registerSymbols", execution_engine)
        self.assertIn("__quantum__qis__h__body", execution_engine)
        self.assertIn("__quantum__qis__cnot__body", execution_engine)
        self.assertIn("__quantum__qis__rx__body", execution_engine)
        self.assertIn("thread_local ActiveExecution", execution_engine)
        self.assertIn("backend_.initialize", execution_engine)
        self.assertIn("backend.get_state_vector", execution_engine)

        self.assertIn("QIRExecutionEngine.cpp", cmake)
        self.assertIn("MLIRExecutionEngine", cmake)
        self.assertIn("orcjit", cmake)

    def test_binding_documents_compile_and_execute_mvp(self):
        path = os.path.join(_PROJECT_ROOT, "bindings.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        self.assertIn("ROCQUANTUM_ENABLE_MLIR_COMPILER", src)
        self.assertIn("MLIR_COMPILER_ENABLED", src)
        self.assertIn("MLIR_COMPILER_QIR_EMISSION_ENABLED", src)
        self.assertIn("MLIR_COMPILER_ARTIFACT_EMISSION_ENABLED", src)
        self.assertIn("MLIR_COMPILER_QIR_JIT_ENABLED", src)
        self.assertIn("MLIR_COMPILER_CPU_EXECUTION_ENABLED", src)
        self.assertIn("MLIR_COMPILER_GPU_EXECUTION_ENABLED", src)
        self.assertIn("MLIR_COMPILER_QIR_PROFILE", src)
        self.assertIn("MLIR_COMPILER_RUNTIME_KIND", src)
        self.assertIn(
            "qir_v2_static_jit_cpu_and_hip_execution_base_emission", src
        )
        self.assertIn("MLIR_COMPILER_ARTIFACT_KINDS", src)
        self.assertIn('def("emit_artifact"', src)
        self.assertIn("disabled_runtime_guard", src)
        self.assertIn("DisabledRuntimeMLIRCompiler", src)
        self.assertIn("MLIR compiler support is disabled", src)
        self.assertIn("supported measurement-free MLIR subset", src)
        self.assertIn("in-process JIT", src)
        self.assertIn("cpu_statevec or hip_statevec backend", src)
        self.assertIn("Unsupported ops raise actionable diagnostics", src)
        self.assertNotIn("Stub API", src)

    def test_root_cmake_wires_cpu_only_and_native_compiler_modes(self):
        path = os.path.join(_PROJECT_ROOT, "CMakeLists.txt")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        self.assertIn("option(ROCQUANTUM_ENABLE_MLIR_COMPILER", src)
        self.assertIn("option(ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR", src)
        self.assertIn("elseif(ROCQUANTUM_ENABLE_MLIR_COMPILER)", src)
        self.assertIn("if(ROCQUANTUM_ENABLE_MLIR_COMPILER)", src)
        self.assertIn("add_subdirectory(rocqCompiler)", src)
        self.assertIn("target_link_libraries(rocquantum_bind PRIVATE rocqCompiler)", src)
        self.assertIn("ROCQUANTUM_ENABLE_MLIR_COMPILER=1", src)
        self.assertNotIn("not wired into the release CMake graph", src)

    def test_compiler_cmake_pins_mlir_and_builds_tools(self):
        src = self._read("rocqCompiler", "CMakeLists.txt")
        self.assertIn("LLVM_VERSION_MAJOR EQUAL 22", src)
        self.assertIn("LLVM_VERSION_MINOR EQUAL 1", src)
        self.assertIn("RocqQuantumIncGen", src)
        self.assertIn("RocqCompilerPassIncGen", src)
        self.assertIn("add_mlir_conversion_library(RocqQuantumToQIR", src)
        self.assertIn("add_library(rocqCompiler STATIC", src)
        self.assertIn("add_executable(rocq-opt", src)
        self.assertIn("add_executable(rocq-translate", src)
        self.assertIn("add_executable(rocq-run", src)
        self.assertIn("CompilerArtifacts.cpp", src)
        self.assertIn("rocqCompilerTooling", src)
        self.assertIn("ROCQ_COMPILER_FINGERPRINT", src)
        self.assertIn('INSTALL_RPATH "${LLVM_LIBRARY_DIRS}"', src)
        self.assertIn("rocq.compiler.artifact-cli", src)
        self.assertIn("rocq.compiler.run-cli", src)

    def test_cpu_ci_builds_release_pinned_compiler(self):
        workflow_path = os.path.join(
            os.path.dirname(_PROJECT_ROOT), ".github", "workflows", "rocm-ci.yml"
        )
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = f.read()

        job = workflow.split("  mlir-compiler:", 1)[1].split(
            "  rocm-runtime-self-hosted:", 1
        )[0]
        self.assertIn("apt.llvm.org", job)
        self.assertIn("llvm-22-dev", job)
        self.assertIn("libmlir-22-dev", job)
        self.assertIn('= "22.1"', job)
        self.assertIn("-DROCQUANTUM_BUILD_NATIVE=OFF", job)
        self.assertIn("-DROCQUANTUM_ENABLE_MLIR_COMPILER=ON", job)
        self.assertNotIn("ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR", job)
        self.assertIn("ctest --test-dir", job)
        self.assertIn("cmake --install", job)
        self.assertIn("installed-bell.qir.ll", job)
        self.assertIn('qir_emission_kind"] == "rocq_translate_cli"', job)
        self.assertIn("actions/upload-artifact@v4", job)

    def test_legacy_binding_uses_conceptual_mlir_holder(self):
        path = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        self.assertIn("ConceptualMLIRCompiler", src)
        self.assertIn("legacy conceptual MLIR holder", src)
        self.assertNotIn("rocquantum::compiler::MLIRCompiler", src)
        self.assertNotIn("mlir::MLIRContext", src)

    def test_legacy_dialect_scaffold_is_source_valid_but_not_release_parity(self):
        header = self._read(
            "rocquantum", "include", "rocquantum", "Dialect", "QuantumOps.h.inc"
        )
        op_list = self._read(
            "rocquantum", "src", "rocqCompiler", "QuantumOps.cpp.inc"
        )
        dialect = self._read(
            "rocquantum", "src", "rocqCompiler", "QuantumDialect.cpp"
        )

        header.encode("ascii")
        op_list.encode("ascii")
        self.assertIn("Legacy source scaffold", header)
        self.assertIn("not treat it as generated TableGen coverage", header)
        self.assertIn("getMeasurement() { return getOperation()->getResult(0); }", header)
        self.assertNotIn("Format()", header)
        self.assertNotIn("Typo", header)
        self.assertIn("#ifdef GET_OP_LIST", op_list)
        for op_name in ["AllocQubitOp,", "DeallocQubitOp,", "GenericGateOp,", "MeasureOp"]:
            self.assertIn(op_name, op_list)
        self.assertIn("not as parity evidence", dialect)
        self.assertIn("outside the release compiler path", dialect)

    def test_legacy_adjoint_generation_scaffold_uses_valid_attr_handling(self):
        source = self._read(
            "rocquantum",
            "src",
            "rocqCompiler",
            "Transforms",
            "AdjointGeneration.cpp",
        )

        source.encode("ascii")
        self.assertIn('#include "rocquantum/Dialect/QuantumDialect.h"', source)
        self.assertNotIn("Assumes this path", source)
        self.assertIn("auto isCurrentlyAdjoint", source)
        self.assertIn("isCurrentlyAdjoint.getValue()", source)
        self.assertNotIn("bool isCurrentlyAdjoint", source)

    def test_tdg_reaches_native_statevec_dispatch(self):
        backend_path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "HipStateVecBackend.cpp")
        simulator_path = os.path.join(_PROJECT_ROOT, "rocquantum", "src", "simulator.cpp")
        hipstatevec_path = os.path.join(_PROJECT_ROOT, "rocquantum", "src", "hipStateVec", "hipStateVec.cpp")
        header_path = os.path.join(_PROJECT_ROOT, "rocquantum", "include", "rocquantum", "hipStateVec.h")
        legacy_binding_path = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")
        legacy_api_path = os.path.join(_PROJECT_ROOT, "python", "rocq", "api.py")

        with open(backend_path, "r", encoding="utf-8") as f:
            backend_src = f.read()
        with open(simulator_path, "r", encoding="utf-8") as f:
            simulator_src = f.read()
        with open(hipstatevec_path, "r", encoding="utf-8") as f:
            hipstatevec_src = f.read()
        with open(header_path, "r", encoding="utf-8") as f:
            header_src = f.read()
        with open(legacy_binding_path, "r", encoding="utf-8") as f:
            binding_src = f.read()
        with open(legacy_api_path, "r", encoding="utf-8") as f:
            legacy_api_src = f.read()

        self.assertIn("emplace_alias(\"tdg\"", backend_src)
        self.assertIn("rocsvApplyTdg", backend_src)
        self.assertIn("normalized == \"TDG\"", simulator_src)
        self.assertIn("rocqStatus_t rocsvApplyTdg", header_src)
        self.assertIn("rocqStatus_t rocsvApplyTdg", hipstatevec_src)
        self.assertIn("phase = -pi / 4.0", hipstatevec_src)
        self.assertIn("apply_tdg", binding_src)
        self.assertIn("def tdg", legacy_api_src)

    def test_phase_gates_reach_native_statevec_dispatch(self):
        backend_path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "HipStateVecBackend.cpp")
        simulator_path = os.path.join(_PROJECT_ROOT, "rocquantum", "src", "simulator.cpp")
        hipstatevec_path = os.path.join(_PROJECT_ROOT, "rocquantum", "src", "hipStateVec", "hipStateVec.cpp")
        header_path = os.path.join(_PROJECT_ROOT, "rocquantum", "include", "rocquantum", "hipStateVec.h")
        legacy_binding_path = os.path.join(_PROJECT_ROOT, "python", "rocq", "bindings.cpp")
        legacy_api_path = os.path.join(_PROJECT_ROOT, "python", "rocq", "api.py")

        with open(backend_path, "r", encoding="utf-8") as f:
            backend_src = f.read()
        with open(simulator_path, "r", encoding="utf-8") as f:
            simulator_src = f.read()
        with open(hipstatevec_path, "r", encoding="utf-8") as f:
            hipstatevec_src = f.read()
        with open(header_path, "r", encoding="utf-8") as f:
            header_src = f.read()
        with open(legacy_binding_path, "r", encoding="utf-8") as f:
            binding_src = f.read()
        with open(legacy_api_path, "r", encoding="utf-8") as f:
            legacy_api_src = f.read()

        self.assertIn("emplace_alias(\"p\"", backend_src)
        self.assertIn("emplace_alias(\"cp\"", backend_src)
        self.assertIn("rocsvApplyP", backend_src)
        self.assertIn("rocsvApplyCP", backend_src)
        self.assertIn("normalized == \"P\"", simulator_src)
        self.assertIn("normalized == \"CP\"", simulator_src)
        self.assertIn("rocqStatus_t rocsvApplyP", header_src)
        self.assertIn("rocqStatus_t rocsvApplyCP", header_src)
        self.assertIn("rocqStatus_t rocsvApplyPBatch", header_src)
        self.assertIn("rocqStatus_t rocsvApplyCPBatch", header_src)
        self.assertIn("rocqStatus_t rocsvApplyP", hipstatevec_src)
        self.assertIn("rocqStatus_t rocsvApplyCP", hipstatevec_src)
        self.assertIn("rocqStatus_t rocsvApplyPBatch", hipstatevec_src)
        self.assertIn("rocqStatus_t rocsvApplyCPBatch", hipstatevec_src)
        self.assertIn("make_complex(std::cos(theta), std::sin(theta))", hipstatevec_src)
        self.assertIn("apply_p", binding_src)
        self.assertIn("apply_cp", binding_src)
        self.assertIn("def p", legacy_api_src)
        self.assertIn("def cp", legacy_api_src)


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
        from rocq.gates import cp, crx, cry, crz, p, rx, ry, rz

        @QuantumKernel
        def param_program():
            q = qvec(3)
            rx(0.125, q[0])
            ry(-0.5, q[1])
            rz(1.25, q[2])
            p(-0.125, q[2])
            crx(0.25, q[0], q[1])
            cry(-0.75, q[1], q[2])
            crz(1.5, q[2], q[0])
            cp(-1.25, q[0], q[2])

        mlir_str = param_program.mlir()
        self.assertIn('"quantum.rx"(%q0) {angle = 0.125 : f64}', mlir_str)
        self.assertIn('"quantum.ry"(%q1) {angle = -0.5 : f64}', mlir_str)
        self.assertIn('"quantum.rz"(%q2) {angle = 1.25 : f64}', mlir_str)
        self.assertIn('"quantum.p"(%q2) {angle = -0.125 : f64}', mlir_str)
        self.assertIn('"quantum.crx"(%q0, %q1) {angle = 0.25 : f64}', mlir_str)
        self.assertIn('"quantum.cry"(%q1, %q2) {angle = -0.75 : f64}', mlir_str)
        self.assertIn('"quantum.crz"(%q2, %q0) {angle = 1.5 : f64}', mlir_str)
        self.assertIn('"quantum.cp"(%q0, %q2) {angle = -1.25 : f64}', mlir_str)

    def test_extended_core_gates_emit_mlir(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import cz, s, sdg, swap, t, tdg

        @QuantumKernel
        def extended_core_program():
            q = qvec(2)
            s(q[0])
            sdg(q[1])
            t(q[0])
            tdg(q[1])
            cz(q[0], q[1])
            swap(q[0], q[1])

        mlir_str = extended_core_program.mlir()
        self.assertIn('"quantum.s"(%q0)', mlir_str)
        self.assertIn('"quantum.sdg"(%q1)', mlir_str)
        self.assertIn('"quantum.t"(%q0)', mlir_str)
        self.assertIn('"quantum.tdg"(%q1)', mlir_str)
        self.assertIn('"quantum.cz"(%q0, %q1)', mlir_str)
        self.assertIn('"quantum.swap"(%q0, %q1)', mlir_str)

    def test_multi_control_gates_emit_mlir(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import ccx, cswap, fredkin, mcx, toffoli

        @QuantumKernel
        def multi_control_program():
            q = qvec(5)
            ccx(q[0], q[1], q[2])
            mcx([q[0], q[1], q[2]], q[4])
            toffoli(q[1], q[2], q[3])
            cswap(q[0], q[3], q[4])
            fredkin(q[2], q[0], q[4])

        mlir_str = multi_control_program.mlir()
        self.assertIn('"quantum.ccx"(%q0, %q1, %q2)', mlir_str)
        self.assertIn('"quantum.mcx"(%q0, %q1, %q2, %q4)', mlir_str)
        self.assertIn('"quantum.ccx"(%q1, %q2, %q3)', mlir_str)
        self.assertIn('"quantum.cswap"(%q0, %q3, %q4)', mlir_str)
        self.assertIn('"quantum.cswap"(%q2, %q0, %q4)', mlir_str)

    def test_mcx_requires_control_and_target(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import mcx

        @QuantumKernel
        def missing_control():
            q = qvec(1)
            mcx([], q[0])

        with self.assertRaises(ValueError) as ctx:
            missing_control.mlir()
        self.assertIn("at least 2", str(ctx.exception))

    def test_mlir_emission_rejects_duplicate_gate_targets(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import cnot, cp, mcx

        @QuantumKernel
        def duplicate_cnot_target():
            q = qvec(1)
            cnot(q[0], q[0])

        @QuantumKernel
        def duplicate_controlled_phase_target():
            q = qvec(1)
            cp(0.125, q[0], q[0])

        @QuantumKernel
        def duplicate_mcx_control_target():
            q = qvec(2)
            mcx([q[0], q[0]], q[1])

        for kernel_obj in [duplicate_cnot_target, duplicate_controlled_phase_target, duplicate_mcx_control_target]:
            with self.subTest(kernel=kernel_obj.name):
                with self.assertRaisesRegex(ValueError, "target qubits must be distinct"):
                    kernel_obj.mlir()

    def test_emitted_ops_exist_in_quantum_dialect(self):
        from rocq.kernel import QuantumKernel
        from rocq.qvec import qvec
        from rocq.gates import ccx, cnot, cp, cswap, h, mcx, p, rx, tdg, z

        @QuantumKernel
        def dialect_covered():
            q = qvec(5)
            h(q[0])
            tdg(q[0])
            z(q[1])
            p(0.25, q[1])
            cnot(q[0], q[1])
            cp(0.5, q[0], q[1])
            ccx(q[0], q[1], q[2])
            cswap(q[0], q[2], q[3])
            mcx([q[0], q[1], q[2]], q[4])
            rx(0.75, q[3])

        mlir_str = dialect_covered.mlir()

        td_path = os.path.join(_PROJECT_ROOT, "rocqCompiler", "QuantumOps.td")
        with open(td_path, "r", encoding="utf-8") as f:
            td_src = f.read()

        defined_ops = set(
            re.findall(
                r'def\s+\w+\s*:\s*(?:Quantum_Op|Quantum_StateOp)<"([a-z_]+)"',
                td_src,
            )
        )
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
