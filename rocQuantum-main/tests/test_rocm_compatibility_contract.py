"""Source contracts for the documented ROCm compatibility surface."""

from __future__ import annotations

import os
import re
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
ROOT_CMAKE = os.path.join(PROJECT_ROOT, "CMakeLists.txt")
PACKAGE_CONFIG_TEMPLATE = os.path.join(PROJECT_ROOT, "cmake", "rocQuantumConfig.cmake.in")
PYTHON_ROCQ_CMAKE = os.path.join(PROJECT_ROOT, "python", "rocq", "CMakeLists.txt")
LOW_LEVEL_BINDINGS = os.path.join(PROJECT_ROOT, "python", "rocq", "bindings.cpp")
SIMULATOR_HEADER = os.path.join(PROJECT_ROOT, "include", "rocquantum", "QuantumSimulator.h")
SIMULATOR_SOURCE = os.path.join(PROJECT_ROOT, "rocquantum", "src", "simulator.cpp")
DENSITYMAT_PYTHON_BINDINGS = os.path.join(
    PROJECT_ROOT, "rocquantum", "src", "python", "py_hip_density_mat.cpp"
)
STATEVEC_HEADER = os.path.join(
    PROJECT_ROOT, "rocquantum", "include", "rocquantum", "hipStateVec.h"
)
STATEVEC_SOURCE = os.path.join(
    PROJECT_ROOT, "rocquantum", "src", "hipStateVec", "hipStateVec.cpp"
)
MULTI_QUBIT_KERNELS = os.path.join(
    PROJECT_ROOT, "rocquantum", "src", "hipStateVec", "multi_qubit_kernels.hip"
)
PYPROJECT = os.path.join(PROJECT_ROOT, "pyproject.toml")
README = os.path.join(PROJECT_ROOT, "README.md")
ROCM_AUDIT = os.path.join(PROJECT_ROOT, "ROCM_INTEGRATION_AUDIT.md")
FEATURE_MATRIX = os.path.join(PROJECT_ROOT, "FEATURE_TRUTH_MATRIX.md")
ROCM_CI_SETUP = os.path.join(REPO_ROOT, "ROCM_CI_SETUP.md")
ROCM_PROBE = os.path.join(PROJECT_ROOT, "scripts", "probe_rocm_runtime.sh")
ROCM_CI_WORKFLOW = os.path.join(REPO_ROOT, ".github", "workflows", "rocm-ci.yml")
ROCM_LINUX_WORKFLOW = os.path.join(REPO_ROOT, ".github", "workflows", "rocm-linux-build.yml")
ROCM_NIGHTLY_WORKFLOW = os.path.join(REPO_ROOT, ".github", "workflows", "rocm-nightly.yml")
STATEVEC_TEST = os.path.join(
    PROJECT_ROOT, "rocquantum", "src", "hipStateVec", "test_hipStateVec_multi_gpu.cpp"
)
TENSORNET_SVD_TEST = os.path.join(
    PROJECT_ROOT, "rocquantum", "src", "hipTensorNet", "test_hipTensorNet_svd.cpp"
)
DENSITYMAT_TEST = os.path.join(
    PROJECT_ROOT, "rocquantum", "tests", "hipDensityMat", "test_hipDensityMat.cpp"
)
DENSITYMAT_TEST_CMAKE = os.path.join(
    PROJECT_ROOT, "rocquantum", "tests", "hipDensityMat", "CMakeLists.txt"
)
COMPONENT_CMAKES = [
    os.path.join(PROJECT_ROOT, "rocquantum", "src", "hipStateVec", "CMakeLists.txt"),
    os.path.join(PROJECT_ROOT, "rocquantum", "src", "hipTensorNet", "CMakeLists.txt"),
    os.path.join(PROJECT_ROOT, "rocquantum", "src", "hipDensityMat", "CMakeLists.txt"),
]


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestRocmCompatibilityContract(unittest.TestCase):
    def test_root_cmake_matches_rocm_hip_language_requirements(self):
        cmake = _read(ROOT_CMAKE)
        match = re.search(r"cmake_minimum_required\(VERSION\s+([0-9]+)\.([0-9]+)", cmake)
        self.assertIsNotNone(match)
        version = (int(match.group(1)), int(match.group(2)))
        self.assertGreaterEqual(version, (3, 21))
        self.assertIn("project(rocQuantum VERSION 0.1.0 LANGUAGES NONE)", cmake)
        self.assertIn("project(rocQuantum VERSION 0.1.0 LANGUAGES CXX HIP)", cmake)
        self.assertIn("option(ROCQUANTUM_BUILD_NATIVE", cmake)
        self.assertIn("if(NOT ROCQUANTUM_BUILD_NATIVE)", cmake)
        self.assertIn("find_package(hip CONFIG REQUIRED)", cmake)
        self.assertIn(
            "find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)",
            cmake,
        )
        self.assertNotIn("find_package(HIP REQUIRED)", cmake)
        self.assertLess(
            cmake.index('set(CMAKE_HIP_ARCHITECTURES "gfx90a"'),
            cmake.index("project(rocQuantum VERSION 0.1.0 LANGUAGES CXX HIP)"),
        )

    def test_c128_precision_option_is_public_and_compile_checked(self):
        root_cmake = _read(ROOT_CMAKE)
        linux_workflow = _read(ROCM_LINUX_WORKFLOW)
        contraction_test = _read(
            os.path.join(
                PROJECT_ROOT,
                "rocquantum",
                "src",
                "hipTensorNet",
                "test_hipTensorNet_contraction_regression.cpp",
            )
        )
        contraction_benchmark = _read(
            os.path.join(PROJECT_ROOT, "benchmarks", "tensornet_contraction_benchmark.cpp")
        )

        self.assertIn("option(ROCQ_PRECISION_DOUBLE", root_cmake)
        self.assertIn("PUBLIC ROCQ_PRECISION_DOUBLE=1", root_cmake)
        for target in ["hipStateVec", "rocqsim_tensornet"]:
            self.assertIn(target, root_cmake)
        precision_block = root_cmake.split("foreach(_rocq_precision_target", 1)[1].split(")", 1)[0]
        self.assertNotIn("rocq_hip_density_mat", precision_block)
        self.assertIn("-DROCQ_PRECISION_DOUBLE=ON", linux_workflow)
        self.assertIn("-DROCQUANTUM_TENSORNET_ENABLE_METIS=ON", linux_workflow)
        self.assertIn("libmetis-dev", linux_workflow)
        self.assertIn("Build C128 precision lane", linux_workflow)
        self.assertIn("validate_cmake_install_consumer.sh build-c128", linux_workflow)
        self.assertIn("ROC_TENSORNET_COMPILED_COMPLEX_DTYPE", contraction_test)
        self.assertIn("ROC_TENSORNET_COMPILED_COMPLEX_DTYPE", contraction_benchmark)
        self.assertNotIn("rocTensorNetworkCreate(&tn, ROC_DATATYPE_C64)", contraction_test)
        self.assertNotIn("rocTensorNetworkCreate(tn, ROC_DATATYPE_C64)", contraction_benchmark)

    def test_python_build_metadata_uses_same_cmake_floor(self):
        with open(PYPROJECT, "rb") as f:
            pyproject = tomllib.load(f)

        build_requirements = pyproject["build-system"]["requires"]
        self.assertNotIn("cmake>=3.21", build_requirements)
        self.assertNotIn("ninja>=1.11", build_requirements)
        scikit_build = pyproject["tool"]["scikit-build"]
        self.assertEqual(scikit_build["cmake"]["version"], ">=3.21")
        self.assertEqual(
            scikit_build["cmake"]["define"]["ROCQUANTUM_BUILD_NATIVE"],
            "OFF",
        )
        self.assertFalse(scikit_build["wheel"]["platlib"])
        self.assertEqual(scikit_build["wheel"]["py-api"], "py3")

    def test_native_python_boundary_has_explicit_precision_and_buffer_safety_contracts(self):
        bindings = _read(LOW_LEVEL_BINDINGS)
        state_header = _read(STATEVEC_HEADER)
        state_source = _read(STATEVEC_SOURCE)

        self.assertNotIn("py::array_t<rocComplex", bindings)
        self.assertIn("using PyComplex = std::complex<double>", bindings)
        self.assertIn("using PyComplex = std::complex<float>", bindings)
        self.assertIn('m.attr("COMPILED_COMPLEX_DTYPE")', bindings)
        self.assertIn("_compiled_complex_roundtrip", bindings)

        self.assertIn("struct ValidatedDeviceMatrix", bindings)
        self.assertIn("matrix_buffer.nbytes() != expected_bytes", bindings)
        self.assertIn("matrix_buffer.is_state_buffer()", bindings)
        self.assertIn("supplied_dimension != matrix.dimension", bindings)
        self.assertNotIn("1U << numTargets", bindings)
        self.assertIn("DeviceBuffer allocation size overflows size_t", bindings)

        self.assertIn("class GateFusionBinding", bindings)
        gate_fusion_body = bindings.split("class GateFusionBinding", 1)[1].split("};", 1)[0]
        self.assertIn('"GateFusion.process_queue"', gate_fusion_body)
        self.assertLess(
            gate_fusion_body.index("get_state_ptr"),
            gate_fusion_body.index("implementation_->processQueue"),
        )
        self.assertIn("py::class_<GateFusionBinding>", bindings)
        self.assertIn("py::keep_alive<1, 2>()", bindings)
        self.assertIn("py::keep_alive<1, 3>()", bindings)

        self.assertIn("rocsvStateInfo_t", state_header)
        self.assertIn("rocsvGetStateInfo", state_header)
        self.assertIn("allocation_generation", state_header)
        self.assertIn("stateGeneration", state_source)
        self.assertIn("rocsvGetStateInfo", state_source)
        self.assertIn("info.allocation_generation != state_generation_", bindings)
        self.assertNotIn("infer_batch_size_from_state_buffer", bindings)

    def test_native_headers_and_read_only_observables_are_compile_safe(self):
        densitymat_bindings = _read(DENSITYMAT_PYTHON_BINDINGS)
        simulator_header = _read(SIMULATOR_HEADER)
        simulator_source = _read(SIMULATOR_SOURCE)
        state_header = _read(STATEVEC_HEADER)

        self.assertIn("#include <pybind11/pybind11.h>", densitymat_bindings)
        self.assertNotIn("#include <pybind11/pybind11>\n", densitymat_bindings)
        self.assertIn("#include <hip/hip_complex.h>", state_header)

        observable_methods = [
            "expectation_value",
            "expectation_pauli_string",
            "expectation_pauli_string_batch",
            "GetExpectationValue",
            "GetExpectationPauliString",
            "GetExpectationPauliStringBatch",
        ]
        for method in observable_methods:
            escaped = re.escape(method)
            self.assertRegex(simulator_header, rf"\b{escaped}\([^;]*\) const;")
            self.assertRegex(
                simulator_source,
                rf"QuantumSimulator::{escaped}\([^{{]*\) const\s*{{",
            )

    def test_root_cmake_activates_legacy_python_backend_owner(self):
        root_cmake = _read(ROOT_CMAKE)
        python_cmake = _read(PYTHON_ROCQ_CMAKE)

        self.assertIn("add_subdirectory(python/rocq)", root_cmake)
        self.assertNotIn("pybind11_add_module(_rocq_hip_backend python/rocq/bindings.cpp", root_cmake)
        self.assertIn("pybind11_add_module(_rocq_hip_backend bindings.cpp)", python_cmake)
        self.assertIn("if(NOT TARGET hipStateVec)", python_cmake)
        self.assertIn("if(NOT TARGET rocqsim_tensornet)", python_cmake)
        self.assertIn("pybind11::module", python_cmake)
        self.assertIn("TARGETS _rocq_hip_backend", python_cmake)
        self.assertNotIn("TARGETS _rocq_hip_backend rocq_hip rocquantum_bind", root_cmake)

    def test_package_config_uses_rocm_config_package_names(self):
        package_config = _read(PACKAGE_CONFIG_TEMPLATE)

        self.assertIn("find_dependency(hip CONFIG REQUIRED)", package_config)
        self.assertNotIn("find_dependency(HIP REQUIRED)", package_config)
        self.assertIn("find_package(rccl QUIET)", package_config)
        self.assertNotIn("find_dependency(rccl", package_config)

    def test_component_cmake_uses_official_rocm_imported_targets(self):
        combined = "\n".join(_read(path) for path in COMPONENT_CMAKES)
        self.assertNotIn("HIP::hip_runtime", combined)
        self.assertIn("hip::host", combined)
        self.assertIn("roc::rocblas", combined)
        self.assertIn("roc::rocsolver", combined)
        self.assertIn("hip::hiprand", combined)
        self.assertNotIn("hiprand::hiprand", combined)
        self.assertIn("TARGET rccl", combined)
        self.assertIn("target_link_libraries(hipStateVec PUBLIC rccl)", combined)

    def test_cpp_sources_with_hip_kernels_are_compiled_as_hip(self):
        statevec_cmake = _read(COMPONENT_CMAKES[0])
        tensornet_cmake = _read(COMPONENT_CMAKES[1])
        densitymat_cmake = _read(COMPONENT_CMAKES[2])
        root_cmake = _read(ROOT_CMAKE)

        self.assertIn(
            "set_source_files_properties(hipStateVec.cpp PROPERTIES LANGUAGE HIP)",
            statevec_cmake,
        )
        self.assertNotIn("HIP_SOURCE_TYPE", statevec_cmake)
        self.assertIn(
            "set_source_files_properties(rocTensorUtil.cpp PROPERTIES LANGUAGE HIP)",
            tensornet_cmake,
        )
        self.assertIn(
            "set_source_files_properties(hipDensityMat.cpp PROPERTIES LANGUAGE HIP)",
            densitymat_cmake,
        )
        self.assertIn(
            "set_source_files_properties(rocquantum/src/kernels.hip.cpp PROPERTIES LANGUAGE HIP)",
            root_cmake,
        )

    def test_native_kernel_entry_points_have_unambiguous_hip_linkage(self):
        statevec_source = _read(STATEVEC_SOURCE)
        multi_qubit_kernels = _read(MULTI_QUBIT_KERNELS)

        internal_declarations = statevec_source.index(
            "namespace {\n\n__global__ void reduce_expectation_z_kernel"
        )
        internal_declarations_end = statevec_source.index(
            "} // namespace", internal_declarations
        )
        external_swap_declarations = statevec_source.index(
            "__global__ void local_bit_swap_permutation_kernel"
        )
        self.assertLess(internal_declarations, internal_declarations_end)
        self.assertLess(internal_declarations_end, external_swap_declarations)

        self.assertIn(
            "__device__ inline void apply_multi_qubit_generic_matrix_device(",
            multi_qubit_kernels,
        )
        self.assertNotIn(
            "__global__ void apply_multi_qubit_generic_matrix_kernel(",
            multi_qubit_kernels,
        )
        self.assertEqual(
            multi_qubit_kernels.count("apply_multi_qubit_generic_matrix_device(state"),
            3,
        )

    def test_docs_record_current_rocm_support_boundary(self):
        readme = _read(README)
        audit = _read(ROCM_AUDIT)
        matrix = _read(FEATURE_MATRIX)
        combined = "\n".join([readme, audit, matrix])

        self.assertIn("7.2.4", combined)
        self.assertIn("CMake `3.21`", combined)
        self.assertIn("gfx950", combined)
        self.assertIn("gfx942", combined)
        self.assertIn("gfx90a", combined)
        self.assertIn("Linux x86_64", combined)
        self.assertIn("`hip` / `hip::host`", combined)
        self.assertIn("`rccl`", combined)

    def test_feature_matrix_uses_versioned_evidence_stages(self):
        matrix = _read(FEATURE_MATRIX)
        for baseline in [
            "cuQuantum SDK `26.06.0`",
            "CUDA-Q `0.15.0`",
            "CUDA-QX `0.6.0`",
            "cuPauliProp `0.4.0`",
            "cuStabilizer `0.4.0`",
        ]:
            self.assertIn(baseline, matrix)

        allowed = {
            "source-present",
            "host-contract-tested",
            "native-single-GPU-verified",
            "native-multi-GPU-verified",
            "performance-verified",
        }
        stages = []
        for line in matrix.splitlines():
            if not line.startswith("|") or line.startswith("| ---"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 8 and cells[0] != "Area":
                stages.append(cells[2])

        self.assertTrue(stages)
        self.assertTrue(set(stages).issubset(allowed))
        self.assertIn("source-present", stages)
        self.assertIn("host-contract-tested", stages)
        self.assertNotIn("native-single-GPU-verified", stages)
        self.assertNotIn("native-multi-GPU-verified", stages)
        self.assertNotIn("performance-verified", stages)

    def test_rocm_runtime_probe_is_fail_fast_and_reused_by_gpu_workflows(self):
        probe = _read(ROCM_PROBE)
        workflows = "\n".join(_read(path) for path in [ROCM_CI_WORKFLOW, ROCM_NIGHTLY_WORKFLOW])
        ci_setup = _read(ROCM_CI_SETUP)

        self.assertIn("set -euo pipefail", probe)
        self.assertIn("require_command hipcc", probe)
        self.assertIn("require_command rocminfo", probe)
        self.assertIn("require_command rocm-smi", probe)
        self.assertIn("cmake_hip_compiler", probe)
        self.assertIn("/llvm/bin/clang++", probe)
        self.assertIn("[[ ! -e /dev/kfd ]]", probe)
        self.assertIn("exit 1", probe)
        self.assertIn("ROCm runtime prerequisites are missing", probe)
        self.assertGreaterEqual(workflows.count("scripts/probe_rocm_runtime.sh"), 2)
        self.assertIn("rocm-runtime-probe.log", workflows)
        self.assertIn("bash scripts/probe_rocm_runtime.sh", ci_setup)
        self.assertIn("CMAKE_HIP_ARCHITECTURES", ci_setup)
        self.assertNotIn("AMDGPU_TARGETS", ci_setup)

    def test_workflows_use_clang_as_cmake_hip_compiler(self):
        workflows = "\n".join(
            _read(path)
            for path in [ROCM_CI_WORKFLOW, ROCM_LINUX_WORKFLOW, ROCM_NIGHTLY_WORKFLOW]
        )

        self.assertGreaterEqual(workflows.count("/opt/rocm/llvm/bin/clang++"), 5)
        self.assertGreaterEqual(workflows.count("CMAKE_PREFIX_PATH"), 4)
        self.assertNotIn("-DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc", workflows)
        self.assertNotIn('-DCMAKE_HIP_COMPILER="${HIPCC_PATH}"', workflows)
        self.assertGreaterEqual(workflows.count("-DROCQUANTUM_BUILD_NATIVE=ON"), 4)

        linux_workflow = _read(ROCM_LINUX_WORKFLOW)
        for python_version in ['"3.9"', '"3.10"', '"3.11"', '"3.12"', '"3.13"']:
            self.assertIn(python_version, linux_workflow)

    def test_native_ctest_graph_has_explicit_evidence_gates(self):
        component_cmake = "\n".join(
            _read(path) for path in [*COMPONENT_CMAKES, DENSITYMAT_TEST_CMAKE]
        )
        statevec_test = _read(STATEVEC_TEST)
        tensornet_svd_test = _read(TENSORNET_SVD_TEST)
        densitymat_test = _read(DENSITYMAT_TEST)

        for test_name in [
            "StateVecSingleGpuSmoke",
            "MultiGPUTests",
            "HipTensorNetSvdRegression",
            "HipDensityMatRegression",
        ]:
            self.assertIn(test_name, component_cmake)
        self.assertGreaterEqual(component_cmake.count('SKIP_RETURN_CODE 77'), 3)
        self.assertIn('LABELS "native;multi-gpu;statevec;rccl;inter-rank"', component_cmake)
        self.assertIn('ENVIRONMENT "HIP_VISIBLE_DEVICES=0"', component_cmake)
        self.assertIn("--require-multi-gpu", component_cmake)
        multi_gpu_properties = component_cmake.split(
            "set_tests_properties(MultiGPUTests PROPERTIES", 1
        )[1].split(")", 1)[0]
        self.assertNotIn("SKIP_RETURN_CODE", multi_gpu_properties)
        self.assertIn("info.local_num_qubits_per_gpu", statevec_test)
        self.assertIn("ROCSV_DISTRIBUTED_BACKEND_RCCL", statevec_test)
        self.assertIn("rocTensorSVD", tensornet_svd_test)
        self.assertIn("reconstruction mismatch", tensornet_svd_test)
        self.assertIn("non-unitary singular vectors", tensornet_svd_test)
        self.assertIn("host_vh", tensornet_svd_test)
        self.assertIn("rocdmApplyCNOT", densitymat_test)
        self.assertNotIn("gtest", densitymat_test.lower())

        for relative_path in [
            ("rocquantum", "src", "hipTensorNet", "test_hipTensorNet_rocTensorUtil.cpp"),
            ("rocquantum", "src", "hipTensorNet", "test_hipTensorNet_slicing.cpp"),
            (
                "rocquantum",
                "src",
                "hipTensorNet",
                "test_hipTensorNet_contraction_regression.cpp",
            ),
            ("rocquantum", "tests", "hipTensorNet", "test_PermutationKernels.cpp"),
        ]:
            source = _read(os.path.join(PROJECT_ROOT, *relative_path))
            self.assertIn("hipGetDeviceCount", source)
            self.assertIn("return 77", source)

        tensornet_header = _read(
            os.path.join(PROJECT_ROOT, "rocquantum", "include", "rocquantum", "hipTensorNet.h")
        )
        self.assertIn("U * S * Vh", tensornet_header)
        self.assertIn("rocTensor* Vh", tensornet_header)

    def test_self_hosted_rocm_runtime_workflow_is_mandatory_source_contract(self):
        workflow = _read(ROCM_CI_WORKFLOW)

        self.assertIn("pull_request:", workflow)
        self.assertIn("rocm-runtime-self-hosted:", workflow)
        self.assertIn("needs: fast-checks", workflow)
        self.assertIn("github.event.pull_request.head.repo.fork == false", workflow)
        for label in ["self-hosted", "linux", "x64", "rocm", "rocm-gpu", "gfx90a"]:
            self.assertIn(f"- {label}", workflow)
        self.assertIn("Probe ROCm runtime prerequisites", workflow)
        self.assertIn("bash scripts/probe_rocm_runtime.sh", workflow)
        self.assertIn("Run ROCm runtime tests (1 GPU smoke)", workflow)
        self.assertIn("-L native -LE multi-gpu", workflow)
        self.assertIn("Run distributed MultiGPUTests when >= 2 GPUs", workflow)
        self.assertIn('-R "MultiGPUTests"', workflow)

        nightly = _read(ROCM_NIGHTLY_WORKFLOW)
        self.assertIn("Require at least two visible GPUs", nightly)
        self.assertIn("visible_gpu_count", nightly)
        self.assertIn('if [ "${GPU_COUNT}" -lt 2 ]', nightly)
        self.assertIn("actions/upload-artifact@v4", workflow)
        self.assertIn("rocm-runtime-${{ github.run_id }}", workflow)


if __name__ == "__main__":
    unittest.main()
