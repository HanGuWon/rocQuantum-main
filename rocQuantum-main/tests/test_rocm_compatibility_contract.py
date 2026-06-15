"""Source contracts for the documented ROCm compatibility surface."""

from __future__ import annotations

import os
import re
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
ROOT_CMAKE = os.path.join(PROJECT_ROOT, "CMakeLists.txt")
PACKAGE_CONFIG_TEMPLATE = os.path.join(PROJECT_ROOT, "cmake", "rocQuantumConfig.cmake.in")
PYTHON_ROCQ_CMAKE = os.path.join(PROJECT_ROOT, "python", "rocq", "CMakeLists.txt")
PYPROJECT = os.path.join(PROJECT_ROOT, "pyproject.toml")
README = os.path.join(PROJECT_ROOT, "README.md")
ROCM_AUDIT = os.path.join(PROJECT_ROOT, "ROCM_INTEGRATION_AUDIT.md")
FEATURE_MATRIX = os.path.join(PROJECT_ROOT, "FEATURE_TRUTH_MATRIX.md")
ROCM_CI_SETUP = os.path.join(REPO_ROOT, "ROCM_CI_SETUP.md")
ROCM_PROBE = os.path.join(PROJECT_ROOT, "scripts", "probe_rocm_runtime.sh")
ROCM_CI_WORKFLOW = os.path.join(REPO_ROOT, ".github", "workflows", "rocm-ci.yml")
ROCM_NIGHTLY_WORKFLOW = os.path.join(REPO_ROOT, ".github", "workflows", "rocm-nightly.yml")
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
        self.assertIn("project(rocQuantum VERSION 0.1.0 LANGUAGES CXX HIP)", cmake)
        self.assertIn("find_package(hip CONFIG REQUIRED)", cmake)
        self.assertNotIn("find_package(HIP REQUIRED)", cmake)

    def test_python_build_metadata_uses_same_cmake_floor(self):
        pyproject = _read(PYPROJECT)
        self.assertIn('"cmake>=3.21"', pyproject)
        self.assertIn('minimum-version = "3.21"', pyproject)
        self.assertNotIn("cmake>=3.18", pyproject)
        self.assertNotIn('minimum-version = "3.18"', pyproject)

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
        self.assertIn("hiprand::hiprand", combined)
        self.assertIn("TARGET rccl", combined)
        self.assertIn("target_link_libraries(hipStateVec PUBLIC rccl)", combined)

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

    def test_rocm_runtime_probe_is_fail_fast_and_reused_by_gpu_workflows(self):
        probe = _read(ROCM_PROBE)
        workflows = "\n".join(_read(path) for path in [ROCM_CI_WORKFLOW, ROCM_NIGHTLY_WORKFLOW])
        ci_setup = _read(ROCM_CI_SETUP)

        self.assertIn("set -euo pipefail", probe)
        self.assertIn("require_command hipcc", probe)
        self.assertIn("require_command rocminfo", probe)
        self.assertIn("require_command rocm-smi", probe)
        self.assertIn("[[ ! -e /dev/kfd ]]", probe)
        self.assertIn("exit 1", probe)
        self.assertIn("ROCm runtime prerequisites are missing", probe)
        self.assertGreaterEqual(workflows.count("scripts/probe_rocm_runtime.sh"), 2)
        self.assertIn("rocm-runtime-probe.log", workflows)
        self.assertIn("bash scripts/probe_rocm_runtime.sh", ci_setup)
        self.assertIn("CMAKE_HIP_ARCHITECTURES", ci_setup)
        self.assertNotIn("AMDGPU_TARGETS", ci_setup)


if __name__ == "__main__":
    unittest.main()
