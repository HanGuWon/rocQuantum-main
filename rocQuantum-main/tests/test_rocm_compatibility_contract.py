"""Source contracts for the documented ROCm compatibility surface."""

from __future__ import annotations

import os
import re
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_CMAKE = os.path.join(PROJECT_ROOT, "CMakeLists.txt")
PYPROJECT = os.path.join(PROJECT_ROOT, "pyproject.toml")
README = os.path.join(PROJECT_ROOT, "README.md")
ROCM_AUDIT = os.path.join(PROJECT_ROOT, "ROCM_INTEGRATION_AUDIT.md")
FEATURE_MATRIX = os.path.join(PROJECT_ROOT, "FEATURE_TRUTH_MATRIX.md")
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


if __name__ == "__main__":
    unittest.main()
