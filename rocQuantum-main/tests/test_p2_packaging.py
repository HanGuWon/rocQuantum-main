"""
P2 Packaging Tests — rocQuantum Stabilization

Validate canonical imports, legacy shim, and pyproject.toml existence.

    python -m unittest tests.test_p2_packaging -v
"""

import os
import importlib.util
import re
import sys
import unittest
import warnings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_INSTALL_CONSUMER_CMAKE = os.path.join(_PROJECT_ROOT, "cmake", "install_consumer_smoke", "CMakeLists.txt")
_INSTALL_CONSUMER_MAIN = os.path.join(_PROJECT_ROOT, "cmake", "install_consumer_smoke", "main.cpp")
_INSTALL_CONSUMER_SCRIPT = os.path.join(_PROJECT_ROOT, "scripts", "validate_cmake_install_consumer.sh")
_ROCM_LINUX_WORKFLOW = os.path.join(_REPO_ROOT, ".github", "workflows", "rocm-linux-build.yml")
_README = os.path.join(_PROJECT_ROOT, "README.md")
_ROOT_CMAKE = os.path.join(_PROJECT_ROOT, "CMakeLists.txt")
_INTEGRATIONS_DIR = os.path.join(_PROJECT_ROOT, "integrations")
_COMPAT_SETUP_HELPER = os.path.join(_INTEGRATIONS_DIR, "_compat_setup.py")
_INTEGRATION_SETUP_FILES = {
    "qiskit": os.path.join(_INTEGRATIONS_DIR, "qiskit-rocquantum-provider", "setup.py"),
    "pennylane": os.path.join(_INTEGRATIONS_DIR, "pennylane-rocq", "setup.py"),
    "cirq": os.path.join(_INTEGRATIONS_DIR, "cirq-rocm", "setup.py"),
}


class TestCanonicalImports(unittest.TestCase):
    """Core symbols must be importable from canonical rocq path."""

    def test_import_rocq_kernel(self):
        from rocq.kernel import (
            QuantumKernel,
            compiler_capabilities,
            execute,
            execute_async,
            get_state,
            get_state_async,
            observe,
            observe_async,
            sample,
            sample_async,
        )
        self.assertIsNotNone(QuantumKernel)
        self.assertIsNotNone(compiler_capabilities)
        self.assertIsNotNone(execute)
        self.assertIsNotNone(execute_async)
        self.assertIsNotNone(get_state)
        self.assertIsNotNone(get_state_async)
        self.assertIsNotNone(observe)
        self.assertIsNotNone(observe_async)
        self.assertIsNotNone(sample)
        self.assertIsNotNone(sample_async)

    def test_import_rocq_operator(self):
        from rocq.operator import PauliOperator, SparseHamiltonianOperator, SumOperator, get_expectation_value
        self.assertIsNotNone(PauliOperator)
        self.assertIsNotNone(SparseHamiltonianOperator)

    def test_import_rocq_gates(self):
        from rocq.gates import h, x, y, z, cnot, rx, ry, rz
        self.assertIsNotNone(h)


class TestFutureCanonicalRuntimeSurface(unittest.TestCase):
    """Forward-looking contract checks for the new canonical runtime API."""

    def test_observe_and_sample_exports(self):
        import rocq
        self.assertTrue(callable(rocq.compiler_capabilities))
        self.assertTrue(callable(rocq.observe))
        self.assertTrue(callable(rocq.sample))
        self.assertTrue(callable(rocq.get_state))
        self.assertTrue(callable(rocq.get_state_async))
        self.assertTrue(callable(rocq.observe_async))
        self.assertTrue(callable(rocq.sample_async))


class TestLegacyShim(unittest.TestCase):
    """rocq.legacy must re-export all symbols with a DeprecationWarning."""

    def test_deprecation_warning(self):
        # Force reimport to trigger the warning
        if "rocq.legacy" in sys.modules:
            del sys.modules["rocq.legacy"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import rocq.legacy
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_exports_pauli(self):
        from rocq.legacy import PauliOperator
        self.assertIsNotNone(PauliOperator)

    def test_exports_kernel(self):
        from rocq.legacy import (
            compile_and_execute,
            compile_and_execute_async,
            execute,
            execute_async,
            get_state,
            get_state_async,
            kernel,
        )
        self.assertIsNotNone(kernel)
        self.assertIsNotNone(compile_and_execute)
        self.assertIsNotNone(compile_and_execute_async)
        self.assertIsNotNone(execute)
        self.assertIsNotNone(execute_async)
        self.assertIsNotNone(get_state)
        self.assertIsNotNone(get_state_async)


class TestVqeImports(unittest.TestCase):
    """VQE solver must import without rocquantum.python.rocq."""

    def test_no_dead_import(self):
        path = os.path.join(_PROJECT_ROOT, "rocquantum", "solvers", "vqe_solver.py")
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("rocquantum.python.rocq", source,
                         "vqe_solver.py still imports from dead path")

    def test_uses_canonical_imports(self):
        path = os.path.join(_PROJECT_ROOT, "rocquantum", "solvers", "vqe_solver.py")
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn("import rocq", source)
        self.assertIn("from rocq.operator import PauliOperator", source)


class TestQecImports(unittest.TestCase):
    """QEC framework must import without rocquantum.python.rocq."""

    def test_no_dead_import(self):
        path = os.path.join(_PROJECT_ROOT, "rocquantum", "qec", "framework.py")
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("rocquantum.python.rocq", source,
                         "framework.py still imports from dead path")


class TestPyprojectExists(unittest.TestCase):
    def test_pyproject_toml_exists(self):
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        self.assertTrue(os.path.isfile(path))

    def test_pyproject_has_name(self):
        import tomllib
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        self.assertEqual(data["project"]["name"], "rocquantum")

    def test_pyproject_version_matches_cmake_package_version(self):
        import tomllib
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        with open(_ROOT_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()

        match = re.search(r"project\(rocQuantum\s+VERSION\s+([0-9]+(?:\.[0-9]+){2})", cmake)
        self.assertIsNotNone(match, "CMake project version must be explicit for install package config.")
        self.assertEqual(data["project"]["version"], match.group(1))
        self.assertIn("write_basic_package_version_file", cmake)
        self.assertIn("VERSION ${PROJECT_VERSION}", cmake)

    def test_pyproject_uses_scikit_build_core(self):
        import tomllib
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        self.assertEqual(data["build-system"]["build-backend"], "scikit_build_core.build")

    def test_pyproject_declares_core_runtime_dependency(self):
        import tomllib
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        self.assertIn("numpy>=1.21", data["project"]["dependencies"])

    def test_pyproject_includes_framework_adapter_packages(self):
        import tomllib
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        packages = data["tool"]["scikit-build"]["wheel"]["packages"]
        self.assertEqual(
            packages["qiskit_rocquantum_provider"],
            "integrations/qiskit-rocquantum-provider/qiskit_rocquantum_provider",
        )
        self.assertEqual(packages["pennylane_rocq"], "integrations/pennylane-rocq/pennylane_rocq")
        self.assertEqual(packages["cirq_rocm"], "integrations/cirq-rocm/cirq_rocm")

    def test_pyproject_all_extra_includes_cirq_adapter_dependency(self):
        import tomllib
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        optional = data["project"]["optional-dependencies"]
        self.assertIn("cirq-core>=1.0", optional["cirq"])
        self.assertIn("scipy>=1.10", optional["solvers"])
        self.assertIn("rocquantum[backends,pennylane,qiskit,cirq,solvers,dev]", optional["all"])

    def test_integration_setup_py_files_are_compatibility_installers(self):
        for setup_path in _INTEGRATION_SETUP_FILES.values():
            with self.subTest(setup_path=setup_path):
                with open(setup_path, "r", encoding="utf-8") as f:
                    source = f.read()

                self.assertIn("root_project_version(__file__)", source)
                self.assertIn("compatibility_long_description", source)
                self.assertIn("Compatibility installer", source)
                self.assertIn("python_requires=\">=3.9\"", source)
                self.assertNotIn("author=\"Gemini\"", source)

    def test_integration_setup_py_versions_follow_root_pyproject(self):
        import tomllib
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            pyproject_version = tomllib.load(f)["project"]["version"]

        spec = importlib.util.spec_from_file_location("compat_setup", _COMPAT_SETUP_HELPER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        compat_setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compat_setup)

        for setup_path in _INTEGRATION_SETUP_FILES.values():
            with self.subTest(setup_path=setup_path):
                self.assertEqual(compat_setup.root_project_version(setup_path), pyproject_version)

    def test_integration_setup_py_dependencies_match_root_optional_extras(self):
        import tomllib
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        optional = data["project"]["optional-dependencies"]
        expected_dependencies = {
            "qiskit": optional["qiskit"][0],
            "pennylane": optional["pennylane"][0],
            "cirq": optional["cirq"][0],
        }
        for name, setup_path in _INTEGRATION_SETUP_FILES.items():
            with self.subTest(name=name):
                with open(setup_path, "r", encoding="utf-8") as f:
                    source = f.read()
                self.assertIn(expected_dependencies[name], source)


class TestCMakeInstallConsumerSmoke(unittest.TestCase):
    def test_consumer_smoke_project_checks_installed_targets_and_headers(self):
        with open(_INSTALL_CONSUMER_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()
        with open(_INSTALL_CONSUMER_MAIN, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("find_package(rocQuantum CONFIG REQUIRED)", cmake)
        for target in [
            "rocquantum::rocquantum",
            "rocquantum::hipStateVec",
            "rocquantum::rocqsim_tensornet",
            "rocquantum::rocq_hip_density_mat",
        ]:
            self.assertIn(target, cmake)
        self.assertIn("target_link_libraries(rocquantum_install_consumer_smoke PRIVATE rocquantum::rocquantum)", cmake)
        self.assertIn("#include <rocquantum/QuantumSimulator.h>", source)
        self.assertIn("#include <rocquantum/hipStateVec.h>", source)
        self.assertIn("#include <rocquantum/hipTensorNet_api.h>", source)
        self.assertIn("#include <rocquantum/hipDensityMat.h>", source)

    def test_install_consumer_script_installs_and_configures_downstream_project(self):
        with open(_INSTALL_CONSUMER_SCRIPT, "r", encoding="utf-8") as f:
            script = f.read()

        self.assertIn("set -euo pipefail", script)
        self.assertIn("cmake --install", script)
        self.assertIn("ROCQUANTUM_INSTALL_PREFIX", script)
        self.assertIn("ROCQUANTUM_INSTALL_CONSUMER_BUILD_DIR", script)
        self.assertIn("cmake/install_consumer_smoke", script)
        self.assertIn("-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}", script)
        self.assertIn("cmake --build", script)

    def test_rocm_workflow_and_readme_expose_install_consumer_validation(self):
        with open(_ROCM_LINUX_WORKFLOW, "r", encoding="utf-8") as f:
            workflow = f.read()
        with open(_README, "r", encoding="utf-8") as f:
            readme = f.read()

        self.assertIn("Validate CMake install-tree consumer", workflow)
        self.assertIn("scripts/validate_cmake_install_consumer.sh", workflow)
        self.assertIn("cmake-install-consumer.log", workflow)
        self.assertIn("CMAKE_HIP_ARCHITECTURES", workflow)
        self.assertIn("scripts/validate_cmake_install_consumer.sh", readme)


if __name__ == "__main__":
    unittest.main()
