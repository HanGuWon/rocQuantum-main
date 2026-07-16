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

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib

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
            runtime_capabilities,
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
        self.assertIsNotNone(runtime_capabilities)
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
        self.assertTrue(callable(rocq.distributed_capabilities))
        self.assertTrue(callable(rocq.compiler_capabilities))
        self.assertTrue(callable(rocq.runtime_capabilities))
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


class TestHostOnlyUtilityImports(unittest.TestCase):
    def test_hamiltonian_utility_does_not_import_native_density_binding(self):
        path = os.path.join(_PROJECT_ROOT, "rocquantum", "utils", "hamiltonian.py")
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertNotIn("import rocq_hip", source)
        from rocquantum.utils.hamiltonian import compute_hamiltonian_expectation

        class FakeState:
            def __init__(self):
                self.gates = []

            def apply_gate(self, matrix, qubit_idx, adjoint=False):
                self.gates.append((matrix.copy(), qubit_idx, adjoint))

            def _compute_z_product_expectation(self, qubit_indices):
                self.measured = list(qubit_indices)
                return 0.25

        state = FakeState()
        value = compute_hamiltonian_expectation([("XI", 2.0), ("II", -0.5)], state)

        self.assertAlmostEqual(value, 0.0)
        self.assertEqual(state.measured, [0])
        self.assertEqual(len(state.gates), 2)


class TestPyprojectExists(unittest.TestCase):
    def test_pyproject_toml_exists(self):
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        self.assertTrue(os.path.isfile(path))

    def test_pyproject_has_name(self):
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        self.assertEqual(data["project"]["name"], "rocquantum")

    def test_pyproject_version_matches_cmake_package_version(self):
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
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        self.assertEqual(data["build-system"]["build-backend"], "scikit_build_core.build")

    def test_host_wheel_uses_current_scikit_build_cmake_configuration(self):
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        scikit_build = data["tool"]["scikit-build"]
        self.assertEqual(scikit_build["cmake"]["version"], ">=3.21")
        self.assertEqual(
            scikit_build["cmake"]["define"]["ROCQUANTUM_BUILD_NATIVE"],
            "OFF",
        )
        self.assertFalse(scikit_build["wheel"]["platlib"])
        self.assertEqual(scikit_build["wheel"]["py-api"], "py3")
        self.assertIn("rocquantum/src/**", scikit_build["wheel"]["exclude"])
        native_overrides = [
            override
            for override in scikit_build["overrides"]
            if override.get("if", {}).get("env", {}).get("ROCQ_BUILD_NATIVE") is True
        ]
        self.assertEqual(len(native_overrides), 1)
        self.assertTrue(native_overrides[0]["wheel"]["platlib"])
        self.assertEqual(
            native_overrides[0]["cmake"]["define"]["ROCQUANTUM_BUILD_NATIVE"],
            "ON",
        )
        with open(_ROOT_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()
        self.assertIn("option(ROCQUANTUM_BUILD_NATIVE", cmake)
        self.assertIn("if(NOT ROCQUANTUM_BUILD_NATIVE)", cmake)
        self.assertIn("project(rocQuantum VERSION 0.1.0 LANGUAGES CXX HIP)", cmake)
        self.assertIn("project(rocQuantum VERSION 0.1.0 LANGUAGES NONE)", cmake)
        self.assertIn("Native scikit-build wheels require ROCQ_BUILD_NATIVE=1", cmake)

    def test_cli_runtime_dependency_is_declared(self):
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        self.assertIn("requests>=2.28", data["project"]["dependencies"])

    def test_pyproject_declares_core_runtime_dependency(self):
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        self.assertIn("numpy>=1.21", data["project"]["dependencies"])

    def test_pyproject_includes_framework_adapter_packages(self):
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
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        optional = data["project"]["optional-dependencies"]
        self.assertIn("cirq-core>=1.0,<2", optional["cirq"])
        self.assertIn("scipy>=1.10", optional["solvers"])
        self.assertIn("qiskit>=2.4,<3; python_version >= '3.10'", optional["qiskit"])
        self.assertEqual(
            optional["pennylane"],
            ["pennylane>=0.45,<0.46; python_version >= '3.11'"],
        )
        self.assertIn("rocquantum[backends,pennylane,qiskit,cirq,solvers,dev]", optional["all"])

    def test_integration_setup_py_files_are_compatibility_installers(self):
        for name, setup_path in _INTEGRATION_SETUP_FILES.items():
            with self.subTest(name=name, setup_path=setup_path):
                with open(setup_path, "r", encoding="utf-8") as f:
                    source = f.read()

                self.assertIn("root_project_version(__file__)", source)
                self.assertIn("compatibility_long_description", source)
                self.assertIn("Compatibility installer", source)
                minimum_python = {
                    "qiskit": ">=3.10",
                    "pennylane": ">=3.11",
                    "cirq": ">=3.9",
                }[name]
                self.assertIn(f'python_requires="{minimum_python}"', source)
                self.assertNotIn("author=\"Gemini\"", source)

    def test_integration_setup_py_versions_follow_root_pyproject(self):
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
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)

        optional = data["project"]["optional-dependencies"]
        expected_dependencies = {
            "qiskit": [optional["qiskit"][0].split(";", 1)[0].strip()],
            "pennylane": [optional["pennylane"][0].split(";", 1)[0].strip()],
            "cirq": optional["cirq"],
        }
        for name, setup_path in _INTEGRATION_SETUP_FILES.items():
            with self.subTest(name=name):
                with open(setup_path, "r", encoding="utf-8") as f:
                    source = f.read()
                for dependency in expected_dependencies[name]:
                    self.assertIn(dependency, source)


class TestCMakeInstallConsumerSmoke(unittest.TestCase):
    def test_native_install_excludes_unwired_mlir_public_headers(self):
        with open(_ROOT_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()

        self.assertIn('PATTERN "Compiler" EXCLUDE', cmake)
        self.assertIn('PATTERN "Dialect" EXCLUDE', cmake)

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
        self.assertIn("target_link_libraries(", cmake)
        self.assertIn("rocquantum_install_consumer_symbols", cmake)
        self.assertTrue(source.startswith("#include <rocquantum/hipDensityMat.hpp>"))
        self.assertIn("sizeof(hipComplex) == 2 * sizeof(float)", source)
        self.assertIn("#ifdef ROCQ_EXPECT_METIS\n#include <metis.h>\n#endif", source)
        self.assertIn("#include <rocquantum/QuantumSimulator.h>", source)
        self.assertIn("#include <rocquantum/hipStateVec.h>", source)
        self.assertIn("#include <rocquantum/hipTensorNet.h>", source)
        self.assertIn("#include <rocquantum/hipTensorNet_api.h>", source)
        self.assertIn("#include <rocquantum/hipDensityMat.h>", source)
        self.assertIn("&rocquantum::QuantumSimulator::num_qubits", source)
        self.assertIn("rocsvDestroy(nullptr)", source)
        self.assertIn("rocdmDestroyState(nullptr)", source)
        self.assertIn("rocTensorNetworkGetCapabilities(&tensornet_caps)", source)
        self.assertIn("#ifdef ROCQ_PRECISION_DOUBLE", source)
        self.assertIn("sizeof(rocComplex) == sizeof(rocDoubleComplex)", source)
        self.assertIn("tensornet_caps.supports_c128 == 1", source)

    def test_install_consumer_script_installs_and_configures_downstream_project(self):
        with open(_INSTALL_CONSUMER_SCRIPT, "r", encoding="utf-8") as f:
            script = f.read()

        self.assertIn("set -euo pipefail", script)
        self.assertIn("cmake --install", script)
        self.assertIn("ROCQUANTUM_INSTALL_PREFIX", script)
        self.assertIn("ROCQUANTUM_INSTALL_CONSUMER_BUILD_DIR", script)
        self.assertIn("cmake/install_consumer_smoke", script)
        self.assertIn('consumer_prefix_path="${INSTALL_PREFIX}"', script)
        self.assertIn('${consumer_prefix_path};${CMAKE_PREFIX_PATH}', script)
        self.assertIn("-DCMAKE_PREFIX_PATH=${consumer_prefix_path}", script)
        self.assertIn("cmake --build", script)
        self.assertIn('ctest --test-dir "${CONSUMER_BUILD_DIR}"', script)
        self.assertIn("--no-tests=error", script)

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

    def test_rocm_workflow_builds_and_installs_a_clean_host_wheel(self):
        with open(_ROCM_LINUX_WORKFLOW, "r", encoding="utf-8") as f:
            workflow = f.read()

        self.assertIn("python -m build --wheel --outdir dist", workflow)
        self.assertIn("dist/rocquantum-*.whl", workflow)
        self.assertIn("working-directory: ${{ runner.temp }}", workflow)
        self.assertIn("-py3-none-any.whl", workflow)
        self.assertIn("Root-Is-Purelib: true", workflow)
        self.assertIn("Run installed-wheel examples outside the source tree", workflow)

    def test_native_wheel_has_relocatable_loader_and_external_import_gate(self):
        with open(_ROOT_CMAKE, "r", encoding="utf-8") as f:
            cmake = f.read()
        with open(_ROCM_LINUX_WORKFLOW, "r", encoding="utf-8") as f:
            workflow = f.read()

        self.assertIn('INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}"', cmake)
        for target in ["_rocq_hip_backend", "rocq_hip", "rocquantum_bind"]:
            self.assertIn(target, cmake)
        self.assertIn("Build and import installed native wheel outside the source tree", workflow)
        self.assertIn("binutils", workflow)
        self.assertIn('ROCQ_BUILD_NATIVE: "1"', workflow)
        self.assertIn("readelf -d", workflow)
        self.assertIn("ldd", workflow)
        self.assertIn("env -u PYTHONPATH", workflow)
        for module_name in ["_rocq_hip_backend", "rocq_hip", "rocquantum_bind"]:
            self.assertIn(f"import {module_name}", workflow)
        self.assertIn('COMPILED_COMPLEX_DTYPE == "complex64"', workflow)
        self.assertIn('COMPILED_COMPLEX_DTYPE == "complex128"', workflow)
        self.assertIn("_compiled_complex_roundtrip", workflow)
        self.assertIn("env -u PYTHONPATH", workflow)
        self.assertIn('test "${EXAMPLE_COUNT}" -eq 19', workflow)
        self.assertIn("Verify minimum Qiskit adapter import", workflow)
        self.assertIn("Verify minimum combined adapter contracts", workflow)
        rocm_build_job = workflow.split("\n  build:\n", 1)[1]
        self.assertIn("Install checkout dependency", rocm_build_job)
        self.assertIn("working-directory: /tmp", rocm_build_job)
        self.assertIn("apt-get install -y --no-install-recommends git", rocm_build_job)
        self.assertLess(
            rocm_build_job.index("Install checkout dependency"),
            rocm_build_job.index("- name: Checkout"),
        )
        self.assertIn('"pennylane==0.45.0"', workflow)
        self.assertIn('"qiskit==2.4.0"', workflow)
        self.assertIn('"cirq-core==1.5.0"', workflow)
        for rocm_development_package in [
            "hiprand-dev",
            "rocblas-dev",
            "rocrand-dev",
            "rocsolver-dev",
        ]:
            self.assertIn(rocm_development_package, workflow)
        self.assertNotIn("pybind11-dev", workflow)
        self.assertGreaterEqual(workflow.count('"pybind11==2.13.6"'), 2)
        self.assertIn("PYBIND11_CMAKE_DIR", workflow)
        self.assertIn("pybind11Config.cmake", workflow)
        self.assertGreaterEqual(workflow.count('-Dpybind11_DIR="${PYBIND11_CMAKE_DIR}"'), 2)
        self.assertIn("integrations/qiskit-rocquantum-provider/tests/test_backend.py", workflow)
        self.assertIn("matrix.python-version == '3.10'", workflow)
        self.assertIn('if [ "${{ matrix.python-version }}" != "3.9" ]', workflow)
        self.assertIn('matrix.python-version }}" != "3.10"', workflow)
        self.assertNotIn("pennylane>=0.38", workflow)
        self.assertNotIn("pip install -e .", workflow)


if __name__ == "__main__":
    unittest.main()
