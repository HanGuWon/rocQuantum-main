"""
P2 Packaging Tests — rocQuantum Stabilization

Validate canonical imports, legacy shim, and pyproject.toml existence.

    python -m unittest tests.test_p2_packaging -v
"""

import os
import sys
import unittest
import warnings

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestCanonicalImports(unittest.TestCase):
    """Core symbols must be importable from canonical rocq path."""

    def test_import_rocq_kernel(self):
        from rocq.kernel import QuantumKernel, execute
        self.assertIsNotNone(QuantumKernel)
        self.assertIsNotNone(execute)

    def test_import_rocq_operator(self):
        from rocq.operator import PauliOperator, SumOperator, get_expectation_value
        self.assertIsNotNone(PauliOperator)

    def test_import_rocq_gates(self):
        from rocq.gates import h, x, y, z, cnot, rx, ry, rz
        self.assertIsNotNone(h)


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
        from rocq.legacy import kernel, execute
        self.assertIsNotNone(kernel)


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
        path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        self.assertEqual(data["project"]["name"], "rocquantum")


if __name__ == "__main__":
    unittest.main()
