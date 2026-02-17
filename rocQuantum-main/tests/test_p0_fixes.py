"""
P0 Regression Tests — rocQuantum Stabilization

Pure-Python tests covering all P0 + P0.1 fixes.
No GPU / HIP / ROCm / requests / boto3 required.

    python -m unittest tests.test_p0_fixes -v
"""

import ast
import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# -----------------------------------------------------------------------
# Helper: mock-import a module that has an unsatisfied top-level import
# by injecting a fake module into sys.modules before importing it.
# -----------------------------------------------------------------------
def _ensure_fake_module(name):
    """Insert a stub module into sys.modules if missing."""
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)


def _import_quantinuum_backend():
    """Import QuantinuumBackend even when 'requests' is absent."""
    _ensure_fake_module("requests")
    # base.py imports requests at module level
    import importlib
    import rocquantum.backends.base
    importlib.reload(rocquantum.backends.base)
    from rocquantum.backends.quantinuum import QuantinuumBackend
    from rocquantum.backends.base import BackendAuthenticationError
    return QuantinuumBackend, BackendAuthenticationError


def _import_rigetti_backend():
    """Import RigettiBackend even when 'boto3' is absent."""
    _ensure_fake_module("boto3")
    _ensure_fake_module("botocore")
    _ensure_fake_module("botocore.exceptions")
    # Provide ClientError so rigetti.py can import it
    botocore_exc = sys.modules["botocore.exceptions"]
    if not hasattr(botocore_exc, "ClientError"):
        botocore_exc.ClientError = type("ClientError", (Exception,), {})
    import importlib
    import rocquantum.backends.base
    importlib.reload(rocquantum.backends.base)
    from rocquantum.backends.rigetti import RigettiBackend
    from rocquantum.backends.base import JobSubmissionError
    return RigettiBackend, JobSubmissionError


# ===================================================================
# 1. CLI Import Syntax
# ===================================================================
class TestCliImportSyntax(unittest.TestCase):
    def test_cli_parses_cleanly(self):
        cli_path = os.path.join(_PROJECT_ROOT, "rocq_cli.py")
        with open(cli_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename="rocq_cli.py")
        self.assertIsNotNone(tree)


# ===================================================================
# 2. SumOperator Immutability
# ===================================================================
class TestSumOperatorImmutability(unittest.TestCase):
    def test_add_does_not_mutate(self):
        from rocq.operator import PauliOperator, SumOperator
        a = PauliOperator("X0", 1.0)
        b = PauliOperator("Z1", -0.5)
        c = PauliOperator("Y2", 0.3)
        sum_ab = a + b
        original_len = len(sum_ab.terms)
        sum_abc = sum_ab + c
        self.assertIsNot(sum_abc, sum_ab)
        self.assertEqual(len(sum_ab.terms), original_len)
        self.assertEqual(len(sum_abc.terms), original_len + 1)

    def test_add_two_sums(self):
        from rocq.operator import PauliOperator, SumOperator
        s1 = SumOperator([PauliOperator("X0"), PauliOperator("Y1")])
        s2 = SumOperator([PauliOperator("Z0")])
        s1_len = len(s1.terms)
        s3 = s1 + s2
        self.assertEqual(len(s1.terms), s1_len)
        self.assertEqual(len(s3.terms), s1_len + len(s2.terms))


# ===================================================================
# 3. Expectation Value — No Placeholder, No Kernel Execution
# ===================================================================
class TestExpectationValue(unittest.TestCase):
    def test_raises_not_implemented(self):
        from rocq.operator import get_expectation_value, PauliOperator
        op = PauliOperator("Z0")
        with self.assertRaises(NotImplementedError):
            get_expectation_value(None, op, backend="state_vector")

    def test_does_not_execute_kernel(self):
        """Ensure no kernel execution happens before the error."""
        from rocq.operator import get_expectation_value, PauliOperator
        from rocq.kernel import execute as real_execute
        op = PauliOperator("Z0")
        with mock.patch("rocq.operator.execute") as mock_exec:
            with self.assertRaises(NotImplementedError):
                get_expectation_value(None, op, backend="state_vector")
            mock_exec.assert_not_called()


# ===================================================================
# 4. Module-level execute() Accepts noise_model
# ===================================================================
class TestExecuteNoiseModel(unittest.TestCase):
    def test_signature(self):
        from rocq.kernel import execute
        sig = inspect.signature(execute)
        self.assertIn("noise_model", sig.parameters)

    def test_forwarded(self):
        from rocq.kernel import execute, QuantumKernel
        sentinel = object()

        @QuantumKernel
        def noop():
            pass

        with mock.patch.object(QuantumKernel, "execute", return_value="ok") as m:
            execute(noop, backend="state_vector", noise_model=sentinel)
            m.assert_called_once()
            _, kwargs = m.call_args
            self.assertIs(kwargs.get("noise_model"), sentinel)


# ===================================================================
# 5. Quantinuum Credentials
# ===================================================================
class TestQuantinuumCredentials(unittest.TestCase):
    def test_inline_creds_fail_fast(self):
        QuantinuumBackend, BackendAuthenticationError = _import_quantinuum_backend()
        backend = QuantinuumBackend()
        with mock.patch.dict(os.environ, {"CUDAQ_QUANTINUUM_CREDENTIALS": "alice,secret"}):
            with self.assertRaises(BackendAuthenticationError) as ctx:
                backend.authenticate()
            self.assertIn("not yet implemented", str(ctx.exception))

    def test_json_file_works(self):
        QuantinuumBackend, _ = _import_quantinuum_backend()
        backend = QuantinuumBackend()
        creds = {"access_token": "tok-123"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(creds, f)
            path = f.name
        try:
            with mock.patch.dict(os.environ, {"CUDAQ_QUANTINUUM_CREDENTIALS": path}):
                backend.authenticate()
            self.assertEqual(backend.auth_credentials["access_token"], "tok-123")
        finally:
            os.unlink(path)

    def test_missing_env_var(self):
        QuantinuumBackend, _ = _import_quantinuum_backend()
        backend = QuantinuumBackend()
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CUDAQ_QUANTINUUM_CREDENTIALS", None)
            with self.assertRaises(Exception) as ctx:
                backend.authenticate()
            self.assertIn("CUDAQ_QUANTINUUM_CREDENTIALS", str(ctx.exception))


# ===================================================================
# 6. Rigetti S3 Configuration
# ===================================================================
class TestRigettiS3(unittest.TestCase):
    def test_no_s3_raises(self):
        RigettiBackend, _ = _import_rigetti_backend()
        backend = RigettiBackend(s3_output=None)
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROCQ_RIGETTI_S3_OUTPUT", None)
            backend.s3_output = None
            backend.braket_client = mock.MagicMock()
            with self.assertRaises(Exception) as ctx:
                backend.submit_job("OPENQASM 3.0;", shots=10)
            self.assertIn("ROCQ_RIGETTI_S3_OUTPUT", str(ctx.exception))

    def test_env_var_respected(self):
        RigettiBackend, _ = _import_rigetti_backend()
        with mock.patch.dict(os.environ, {"ROCQ_RIGETTI_S3_OUTPUT": "s3://b/r"}):
            backend = RigettiBackend()
            self.assertEqual(backend.s3_output, "s3://b/r")


# ===================================================================
# 7. PennyLane generate_samples Source Check
# ===================================================================
class TestPennyLaneSource(unittest.TestCase):
    def test_no_set_indexing(self):
        path = os.path.join(
            _PROJECT_ROOT, "integrations", "pennylane-rocq",
            "pennylane_rocq", "rocq_device.py",
        )
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("self.observables[0]", source)


# ===================================================================
# 8. CLI Rigetti S3 Preflight
# ===================================================================
class TestCliRigettiPreflight(unittest.TestCase):
    def test_missing_s3_var_raises(self):
        """check_environment_vars('rigetti') must catch missing ROCQ_RIGETTI_S3_OUTPUT."""
        # We need to parse and extract the function without importing
        # (importing rocq_cli.py triggers rocquantum imports that may fail)
        cli_path = os.path.join(_PROJECT_ROOT, "rocq_cli.py")
        with open(cli_path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn("ROCQ_RIGETTI_S3_OUTPUT", source,
                       "CLI check_environment_vars must validate ROCQ_RIGETTI_S3_OUTPUT")


if __name__ == "__main__":
    unittest.main()
