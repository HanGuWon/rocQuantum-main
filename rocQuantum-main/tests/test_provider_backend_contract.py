"""Contracts for third-party provider backend support boundaries."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestUnsupportedProviderBackends(unittest.TestCase):
    def test_list_backends_hides_skeleton_providers_by_default(self):
        from rocquantum.core import list_backends

        backends = list_backends()

        self.assertIn("ionq", backends)
        self.assertIn("rigetti", backends)
        self.assertEqual(backends["qristal"]["status"], "local_cli_client")
        self.assertTrue(backends["qristal"]["requires_local_runtime"])
        self.assertEqual(backends["qristal"]["runtime"], "Qristal SDK CLI")
        self.assertNotIn("iqm", backends)
        self.assertNotIn("alice_bob", backends)
        self.assertFalse(any(info["requires_experimental_opt_in"] for info in backends.values()))

    def test_list_backends_can_include_skeleton_provider_metadata(self):
        from rocquantum.core import list_backends

        backends = list_backends(include_experimental=True)

        self.assertEqual(backends["iqm"]["status"], "unsupported_stub")
        self.assertTrue(backends["iqm"]["requires_experimental_opt_in"])
        self.assertFalse(backends["iqm"]["safe_to_submit_jobs"])
        self.assertIn("Provider SDK/API integration", backends["iqm"]["unsupported_reason"])
        self.assertEqual(
            backends["iqm"]["missing_capabilities"],
            [
                "authentication",
                "authorization_headers",
                "payload_builder",
                "job_submission",
                "status_polling",
                "result_retrieval",
            ],
        )
        self.assertIn("rocquantum.backends.iqm.IQMBackend", backends["iqm"]["import_path"])

    def test_set_target_blocks_skeleton_provider_by_default(self):
        from rocquantum.core import set_target

        with self.assertRaisesRegex(ValueError, "unsupported skeleton provider"):
            set_target("iqm")

    def test_set_target_allows_explicit_skeleton_provider_opt_in(self):
        from rocquantum.backends.base import UnsupportedBackendError
        from rocquantum.core import set_target

        with self.assertRaises(UnsupportedBackendError):
            set_target("iqm", allow_experimental=True)

    def test_set_target_allows_environment_skeleton_provider_opt_in(self):
        from rocquantum.backends.base import UnsupportedBackendError
        from rocquantum.core import set_target

        with mock.patch.dict(os.environ, {"ROCQ_ENABLE_EXPERIMENTAL_PROVIDERS": "1"}):
            with self.assertRaises(UnsupportedBackendError):
                set_target("iqm")

    def test_qristal_authenticate_requires_real_cli(self):
        from rocquantum.backends.base import BackendAuthenticationError
        from rocquantum.backends.qristal import QuantumBrillianceBackend

        backend = QuantumBrillianceBackend()

        with mock.patch("rocquantum.backends.qristal.shutil.which", return_value=None):
            with self.assertRaisesRegex(BackendAuthenticationError, "Qristal SDK CLI"):
                backend.authenticate()

    def test_qristal_submit_job_invokes_cli_and_parses_histogram(self):
        from rocquantum.backends.qristal import QuantumBrillianceBackend
        from rocquantum.circuit import QuantumCircuit

        backend = QuantumBrillianceBackend()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        completed = mock.Mock(stdout='Execution complete\nHistogram: {"00": 3, "11": 2}\n', stderr="")

        with mock.patch("rocquantum.backends.qristal.shutil.which", return_value="qristal"):
            backend.authenticate()
            with mock.patch("rocquantum.backends.qristal.subprocess.run", return_value=completed) as run:
                job_id = backend.submit_job(circuit, shots=5)

        command = run.call_args.args[0]
        self.assertEqual(command[0], "qristal")
        self.assertEqual(command[-2:], ["--shots", "5"])
        self.assertFalse(os.path.exists(command[2]))
        self.assertEqual(backend.get_job_status(job_id), "completed")
        self.assertEqual(backend.get_job_result(job_id), {"00": 3, "11": 2})

    def test_qristal_submit_job_reports_missing_cli(self):
        from rocquantum.backends.base import JobSubmissionError
        from rocquantum.backends.qristal import QuantumBrillianceBackend
        from rocquantum.circuit import QuantumCircuit

        backend = QuantumBrillianceBackend()

        with mock.patch("rocquantum.backends.qristal.shutil.which", return_value=None):
            with self.assertRaisesRegex(JobSubmissionError, "Qristal SDK CLI"):
                backend.submit_job(QuantumCircuit(1), shots=1)

    def test_qristal_submit_job_reports_cli_failure(self):
        import subprocess

        from rocquantum.backends.base import JobSubmissionError
        from rocquantum.backends.qristal import QuantumBrillianceBackend
        from rocquantum.circuit import QuantumCircuit

        backend = QuantumBrillianceBackend()
        error = subprocess.CalledProcessError(2, ["qristal"], stderr="bad qasm")

        with mock.patch("rocquantum.backends.qristal.shutil.which", return_value="qristal"):
            backend.authenticate()
            with mock.patch("rocquantum.backends.qristal.subprocess.run", side_effect=error):
                with self.assertRaisesRegex(JobSubmissionError, "bad qasm"):
                    backend.submit_job(QuantumCircuit(1), shots=1)

    def test_qristal_rejects_nonpositive_shots(self):
        from rocquantum.backends.base import JobSubmissionError
        from rocquantum.backends.qristal import QuantumBrillianceBackend
        from rocquantum.circuit import QuantumCircuit

        backend = QuantumBrillianceBackend()

        with self.assertRaisesRegex(JobSubmissionError, "positive integer"):
            backend.submit_job(QuantumCircuit(1), shots=0)

    def test_skeleton_provider_backends_fail_explicitly(self):
        from rocquantum.backends.alice_bob import AliceBobBackend
        from rocquantum.backends.base import UnsupportedBackendError
        from rocquantum.backends.iqm import IQMBackend
        from rocquantum.backends.orca import OrcaBackend
        from rocquantum.backends.quantum_machines import QuantumMachinesBackend
        from rocquantum.backends.quera import QuEraBackend
        from rocquantum.backends.seeqc import SeeqcBackend
        from rocquantum.backends.xanadu import XanaduBackend

        backend_classes = [
            AliceBobBackend,
            IQMBackend,
            OrcaBackend,
            QuantumMachinesBackend,
            QuEraBackend,
            SeeqcBackend,
            XanaduBackend,
        ]

        for backend_cls in backend_classes:
            backend = backend_cls()
            with self.subTest(backend=backend_cls.__name__):
                capabilities = backend.capabilities()
                self.assertEqual(capabilities["status"], "unsupported_stub")
                self.assertFalse(capabilities["safe_to_submit_jobs"])
                self.assertIn("job_submission", capabilities["missing_capabilities"])
                with self.assertRaises(UnsupportedBackendError):
                    backend.authenticate()
                with self.assertRaises(UnsupportedBackendError):
                    backend.submit_job("OPENQASM 2.0;", 100)
                with self.assertRaises(UnsupportedBackendError):
                    backend.get_job_status("job-id")
                with self.assertRaises(UnsupportedBackendError):
                    backend.get_job_result("job-id")

    def test_skeleton_provider_sources_do_not_silently_pass(self):
        skeleton_files = [
            "alice_bob.py",
            "iqm.py",
            "orca.py",
            "quantum_machines.py",
            "quera.py",
            "seeqc.py",
            "xanadu.py",
        ]
        backends_dir = os.path.join(_PROJECT_ROOT, "rocquantum", "backends")

        for filename in skeleton_files:
            with self.subTest(filename=filename):
                with open(os.path.join(backends_dir, filename), "r", encoding="utf-8") as f:
                    source = f.read()
                self.assertNotIn("pass", source)
                self.assertNotIn("TODO", source)
                self.assertIn("UnsupportedBackend", source)


if __name__ == "__main__":
    unittest.main()
