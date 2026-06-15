"""Contracts for third-party provider backend support boundaries."""

from __future__ import annotations

import os
import sys
import unittest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestUnsupportedProviderBackends(unittest.TestCase):
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
