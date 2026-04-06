"""Focused smoke tests for the canonical rocq framework surface."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

# Add the parent directory to the path to allow importing 'rocq'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rocq
from rocq.kernel import QuantumKernel


class TestRocqFramework(unittest.TestCase):
    """Test suite for the core rocq framework components."""

    def test_kernel_creation_and_structure(self):
        @rocq.kernel
        def my_kernel(theta: float):
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])
            rocq.rz(theta, q[1])

        self.assertIsInstance(my_kernel, QuantumKernel)
        self.assertEqual(my_kernel.name, "my_kernel")

        ctx = my_kernel.build(0.123)
        self.assertEqual(len(ctx.ops), 3)
        self.assertEqual([op.name.lower() for op in ctx.ops], ["h", "cnot", "rz"])
        self.assertEqual(my_kernel.num_qubits, 2)

    def test_backend_factory_validation(self):
        @rocq.kernel
        def dummy_kernel():
            q = rocq.qvec(1)
            rocq.h(q[0])

        with self.assertRaises(ValueError) as cm:
            rocq.execute(dummy_kernel, backend="invalid_backend")

        self.assertIn("Unsupported backend 'invalid_backend'", str(cm.exception))
        self.assertIn("['state_vector', 'density_matrix']", str(cm.exception))

    def test_state_vector_backend_noise_rejection(self):
        @rocq.kernel
        def dummy_kernel():
            q = rocq.qvec(1)
            rocq.h(q[0])

        noise = rocq.NoiseModel()
        noise.add_channel("depolarizing", 0.1)

        fake_backend = mock.Mock()
        fake_backend.run_ops.side_effect = NotImplementedError(
            "Noise models are only supported by the 'density_matrix' backend."
        )

        with mock.patch("rocq.kernel.get_backend", return_value=fake_backend):
            with self.assertRaises(NotImplementedError) as cm:
                rocq.execute(dummy_kernel, backend="state_vector", noise_model=noise)

        self.assertEqual(
            str(cm.exception),
            "Noise models are only supported by the 'density_matrix' backend.",
        )


if __name__ == "__main__":
    unittest.main()
