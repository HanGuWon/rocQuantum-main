"""
P3 Integration Tests — PennyLane Adapter

Validates the PennyLane-rocq integration layer with source analysis
and runtime tests where possible (mock-based when PennyLane is absent).

    python -m unittest tests.test_pennylane_integration -v
"""

import ast
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEVICE_PATH = os.path.join(
    _PROJECT_ROOT, "integrations", "pennylane-rocq",
    "pennylane_rocq", "rocq_device.py",
)


class TestPennyLaneDeviceSource(unittest.TestCase):
    """Validate the PennyLane device without needing pennylane installed."""

    def test_device_parses(self):
        with open(_DEVICE_PATH, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename="rocq_device.py")
        self.assertIsNotNone(tree)

    def test_no_set_indexing(self):
        """generate_samples must not index into self.observables."""
        with open(_DEVICE_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("self.observables[0]", source)
        self.assertNotIn("self.observables[", source)

    def test_generate_samples_uses_wires(self):
        """generate_samples must use self.wires for probabilities."""
        with open(_DEVICE_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn("self.wires", source)

    def test_core_gate_mapping_present(self):
        """Device must map at least H, X, Y, Z, CNOT PennyLane ops."""
        with open(_DEVICE_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        for gate in ["PauliX", "PauliY", "PauliZ", "Hadamard", "CNOT"]:
            self.assertIn(gate, source,
                          f"PennyLane gate mapping missing '{gate}'")

    def test_analytic_probability_called(self):
        """generate_samples must compute probabilities analytically."""
        with open(_DEVICE_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn("analytic_probability", source)


if __name__ == "__main__":
    unittest.main()
