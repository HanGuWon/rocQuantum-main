"""
P3 Integration Tests — Cirq Adapter

Pure-Python structural tests for the Cirq-rocm integration layer.
Skip if Cirq is not installed; but always validate the adapter source.

    python -m unittest tests.test_cirq_integration -v
"""

import ast
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_ADAPTER_PATH = os.path.join(
    _PROJECT_ROOT, "integrations", "cirq-rocm", "cirq_rocm",
    "roc_quantum_simulator.py",
)


class TestCirqAdapterSource(unittest.TestCase):
    """Validate the Cirq adapter without needing cirq installed."""

    def test_adapter_parses(self):
        with open(_ADAPTER_PATH, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename="roc_quantum_simulator.py")
        self.assertIsNotNone(tree)

    def test_gate_map_has_core_gates(self):
        with open(_ADAPTER_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        for gate in ["X", "Y", "Z", "H", "CNOT"]:
            self.assertIn(f'"{gate}"', source,
                          f"Gate map missing '{gate}'")

    def test_multi_qubit_gate_handling(self):
        """CNOT and CZ must pass multiple qubit indices to ApplyGate."""
        with open(_ADAPTER_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        # The gate map should contain multi-qubit entries
        self.assertIn("cirq.CNOT", source)
        self.assertIn("cirq.CZ", source)
        # ApplyGate call should unpack indices (not pass a list)
        self.assertIn("*indices", source,
                       "ApplyGate must unpack qubit indices for multi-qubit gates")

    def test_unsupported_gate_raises(self):
        """Adapter must raise TypeError for unsupported gates."""
        with open(_ADAPTER_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn("Unsupported gate", source)


if __name__ == "__main__":
    unittest.main()
