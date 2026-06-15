"""Statevector contract tests for canonical advanced gates.

These tests intentionally use the explicit Python mock backend so they can run
without AMD GPU hardware. They prove local statevector semantics only; native
ROCm performance and device dispatch still require the ROCm CI lane.
"""

from __future__ import annotations

import math
import os
import sys
from unittest import mock

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rocq.backends import MockBackendWarning, StateVectorBackend
from rocq.kernel import GateOp


def _mock_statevector_backend(num_qubits: int) -> StateVectorBackend:
    with mock.patch("rocq.backends.hip_backend", None):
        with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
            with pytest.warns(MockBackendWarning, match="Python mock fallback"):
                return StateVectorBackend(num_qubits)


def test_mock_statevector_executes_phase_and_tdg_gates():
    backend = _mock_statevector_backend(1)
    phase = math.pi / 3.0

    backend.run_ops([GateOp("h", [0], {}), GateOp("p", [0], {"phi": phase})])

    expected = np.array(
        [1.0 / math.sqrt(2.0), complex(math.cos(phase), math.sin(phase)) / math.sqrt(2.0)],
        dtype=np.complex64,
    )
    np.testing.assert_allclose(backend.get_state(), expected, atol=1e-6)

    tdg_backend = _mock_statevector_backend(1)
    tdg_backend.run_ops([GateOp("x", [0], {}), GateOp("tdg", [0], {})])

    expected_tdg = np.array(
        [0.0, complex(math.cos(-math.pi / 4.0), math.sin(-math.pi / 4.0))],
        dtype=np.complex64,
    )
    np.testing.assert_allclose(tdg_backend.get_state(), expected_tdg, atol=1e-6)


def test_mock_statevector_executes_controlled_phase_and_rotation_gates():
    cp_backend = _mock_statevector_backend(2)
    phase = math.pi / 5.0

    cp_backend.run_ops(
        [
            GateOp("x", [0], {}),
            GateOp("x", [1], {}),
            GateOp("cp", [0, 1], {"phi": phase}),
        ]
    )

    expected_cp = np.zeros(4, dtype=np.complex64)
    expected_cp[3] = complex(math.cos(phase), math.sin(phase))
    np.testing.assert_allclose(cp_backend.get_state(), expected_cp, atol=1e-6)

    crx_backend = _mock_statevector_backend(2)
    crx_backend.run_ops([GateOp("x", [0], {}), GateOp("crx", [0, 1], {"theta": math.pi})])

    expected_crx = np.zeros(4, dtype=np.complex64)
    expected_crx[3] = -1j
    np.testing.assert_allclose(crx_backend.get_state(), expected_crx, atol=1e-6)


def test_mock_statevector_executes_toffoli_and_cswap_gates():
    ccx_backend = _mock_statevector_backend(3)
    ccx_backend.run_ops(
        [
            GateOp("x", [0], {}),
            GateOp("x", [1], {}),
            GateOp("ccx", [0, 1, 2], {}),
        ]
    )

    expected_ccx = np.zeros(8, dtype=np.complex64)
    expected_ccx[7] = 1.0
    np.testing.assert_allclose(ccx_backend.get_state(), expected_ccx, atol=1e-6)

    cswap_backend = _mock_statevector_backend(3)
    cswap_backend.run_ops(
        [
            GateOp("x", [0], {}),
            GateOp("x", [1], {}),
            GateOp("cswap", [0, 1, 2], {}),
        ]
    )

    expected_cswap = np.zeros(8, dtype=np.complex64)
    expected_cswap[5] = 1.0
    np.testing.assert_allclose(cswap_backend.get_state(), expected_cswap, atol=1e-6)
