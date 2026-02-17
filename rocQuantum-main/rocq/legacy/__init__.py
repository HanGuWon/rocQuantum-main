"""
rocq.legacy — Compatibility shim for deprecated import paths.

This module provides a bridge for legacy callers that used the old
``rocquantum.python.rocq`` import path. All public symbols are re-exported
from the canonical ``rocq`` package with a deprecation warning.

Usage (deprecated, will be removed in a future release)::

    from rocq.legacy import PauliOperator, kernel, execute

Preferred (canonical)::

    from rocq.operator import PauliOperator
    from rocq.kernel import kernel, execute
"""

import warnings as _warnings

_warnings.warn(
    "Importing from 'rocq.legacy' is deprecated and will be removed in "
    "a future release. Use the canonical 'rocq' package directly. "
    "Example: 'from rocq.operator import PauliOperator'",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical rocq
from rocq.operator import (
    QuantumOperator,
    PauliOperator,
    HermitianOperator,
    SumOperator,
    get_expectation_value,
)
from rocq.kernel import kernel, execute, QuantumKernel
from rocq.qvec import qvec
from rocq.noise import NoiseModel
from rocq.gates import (
    h, x, y, z, s, sdg, t,
    rx, ry, rz,
    cnot, cx, cz, swap,
    crx, cry, crz,
)
