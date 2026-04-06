"""Canonical rocQuantum Python API."""

from .noise import NoiseModel
from .kernel import QuantumKernel, execute, kernel, observe, sample
from .operator import (
    HermitianOperator,
    PauliOperator,
    QuantumOperator,
    SumOperator,
    get_expectation_value,
)
from .qvec import qvec
from .gates import (
    cnot,
    crx,
    cry,
    crz,
    cx,
    cz,
    h,
    rx,
    ry,
    rz,
    s,
    sdg,
    swap,
    t,
    x,
    y,
    z,
)

__all__ = [
    "NoiseModel",
    "QuantumKernel",
    "QuantumOperator",
    "PauliOperator",
    "HermitianOperator",
    "SumOperator",
    "kernel",
    "execute",
    "sample",
    "observe",
    "get_expectation_value",
    "qvec",
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "rx",
    "ry",
    "rz",
    "cnot",
    "cx",
    "cz",
    "swap",
    "crx",
    "cry",
    "crz",
]
