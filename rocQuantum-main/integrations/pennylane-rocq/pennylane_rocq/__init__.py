# pennylane_rocq/__init__.py
"""
This package provides the rocQuantum device for PennyLane.
"""
from .rocq_device import RocQDevice
from .roc_device import RocqDevice

__all__ = ["RocQDevice", "RocqDevice"]
