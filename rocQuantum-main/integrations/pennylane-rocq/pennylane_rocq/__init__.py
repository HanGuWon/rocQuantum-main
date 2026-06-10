# pennylane_rocq/__init__.py
"""
This package provides the rocQuantum device for PennyLane.
"""
from .rocq_device import RocQDevice
from .roc_device import LightningRocmDevice, LightningRocqDevice, RocqDevice

__all__ = ["LightningRocmDevice", "LightningRocqDevice", "RocQDevice", "RocqDevice"]
