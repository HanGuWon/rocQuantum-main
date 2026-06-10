from __future__ import annotations

from .rocq_device import RocQDevice


class RocqDevice(RocQDevice):
    """Compatibility alias for the historical rocq.pennylane entry point."""

    short_name = "rocq.pennylane"


class LightningRocqDevice(RocQDevice):
    """PennyLane Lightning-style AMD GPU entry point backed by rocQuantum."""

    name = "rocQuantum Lightning-compatible AMD GPU Device"
    short_name = "lightning.rocq"


class LightningRocmDevice(LightningRocqDevice):
    """Alias matching the ROCm platform name for Lightning-style discovery."""

    short_name = "lightning.rocm"
