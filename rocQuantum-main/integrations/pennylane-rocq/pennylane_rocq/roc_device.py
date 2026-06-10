from __future__ import annotations

from .rocq_device import RocQDevice


class RocqDevice(RocQDevice):
    """Compatibility alias for the historical rocq.pennylane entry point."""

    short_name = "rocq.pennylane"
