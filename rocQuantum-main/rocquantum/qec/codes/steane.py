"""Dense CSS metadata for the seven-qubit Steane code."""

from __future__ import annotations

import numpy as np

from .base import Code, CodeMetadata, register_code


@register_code("steane")
class SteaneCode(Code):
    """The canonical ``[[7, 1, 3]]`` self-dual Steane CSS code."""

    def __init__(self) -> None:
        parity = np.asarray(
            (
                (1, 1, 1, 1, 0, 0, 0),
                (0, 1, 1, 0, 1, 1, 0),
                (0, 0, 1, 1, 0, 1, 1),
            ),
            dtype=np.uint8,
        )
        logical = np.asarray(((0, 0, 0, 0, 1, 1, 1),), dtype=np.uint8)
        super().__init__(
            CodeMetadata(
                name="steane",
                num_data_qubits=7,
                num_logical_qubits=1,
                distance=3,
                num_ancilla_qubits=6,
                description="Self-dual seven-qubit Steane CSS code.",
            ),
            Hx=parity,
            Hz=parity,
            logical_x=logical,
            logical_z=logical,
        )


__all__ = ["SteaneCode"]
