# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Repetition-code metadata plus legacy 3-qubit circuit helpers."""

from typing import Callable, Dict, List

import numpy as np
import rocq
from rocq.operator import PauliOperator

from rocquantum.qec._gf2 import positive_integer
from rocquantum.qec.codes.base import Code, CodeMetadata, register_code
from rocquantum.qec.framework import (
    QuantumErrorCode,
    _validate_initial_state_kernel,
    _validate_positive_integer,
)


AnsatzKernel = Callable[..., None]


@register_code("repetition")
class RepetitionCode(Code):
    """Odd-distance Z-basis bit-flip repetition code.

    The dense ``Hz`` rows are adjacent ``Z_i Z_{i+1}`` checks.  The logical
    Z observable is represented canonically by the first data-qubit Z.  Like
    CUDA-QX's repetition code, this basis-specific code does not expose a
    logical X observable.
    """

    def __init__(self, distance: int = 3) -> None:
        normalized_distance = positive_integer(distance, "distance")
        if normalized_distance < 3 or normalized_distance % 2 == 0:
            raise ValueError("Repetition-code distance must be an odd integer >= 3.")

        hz = np.zeros(
            (normalized_distance - 1, normalized_distance), dtype=np.uint8
        )
        for row in range(normalized_distance - 1):
            hz[row, row : row + 2] = 1
        logical_x = np.zeros((0, normalized_distance), dtype=np.uint8)
        logical_z = np.zeros((1, normalized_distance), dtype=np.uint8)
        logical_z[0, 0] = 1

        super().__init__(
            CodeMetadata(
                name="repetition",
                num_data_qubits=normalized_distance,
                num_logical_qubits=1,
                distance=normalized_distance,
                num_ancilla_qubits=normalized_distance - 1,
                description="Odd-distance bit-flip repetition code.",
            ),
            Hx=np.zeros((0, normalized_distance), dtype=np.uint8),
            Hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
        )


class ThreeQubitRepetitionCode(QuantumErrorCode):
    """
    Experimental 3-qubit bit-flip repetition code.

    Data qubits are 0, 1, 2 and ancillas are 3, 4 by convention.
    """

    def generate_stabilizer_circuits(
        self,
        initial_state_kernel: AnsatzKernel,
        num_qubits: int,
        backend: str = "state_vector",
    ) -> List[object]:
        del backend
        num_qubits = _validate_positive_integer(num_qubits, "num_qubits")
        if num_qubits < 5:
            raise ValueError("ThreeQubitRepetitionCode requires at least 5 qubits.")
        initial_state_kernel = _validate_initial_state_kernel(initial_state_kernel)

        def apply_initial_state(q):
            if initial_state_kernel is not None:
                initial_state_kernel(q)

        @rocq.kernel
        def z0z1_stabilizer_kernel():
            q = rocq.qvec(num_qubits)
            apply_initial_state(q)
            rocq.cnot(q[0], q[3])
            rocq.cnot(q[1], q[3])

        @rocq.kernel
        def z1z2_stabilizer_kernel():
            q = rocq.qvec(num_qubits)
            apply_initial_state(q)
            rocq.cnot(q[1], q[4])
            rocq.cnot(q[2], q[4])

        return [z0z1_stabilizer_kernel, z1z2_stabilizer_kernel]

    def define_logical_operators(self) -> Dict[str, PauliOperator]:
        return {
            "logical_Z": PauliOperator("Z0"),
            "logical_X": PauliOperator("X0 X1 X2"),
        }


__all__ = ["RepetitionCode", "ThreeQubitRepetitionCode"]
