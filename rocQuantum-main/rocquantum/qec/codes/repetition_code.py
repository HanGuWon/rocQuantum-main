# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Concrete helpers for the 3-qubit bit-flip repetition code."""

from typing import Callable, Dict, List

import rocq
from rocq.operator import PauliOperator

from rocquantum.qec.framework import QuantumErrorCode, _validate_positive_integer


AnsatzKernel = Callable[..., None]


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
        if initial_state_kernel is not None and not callable(initial_state_kernel):
            raise ValueError("initial_state_kernel must be callable or None.")

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
