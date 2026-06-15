# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Concrete implementation of a decoder for the 3-qubit bit-flip repetition code.
"""

from typing import List

from rocq.operator import PauliOperator
from rocquantum.qec.framework import Decoder, _validate_binary_bit


def _validate_repetition_syndrome(syndrome: List[int]) -> List[int]:
    try:
        bits = list(syndrome)
    except TypeError as exc:
        raise ValueError("syndrome must be a length-2 sequence containing only 0 or 1.") from exc
    if len(bits) != 2:
        raise ValueError("syndrome must be a length-2 sequence containing only 0 or 1.")
    return [_validate_binary_bit(bit, "syndrome") for bit in bits]

class RepetitionCodeDecoder(Decoder):
    """
    A simple lookup-table decoder for the 3-qubit repetition code.
    """
    def decode(self, syndrome: List[int]) -> PauliOperator:
        """
        Decodes the 2-bit syndrome to find the location of a single X error.
        Syndrome bits correspond to [Z0Z1, Z1Z2] measurements.
        """
        syndrome = _validate_repetition_syndrome(syndrome)
        if syndrome == [0, 0]:
            # No error detected
            return PauliOperator("I")
        if syndrome == [1, 0]:
            # Error on data qubit 0
            return PauliOperator("X0")
        if syndrome == [1, 1]:
            # Error on data qubit 1
            return PauliOperator("X1")
        # Error on data qubit 2
        return PauliOperator("X2")
