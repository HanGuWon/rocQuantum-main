"""Deterministic single-error lookup-table decoder."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import Decoder, DecoderResult, register_decoder


@register_decoder("single_error_lut")
class SingleErrorLUTDecoder(Decoder):
    """Decode zero or one error when parity-check columns are distinguishable."""

    def __init__(self, H: object) -> None:
        super().__init__(H)
        matches: Dict[Tuple[int, ...], List[int]] = {}
        for column in range(self.block_size):
            key = tuple(int(bit) for bit in self._H[:, column])
            matches.setdefault(key, []).append(column)
        self._matches = {key: tuple(indices) for key, indices in matches.items()}

    def decode(self, syndrome: object) -> DecoderResult:
        _, hard = self._normalize_syndrome(syndrome)
        key = tuple(int(bit) for bit in hard)
        result = np.zeros(self.block_size, dtype=float)
        if not any(key):
            return DecoderResult(True, result)

        matches = self._matches.get(key, ())
        if len(matches) == 1:
            result[matches[0]] = 1.0
            return DecoderResult(True, result)
        if matches:
            result[list(matches)] = 1.0 / len(matches)
        return DecoderResult(False, result)


__all__ = ["SingleErrorLUTDecoder"]
