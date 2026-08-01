"""Pure-NumPy code-capacity sampling helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._gf2 import (
    binary_matrix,
    normalize_seed,
    positive_integer,
    probability,
)
from .codes.base import Code


def generate_random_bit_flips(
    num_bits: int,
    error_probability: object,
    seed: object = None,
) -> np.ndarray:
    """Generate one reproducible dense uint8 Bernoulli error vector."""

    normalized_bits = positive_integer(num_bits, "num_bits")
    normalized_probability = probability(error_probability, "error_probability")
    normalized_seed = normalize_seed(seed)
    generator = np.random.default_rng(normalized_seed)
    return (generator.random(normalized_bits) < normalized_probability).astype(
        np.uint8
    )


def _capacity_check_matrix(H_or_Code: object) -> np.ndarray:
    if isinstance(H_or_Code, Code):
        # Match the CUDA-QX code overload: a complete Code samples its full
        # binary-symplectic H, while callers can pass Hx/Hz explicitly for a
        # one-Pauli-channel code-capacity experiment.
        parity = H_or_Code.H
        return binary_matrix(
            parity,
            "H_or_Code parity matrix",
            allow_empty_rows=False,
        )
    normalized = binary_matrix(
        H_or_Code,
        "H_or_Code",
        allow_empty_rows=False,
    )
    if normalized.shape[1] == 0:
        raise ValueError("H_or_Code must contain at least one error-variable column.")
    return normalized


def sample_code_capacity(
    H_or_Code: object,
    num_shots: int,
    error_probability: object,
    seed: object = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample independent bit errors and their exact GF(2) syndromes.

    Returns ``(syndromes, errors)`` with shapes ``(num_shots, checks)`` and
    ``(num_shots, variables)`` respectively.  Both arrays use dense ``uint8``.
    Passing a :class:`Code` uses its full binary-symplectic parity matrix;
    pass ``code.Hx`` or ``code.Hz`` for a single Pauli error channel.
    """

    parity = _capacity_check_matrix(H_or_Code)
    normalized_shots = positive_integer(num_shots, "num_shots")
    normalized_probability = probability(error_probability, "error_probability")
    normalized_seed = normalize_seed(seed)
    generator = np.random.default_rng(normalized_seed)
    errors = (
        generator.random((normalized_shots, parity.shape[1]))
        < normalized_probability
    ).astype(np.uint8)
    syndromes = ((errors @ parity.T) % 2).astype(np.uint8)
    return syndromes, errors


__all__ = ["generate_random_bit_flips", "sample_code_capacity"]
