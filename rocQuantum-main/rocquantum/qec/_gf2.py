"""Strict host-side helpers for dense binary linear algebra over GF(2)."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Optional

import numpy as np


def positive_integer(value: object, name: str) -> int:
    """Return a normalized positive integer while rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def nonnegative_integer(value: object, name: str) -> int:
    """Return a normalized non-negative integer while rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative.")
    return normalized


def probability(value: object, name: str = "probability") -> float:
    """Validate a finite real probability in the closed interval [0, 1]."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [0, 1].")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{name} must be a finite real number in [0, 1].")
    return normalized


def normalize_seed(seed: object, name: str = "seed") -> Optional[int]:
    """Validate an optional non-negative NumPy RNG seed."""

    if seed is None:
        return None
    return nonnegative_integer(seed, name)


def binary_matrix(
    values: object,
    name: str,
    *,
    columns: Optional[int] = None,
    allow_empty_rows: bool = True,
) -> np.ndarray:
    """Return a C-contiguous uint8 rank-2 matrix containing only integer 0/1."""

    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a dense rank-2 binary matrix.") from exc
    if raw.ndim != 2:
        raise ValueError(f"{name} must be a dense rank-2 binary matrix.")
    if columns is not None and raw.shape[1] != columns:
        raise ValueError(f"{name} must have exactly {columns} columns.")
    if not allow_empty_rows and raw.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one parity check row.")

    normalized = np.empty(raw.shape, dtype=np.uint8)
    for index, value in np.ndenumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} entries must be integer GF(2) values 0 or 1.")
        bit = int(value)
        if bit not in (0, 1):
            raise ValueError(f"{name} entries must be integer GF(2) values 0 or 1.")
        normalized[index] = bit
    return np.ascontiguousarray(normalized)


def binary_vector(values: object, length: int, name: str) -> np.ndarray:
    """Return a strict rank-1 uint8 binary vector of the requested length."""

    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a length-{length} binary vector.") from exc
    if raw.ndim != 1 or raw.size != length:
        raise ValueError(f"{name} must be a length-{length} binary vector.")
    matrix = binary_matrix(raw.reshape(1, -1), name, columns=length)
    return matrix.reshape(-1)


def soft_probability_vector(values: object, length: int, name: str) -> np.ndarray:
    """Validate a finite rank-1 vector of soft probabilities in [0, 1]."""

    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a length-{length} vector of finite probabilities in [0, 1]."
        ) from exc
    if raw.ndim != 1 or raw.size != length:
        raise ValueError(
            f"{name} must be a length-{length} vector of finite probabilities in [0, 1]."
        )

    normalized = np.empty(length, dtype=float)
    for index, value in enumerate(raw):
        normalized[index] = probability(value, f"{name} entry")
    return normalized


def gf2_rank(matrix: object) -> int:
    """Compute exact row rank over GF(2) without floating-point elimination."""

    normalized = binary_matrix(matrix, "matrix")
    work = normalized.copy()
    rows, columns = work.shape
    pivot_row = 0
    for column in range(columns):
        candidates = np.flatnonzero(work[pivot_row:, column])
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            work[[pivot_row, selected]] = work[[selected, pivot_row]]
        for row in range(rows):
            if row != pivot_row and work[row, column]:
                work[row] ^= work[pivot_row]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def readonly_copy(values: np.ndarray) -> np.ndarray:
    """Return an owned, immutable array suitable for validated model storage."""

    result = np.array(values, copy=True, order="C")
    result.setflags(write=False)
    return result
