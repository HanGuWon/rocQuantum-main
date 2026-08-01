"""Validated QEC code metadata and extension-point registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np

from rocq.operator import PauliOperator

from .._gf2 import binary_matrix, positive_integer, readonly_copy


@dataclass(frozen=True)
class CodeMetadata:
    """Descriptive ``[[n, k, d]]`` metadata for a host-side QEC code."""

    name: str
    num_data_qubits: int
    num_logical_qubits: int
    distance: int
    num_ancilla_qubits: int
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Code metadata name must be a non-empty string.")
        if not isinstance(self.description, str):
            raise ValueError("Code metadata description must be a string.")
        n = positive_integer(self.num_data_qubits, "num_data_qubits")
        k = positive_integer(self.num_logical_qubits, "num_logical_qubits")
        distance = positive_integer(self.distance, "distance")
        if isinstance(self.num_ancilla_qubits, bool) or not isinstance(
            self.num_ancilla_qubits, (int, np.integer)
        ):
            raise ValueError("num_ancilla_qubits must be a non-negative integer.")
        ancillas = int(self.num_ancilla_qubits)
        if ancillas < 0:
            raise ValueError("num_ancilla_qubits must be non-negative.")
        if k > n:
            raise ValueError("num_logical_qubits cannot exceed num_data_qubits.")
        if distance > n:
            raise ValueError("distance cannot exceed num_data_qubits.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "num_data_qubits", n)
        object.__setattr__(self, "num_logical_qubits", k)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "num_ancilla_qubits", ancillas)


class Code:
    """Dense CSS-code metadata with strict GF(2) validation.

    ``Hx`` and ``Hz`` contain X- and Z-stabilizer supports respectively.  ``H``
    is exposed in binary symplectic ``[X | Z]`` form.  A code may expose up to
    one logical observable of each Pauli type per logical qubit.  When both
    families are complete, their canonical anticommutation pairing is checked;
    a basis-specific code may expose only one family.
    """

    def __init__(
        self,
        metadata: CodeMetadata,
        *,
        Hx: object,
        Hz: object,
        logical_x: object,
        logical_z: object,
    ) -> None:
        if not isinstance(metadata, CodeMetadata):
            raise ValueError("metadata must be a CodeMetadata instance.")
        n = metadata.num_data_qubits
        normalized_hx = binary_matrix(Hx, "Hx", columns=n)
        normalized_hz = binary_matrix(Hz, "Hz", columns=n)
        normalized_lx = binary_matrix(logical_x, "logical_x", columns=n)
        normalized_lz = binary_matrix(logical_z, "logical_z", columns=n)

        expected_logicals = metadata.num_logical_qubits
        if normalized_lx.shape[0] > expected_logicals:
            raise ValueError(
                "logical_x cannot contain more rows than encoded logical qubits."
            )
        if normalized_lz.shape[0] > expected_logicals:
            raise ValueError(
                "logical_z cannot contain more rows than encoded logical qubits."
            )
        actual_ancillas = normalized_hx.shape[0] + normalized_hz.shape[0]
        if metadata.num_ancilla_qubits != actual_ancillas:
            raise ValueError(
                "num_ancilla_qubits must equal the total number of Hx and Hz checks."
            )

        if np.any((normalized_hx @ normalized_hz.T) % 2):
            raise ValueError("Hx and Hz stabilizers must commute over GF(2).")
        if np.any((normalized_hz @ normalized_lx.T) % 2):
            raise ValueError("logical_x observables must commute with Hz stabilizers.")
        if np.any((normalized_hx @ normalized_lz.T) % 2):
            raise ValueError("logical_z observables must commute with Hx stabilizers.")

        has_complete_logical_pairing = (
            normalized_lx.shape[0] == expected_logicals
            and normalized_lz.shape[0] == expected_logicals
        )
        if has_complete_logical_pairing:
            logical_pairing = (normalized_lx @ normalized_lz.T) % 2
            expected_pairing = np.eye(expected_logicals, dtype=np.uint8)
            if not np.array_equal(logical_pairing, expected_pairing):
                raise ValueError(
                    "complete logical_x/logical_z families must have the canonical "
                    "GF(2) anticommutation matrix."
                )

        zeros_x = np.zeros((normalized_hx.shape[0], n), dtype=np.uint8)
        zeros_z = np.zeros((normalized_hz.shape[0], n), dtype=np.uint8)
        full_h = np.vstack(
            (
                np.hstack((normalized_hx, zeros_x)),
                np.hstack((zeros_z, normalized_hz)),
            )
        )
        pauli_observables = np.vstack(
            (
                np.hstack(
                    (
                        normalized_lx,
                        np.zeros((normalized_lx.shape[0], n), dtype=np.uint8),
                    )
                ),
                np.hstack(
                    (
                        np.zeros((normalized_lz.shape[0], n), dtype=np.uint8),
                        normalized_lz,
                    )
                ),
            )
        )

        self._metadata = metadata
        self._Hx = readonly_copy(normalized_hx)
        self._Hz = readonly_copy(normalized_hz)
        self._H = readonly_copy(full_h)
        self._logical_x = readonly_copy(normalized_lx)
        self._logical_z = readonly_copy(normalized_lz)
        self._pauli_observables = readonly_copy(pauli_observables)

    @property
    def metadata(self) -> CodeMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def distance(self) -> int:
        return self._metadata.distance

    @property
    def H(self) -> np.ndarray:
        return self._H.copy()

    @property
    def Hx(self) -> np.ndarray:
        return self._Hx.copy()

    @property
    def Hz(self) -> np.ndarray:
        return self._Hz.copy()

    @property
    def logical_x(self) -> np.ndarray:
        return self._logical_x.copy()

    @property
    def logical_z(self) -> np.ndarray:
        return self._logical_z.copy()

    @property
    def pauli_observables(self) -> np.ndarray:
        return self._pauli_observables.copy()

    def get_num_data_qubits(self) -> int:
        return self._metadata.num_data_qubits

    def get_num_logical_qubits(self) -> int:
        return self._metadata.num_logical_qubits

    def get_num_ancilla_x_qubits(self) -> int:
        return int(self._Hx.shape[0])

    def get_num_ancilla_z_qubits(self) -> int:
        return int(self._Hz.shape[0])

    def get_num_ancilla_qubits(self) -> int:
        return self._metadata.num_ancilla_qubits

    def get_num_x_stabilizers(self) -> int:
        return int(self._Hx.shape[0])

    def get_num_z_stabilizers(self) -> int:
        return int(self._Hz.shape[0])

    def get_parity(self) -> np.ndarray:
        return self.H

    def get_parity_x(self) -> np.ndarray:
        return self.Hx

    def get_parity_z(self) -> np.ndarray:
        return self.Hz

    def get_observables_x(self) -> np.ndarray:
        return self.logical_x

    def get_observables_z(self) -> np.ndarray:
        return self.logical_z

    def get_pauli_observables_matrix(self) -> np.ndarray:
        """Return full binary-symplectic observables, Lx stacked on Lz."""

        return self.pauli_observables

    @staticmethod
    def _pauli_operator(row: np.ndarray, pauli: str) -> PauliOperator:
        terms = [f"{pauli}{index}" for index, bit in enumerate(row) if bit]
        return PauliOperator(
            " ".join(terms) if terms else "I",
            num_qubits=int(row.size),
        )

    def get_stabilizers(self) -> List[PauliOperator]:
        """Return stabilizer generators as canonical rocq Pauli operators."""

        return [self._pauli_operator(row, "X") for row in self._Hx] + [
            self._pauli_operator(row, "Z") for row in self._Hz
        ]

    def get_logical_x_operators(self) -> Tuple[PauliOperator, ...]:
        return tuple(self._pauli_operator(row, "X") for row in self._logical_x)

    def get_logical_z_operators(self) -> Tuple[PauliOperator, ...]:
        return tuple(self._pauli_operator(row, "Z") for row in self._logical_z)


CodeFactory = Callable[..., Code]
_CODE_REGISTRY: Dict[str, CodeFactory] = {}
_FactoryT = TypeVar("_FactoryT", bound=CodeFactory)


def _registry_name(name: object, label: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} name must be a non-empty string.")
    return name.strip().lower()


def register_code(
    name: str,
    factory: Optional[_FactoryT] = None,
) -> Callable[[_FactoryT], _FactoryT]:
    """Register a code factory directly or as ``@register_code(name)``."""

    normalized_name = _registry_name(name, "code")

    def decorator(candidate: _FactoryT) -> _FactoryT:
        if not callable(candidate):
            raise ValueError("code factory must be callable.")
        if normalized_name in _CODE_REGISTRY:
            raise ValueError(f"QEC code '{normalized_name}' is already registered.")
        _CODE_REGISTRY[normalized_name] = candidate
        return candidate

    if factory is None:
        return decorator
    decorator(factory)
    return factory  # type: ignore[return-value]


def code(name: str) -> Callable[[_FactoryT], _FactoryT]:
    """Official-style decorator alias for :func:`register_code`."""

    return register_code(name)


def get_code(name: str, **kwargs: object) -> Code:
    normalized_name = _registry_name(name, "code")
    try:
        factory = _CODE_REGISTRY[normalized_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown QEC code '{normalized_name}'. Available codes: {get_available_codes()}."
        ) from exc
    instance = factory(**kwargs)
    if not isinstance(instance, Code):
        raise TypeError(
            f"Registered QEC code factory '{normalized_name}' must return a Code instance."
        )
    return instance


def get_available_codes() -> List[str]:
    return sorted(_CODE_REGISTRY)


__all__ = [
    "Code",
    "CodeMetadata",
    "code",
    "get_available_codes",
    "get_code",
    "register_code",
]
