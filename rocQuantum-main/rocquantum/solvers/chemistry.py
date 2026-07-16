"""Host chemistry primitives compatible with the CUDA-QX solver surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Number, Real

import numpy as np

try:
    from rocq.operator import (
        PauliOperator,
        QuantumOperator,
        SumOperator,
        iter_pauli_terms,
    )
except ImportError:  # pragma: no cover - import-only environments.
    PauliOperator = None  # type: ignore
    QuantumOperator = None  # type: ignore
    SumOperator = None  # type: ignore
    iter_pauli_terms = None  # type: ignore


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_tolerance(value) -> float:
    tolerance = _finite_real(value, "tolerance")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    return tolerance


def _complex_array(values, name: str, rank: int) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric rank-{rank} tensor.") from exc
    if raw.ndim != rank:
        raise ValueError(f"{name} must be a rank-{rank} tensor.")
    normalized = []
    for value in raw.reshape(-1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise ValueError(f"{name} must contain finite numeric values.")
        scalar = complex(value)
        if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
            raise ValueError(f"{name} must contain finite values.")
        normalized.append(scalar)
    return np.asarray(normalized, dtype=np.complex128).reshape(raw.shape)


def _normalize_integrals(hpq, hpqrs=None):
    first = np.asarray(hpq, dtype=object)
    if hpqrs is None and first.ndim == 4:
        two_body = _complex_array(hpq, "hpqrs", 4)
        if len(set(two_body.shape)) != 1:
            raise ValueError("hpqrs dimensions must all have the same size.")
        dimension = int(two_body.shape[0])
        one_body = np.zeros((dimension, dimension), dtype=np.complex128)
        return one_body, two_body

    one_body = _complex_array(hpq, "hpq", 2)
    if one_body.shape[0] != one_body.shape[1] or one_body.shape[0] <= 0:
        raise ValueError("hpq must be a non-empty square tensor.")
    dimension = int(one_body.shape[0])
    if hpqrs is None:
        two_body = np.zeros(
            (dimension, dimension, dimension, dimension),
            dtype=np.complex128,
        )
    else:
        two_body = _complex_array(hpqrs, "hpqrs", 4)
        if two_body.shape != (dimension, dimension, dimension, dimension):
            raise ValueError("hpqrs shape must be (n, n, n, n) matching hpq.")
    return one_body, two_body


def _jw_ladder(index: int, *, creation: bool):
    z_product = PauliOperator("I")
    for qubit in range(index):
        z_product = z_product * PauliOperator(f"Z{qubit}")
    sign = -1.0j if creation else 1.0j
    return 0.5 * z_product * (
        PauliOperator(f"X{index}") + sign * PauliOperator(f"Y{index}")
    )


def _canonical_pauli_sum(operator, tolerance: float):
    combined = {}
    for coefficient, paulis in iter_pauli_terms(operator):
        key = tuple(sorted((str(pauli).upper(), int(qubit)) for pauli, qubit in paulis))
        combined[key] = combined.get(key, 0.0j) + complex(coefficient)

    terms = []
    for paulis, coefficient in sorted(combined.items()):
        if abs(coefficient.real) < tolerance and abs(coefficient.imag) < tolerance:
            continue
        if abs(coefficient.imag) < tolerance:
            coefficient = complex(coefficient.real, 0.0)
        word = "I" if not paulis else " ".join(
            f"{pauli}{qubit}" for pauli, qubit in paulis
        )
        terms.append(PauliOperator(word, coefficient=coefficient))

    if not terms:
        return PauliOperator("I", coefficient=0.0)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(terms)


def jordan_wigner(
    hpq,
    hpqrs=None,
    core_energy: float = 0.0,
    *,
    tolerance: float = 1.0e-15,
    tol=None,
):
    """Transform precomputed one-/two-body integrals to a Pauli Hamiltonian.

    This matches CUDA-QX's integral convention directly:
    ``sum hpq[p,q] a†_p a_q + sum hpqrs[p,q,r,s] a†_p a†_q a_r a_s``.
    Any conventional one-half factor must already be included in ``hpqrs``.
    """

    if PauliOperator is None:
        raise RuntimeError("Canonical 'rocq' package is required for jordan_wigner().")
    if isinstance(hpqrs, Number):
        if core_energy != 0.0:
            raise ValueError("core_energy was provided more than once.")
        core_energy = hpqrs
        hpqrs = None
    one_body, two_body = _normalize_integrals(hpq, hpqrs)
    core_energy = _finite_real(core_energy, "core_energy")
    normalized_tolerance = _positive_tolerance(tolerance)
    if tol is not None:
        alias_tolerance = _positive_tolerance(tol)
        if tolerance != 1.0e-15 and alias_tolerance != normalized_tolerance:
            raise ValueError("tolerance and tol must agree when both are provided.")
        normalized_tolerance = alias_tolerance
    tolerance = normalized_tolerance
    dimension = int(one_body.shape[0])

    creation = [_jw_ladder(index, creation=True) for index in range(dimension)]
    annihilation = [_jw_ladder(index, creation=False) for index in range(dimension)]
    hamiltonian = PauliOperator("I", coefficient=core_energy)

    for p in range(dimension):
        for q in range(dimension):
            coefficient = one_body[p, q]
            if abs(coefficient.real) < tolerance and abs(coefficient.imag) < tolerance:
                continue
            hamiltonian = hamiltonian + coefficient * creation[p] * annihilation[q]

    for p in range(dimension):
        for q in range(dimension):
            for r in range(dimension):
                for s in range(dimension):
                    coefficient = two_body[p, q, r, s]
                    if (
                        abs(coefficient.real) < tolerance
                        and abs(coefficient.imag) < tolerance
                    ):
                        continue
                    hamiltonian = (
                        hamiltonian
                        + coefficient
                        * creation[p]
                        * creation[q]
                        * annihilation[r]
                        * annihilation[s]
                    )
    return _canonical_pauli_sum(hamiltonian, tolerance)


@dataclass(frozen=True)
class MolecularHamiltonian:
    """Container for a Pauli Hamiltonian and its precomputed integrals."""

    hamiltonian: object
    hpq: np.ndarray
    hpqrs: np.ndarray
    n_electrons: int
    n_orbitals: int
    energies: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if QuantumOperator is not None and not isinstance(
            self.hamiltonian, QuantumOperator
        ):
            raise ValueError("hamiltonian must be a rocq.operator.QuantumOperator.")
        hpq, hpqrs = _normalize_integrals(self.hpq, self.hpqrs)
        for name, value in (
            ("n_electrons", self.n_electrons),
            ("n_orbitals", self.n_orbitals),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a non-negative integer.")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not isinstance(self.energies, Mapping):
            raise ValueError("energies must be a string-to-finite-real mapping.")
        energies = {}
        for key, value in self.energies.items():
            if not isinstance(key, str) or not key:
                raise ValueError("energies keys must be non-empty strings.")
            energies[key] = _finite_real(value, f"energies['{key}']")
        hpq.setflags(write=False)
        hpqrs.setflags(write=False)
        object.__setattr__(self, "hpq", hpq)
        object.__setattr__(self, "hpqrs", hpqrs)
        object.__setattr__(self, "n_electrons", int(self.n_electrons))
        object.__setattr__(self, "n_orbitals", int(self.n_orbitals))
        object.__setattr__(self, "energies", energies)

    @classmethod
    def from_integrals(
        cls,
        hpq,
        hpqrs=None,
        core_energy: float = 0.0,
        *,
        n_electrons: int = 0,
        n_orbitals=None,
        energies=None,
        tolerance: float = 1.0e-15,
    ):
        one_body, two_body = _normalize_integrals(hpq, hpqrs)
        if n_orbitals is None:
            n_orbitals = int(one_body.shape[0])
        operator = jordan_wigner(
            one_body,
            two_body,
            core_energy,
            tolerance=tolerance,
        )
        return cls(
            operator,
            one_body,
            two_body,
            n_electrons,
            n_orbitals,
            {} if energies is None else energies,
        )


def create_molecule(*args, **kwargs):
    """Fail closed for the not-yet-ported geometry/PySCF driver workflow."""

    raise NotImplementedError(
        "geometry/PySCF molecule construction is not implemented; provide "
        "precomputed hpq/hpqrs integrals to MolecularHamiltonian.from_integrals()."
    )
