"""Validated noise channels and the canonical rocq noise model."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def _normalize_probability(probability: float) -> float:
    if isinstance(probability, bool) or not isinstance(probability, Real):
        raise ValueError("Probability must be between 0 and 1.")
    probability = float(probability)
    if not math.isfinite(probability) or not (0 <= probability <= 1):
        raise ValueError("Probability must be between 0 and 1.")
    return probability


def _normalize_qubits(on_qubits):
    if on_qubits is None:
        return None
    if isinstance(on_qubits, (str, bytes)):
        raise TypeError("on_qubits must be an integer index or a sequence of integer indices.")
    if isinstance(on_qubits, Integral) and not isinstance(on_qubits, bool):
        raw_qubits = [on_qubits]
    else:
        try:
            raw_qubits = list(on_qubits)
        except TypeError as exc:
            raise TypeError(
                "on_qubits must be an integer index or a sequence of integer indices."
            ) from exc

    if not raw_qubits:
        raise ValueError("on_qubits must include at least one qubit.")

    normalized = []
    for qubit in raw_qubits:
        if isinstance(qubit, bool) or not isinstance(qubit, Integral):
            raise ValueError("on_qubits must contain integer qubit indices.")
        index = int(qubit)
        if index < 0:
            raise ValueError("on_qubits must contain non-negative qubit indices.")
        normalized.append(index)

    if len(set(normalized)) != len(normalized):
        raise ValueError("on_qubits must contain unique qubit indices.")
    return normalized


def _normalize_optional_op(after_op):
    if after_op is None:
        return None
    if not isinstance(after_op, str) or not after_op.strip():
        raise ValueError("after_op must be a non-empty string when provided.")
    return after_op.strip().lower()


def _normalize_kraus_matrices(operators) -> np.ndarray:
    if isinstance(operators, np.ndarray):
        raw_operators = operators
    else:
        try:
            raw_operators = list(operators)
        except TypeError as exc:
            raise TypeError("KrausChannel requires a sequence of square matrices.") from exc

    try:
        matrices = np.asarray(raw_operators, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("Kraus operators must form a rectangular numeric array.") from exc
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis, :, :]
    if matrices.ndim != 3 or matrices.shape[0] == 0:
        raise ValueError("KrausChannel requires one or more square matrices.")
    if matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Kraus operators must be square matrices of equal shape.")

    dimension = matrices.shape[1]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("Kraus operator dimension must be a positive power of two.")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("Kraus operators must contain only finite values.")

    completeness = sum(
        (matrix.conj().T @ matrix for matrix in matrices),
        start=np.zeros((dimension, dimension), dtype=np.complex128),
    )
    if not np.allclose(
        completeness,
        np.eye(dimension, dtype=np.complex128),
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError("Kraus operators must satisfy sum(K^\u2020 K) = I.")

    normalized = np.array(matrices, dtype=np.complex128, order="C", copy=True)
    normalized.setflags(write=False)
    return normalized


class KrausChannel:
    """A validated completely-positive trace-preserving Kraus channel."""

    def __init__(self, operators):
        self._operators = _normalize_kraus_matrices(operators)

    def __len__(self) -> int:
        return self._operators.shape[0]

    def __iter__(self):
        return iter(self._operators)

    def __getitem__(self, index):
        return self._operators[index]

    @property
    def num_qubits(self) -> int:
        return int(math.log2(self._operators.shape[1]))

    @property
    def kraus_matrices(self) -> np.ndarray:
        """Return a defensive copy of the normalized Kraus matrices."""

        return np.array(self._operators, copy=True)

    def _as_backend_spec(self, *, qubits, after_op):
        return {
            "type": "kraus",
            "prob": 1.0,
            "qubits": qubits,
            "op": after_op,
            "kraus_matrices": self.kraus_matrices,
        }


class _ProbabilityKrausChannel(KrausChannel):
    _backend_type = None

    def __init__(self, probability, operators):
        self.probability = _normalize_probability(probability)
        super().__init__(operators)

    def _as_backend_spec(self, *, qubits, after_op):
        if self._backend_type is None:
            return super()._as_backend_spec(qubits=qubits, after_op=after_op)
        return {
            "type": self._backend_type,
            "prob": self.probability,
            "qubits": qubits,
            "op": after_op,
            "kraus_matrices": None,
        }


_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class BitFlipChannel(_ProbabilityKrausChannel):
    _backend_type = "bit_flip"

    def __init__(self, probability):
        p = _normalize_probability(probability)
        super().__init__(p, [math.sqrt(1 - p) * _I, math.sqrt(p) * _X])


class PhaseFlipChannel(_ProbabilityKrausChannel):
    _backend_type = "phase_flip"

    def __init__(self, probability):
        p = _normalize_probability(probability)
        super().__init__(p, [math.sqrt(1 - p) * _I, math.sqrt(p) * _Z])


class DepolarizationChannel(_ProbabilityKrausChannel):
    _backend_type = "depolarizing"

    def __init__(self, probability):
        p = _normalize_probability(probability)
        pauli_scale = math.sqrt(p / 3)
        super().__init__(
            p,
            [
                math.sqrt(1 - p) * _I,
                pauli_scale * _X,
                pauli_scale * _Y,
                pauli_scale * _Z,
            ],
        )


class AmplitudeDampingChannel(_ProbabilityKrausChannel):
    _backend_type = "amplitude_damping"

    def __init__(self, probability):
        p = _normalize_probability(probability)
        super().__init__(
            p,
            [
                np.array([[1, 0], [0, math.sqrt(1 - p)]], dtype=np.complex128),
                np.array([[0, math.sqrt(p)], [0, 0]], dtype=np.complex128),
            ],
        )


class PhaseDampingChannel(_ProbabilityKrausChannel):
    """Phase damping normalized to the backend's generic Kraus path."""

    def __init__(self, probability):
        p = _normalize_probability(probability)
        super().__init__(
            p,
            [
                np.array([[1, 0], [0, math.sqrt(1 - p)]], dtype=np.complex128),
                np.array([[0, 0], [0, math.sqrt(p)]], dtype=np.complex128),
            ],
        )


# CUDA-Q exposes this built-in as ``PhaseDamping``; retain the explicit
# ``Channel`` spelling for consistency with the other public classes.
PhaseDamping = PhaseDampingChannel


class NoiseModel:
    """A declarative collection of backend-normalized noise channels."""

    def __init__(self):
        self._channels = []

    def _append_channel_object(self, channel, *, on_qubits, after_op):
        if not isinstance(channel, KrausChannel):
            raise TypeError("channel must be a KrausChannel instance.")
        normalized_qubits = _normalize_qubits(on_qubits)
        if (
            normalized_qubits is not None
            and len(normalized_qubits) != channel.num_qubits
        ):
            raise ValueError(
                f"A {channel.num_qubits}-qubit KrausChannel requires exactly "
                f"{channel.num_qubits} target qubit(s)."
            )
        channel_spec = channel._as_backend_spec(
            qubits=normalized_qubits,
            after_op=_normalize_optional_op(after_op),
        )
        self._channels.append(channel_spec)

    def add_channel(
        self,
        channel_type,
        probability=None,
        on_qubits=None,
        after_op=None,
        kraus_matrices=None,
    ):
        """Add a legacy channel spec or a CUDA-Q-style channel object.

        Supported forms are::

            add_channel("bit_flip", 0.01, on_qubits=[0], after_op="x")
            add_channel("x", [0], BitFlipChannel(0.01))
            add_channel(KrausChannel(...), on_qubits=[0], after_op="x")
        """

        if isinstance(on_qubits, KrausChannel):
            if after_op is not None or kraus_matrices is not None:
                raise TypeError(
                    "CUDA-Q-style add_channel(operation, qubits, channel) "
                    "does not accept after_op or kraus_matrices."
                )
            self._append_channel_object(
                on_qubits,
                on_qubits=probability,
                after_op=channel_type,
            )
            return

        if isinstance(channel_type, KrausChannel):
            if probability is not None or kraus_matrices is not None:
                raise TypeError(
                    "Object-style add_channel does not accept probability or kraus_matrices."
                )
            self._append_channel_object(
                channel_type,
                on_qubits=on_qubits,
                after_op=after_op,
            )
            return

        probability = _normalize_probability(probability)
        if not isinstance(channel_type, str) or not channel_type.strip():
            raise ValueError("channel_type must be a non-empty string.")
        channel_lower = channel_type.strip().lower()
        if channel_lower == "kraus" and kraus_matrices is None:
            raise ValueError("Kraus noise channels require kraus_matrices.")
        if channel_lower != "kraus" and kraus_matrices is not None:
            raise ValueError(
                "kraus_matrices may only be supplied for 'kraus' noise channels."
            )

        channel_spec = {
            "type": channel_lower,
            "prob": probability,
            "qubits": _normalize_qubits(on_qubits),
            "op": _normalize_optional_op(after_op),
            "kraus_matrices": kraus_matrices,
        }
        self._channels.append(channel_spec)

    def add_all_qubit_channel(self, after_op, channel):
        """Apply ``channel`` after every matching operation on its gate qubits."""

        self._append_channel_object(channel, on_qubits=None, after_op=after_op)

    def get_channels(self):
        """Return the mutable legacy channel-spec list for backend compatibility."""

        return self._channels


__all__ = [
    "NoiseModel",
    "KrausChannel",
    "BitFlipChannel",
    "PhaseFlipChannel",
    "DepolarizationChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
    "PhaseDamping",
]
