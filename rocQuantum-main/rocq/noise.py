# Task 1: High-Level NoiseModel Class
import math
from numbers import Integral, Real


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
            raise TypeError("on_qubits must be an integer index or a sequence of integer indices.") from exc

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
    return after_op.lower()


class NoiseModel:
    """A declarative noise model for specifying noise channels.

    This object collects specifications for noise channels that will be applied
    during the execution of a quantum kernel on a compatible backend (e.g.,
    the 'density_matrix' backend).

    Attributes:
        _channels (list): A private list storing the configuration of each
            added noise channel.

    Usage Example:
        >>> noise_model = rocq.NoiseModel()
        >>> # Add a 1% depolarizing channel after any gate on qubits 0 and 1.
        >>> noise_model.add_channel('depolarizing', 0.01, on_qubits=[0, 1])
        >>> # Add a 0.5% bit-flip channel specifically after CNOT gates.
        >>> noise_model.add_channel('bit_flip', 0.005, after_op='cnot')
    """

    def __init__(self):
        """Initializes an empty noise model."""
        self._channels = []

    def add_channel(
        self,
        channel_type: str,
        probability: float,
        on_qubits=None,
        after_op: str = None,
        kraus_matrices=None,
    ):
        """Adds a noise channel to the model.

        Args:
            channel_type (str): The type of noise (e.g., 'depolarizing',
                'bit_flip', or 'kraus'). This must match a channel supported
                by the target backend.
            probability (float): The probability of the noise occurring. Must
                be between 0.0 and 1.0.
            on_qubits (Optional[list[int]]): A specific list of qubits to
                apply the noise to. If None, the noise is applied to all
                qubits involved in the preceding gate operation. Defaults to None.
            after_op (Optional[str]): Apply noise only after a specific gate
                type (e.g., 'cnot'). If None, the noise is applied after any
                gate. Defaults to None.
            kraus_matrices: Required when channel_type is 'kraus'. The backend
                interprets this as shape (num_kraus, 2**len(on_qubits),
                2**len(on_qubits)).
        """
        probability = _normalize_probability(probability)

        if not isinstance(channel_type, str) or not channel_type.strip():
            raise ValueError("channel_type must be a non-empty string.")
        channel_lower = channel_type.lower()
        if channel_lower == "kraus" and kraus_matrices is None:
            raise ValueError("Kraus noise channels require kraus_matrices.")
        if channel_lower != "kraus" and kraus_matrices is not None:
            raise ValueError("kraus_matrices may only be supplied for 'kraus' noise channels.")

        channel_spec = {
            "type": channel_lower,
            "prob": probability,
            "qubits": _normalize_qubits(on_qubits),
            "op": _normalize_optional_op(after_op),
            "kraus_matrices": kraus_matrices,
        }
        self._channels.append(channel_spec)
        print(f"NOISE_MODEL: Added '{channel_type}' channel with prob={probability}.")

    def get_channels(self):
        """Returns the list of configured noise channels."""
        return self._channels
