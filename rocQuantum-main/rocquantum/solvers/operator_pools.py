"""Named operator-pool registry for CUDA-QX-style solver workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock

try:
    from rocq.operator import QuantumOperator
except ImportError:  # pragma: no cover - import-only environments.
    QuantumOperator = None  # type: ignore

from .stateprep import get_uccsd_operator_pool


_POOL_FACTORIES = {}
_POOL_LOCK = RLock()


def _pool_name(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("operator pool name must be a non-empty string.")
    return name.strip().lower()


def register_operator_pool(name, factory=None, *, replace: bool = False):
    """Register an operator-pool factory.

    Factories receive the keyword configuration supplied to
    :func:`get_operator_pool` and must return an iterable of
    ``rocq.operator.QuantumOperator`` instances.  An empty iterable is valid
    when the requested configuration has no admissible excitations.
    """

    normalized_name = _pool_name(name)
    if not isinstance(replace, bool):
        raise ValueError("replace must be a boolean.")

    def register(candidate):
        if not isinstance(candidate, Callable):
            raise ValueError("operator pool factory must be callable.")
        with _POOL_LOCK:
            if normalized_name in _POOL_FACTORIES and not replace:
                raise ValueError(
                    f"operator pool '{normalized_name}' is already registered."
                )
            _POOL_FACTORIES[normalized_name] = candidate
        return candidate

    if factory is None:
        return register
    return register(factory)


def operator_pool(name, *, replace: bool = False):
    """Decorator form of :func:`register_operator_pool`."""

    return register_operator_pool(name, replace=replace)


def get_available_operator_pools():
    """Return the currently registered pool names in deterministic order."""

    with _POOL_LOCK:
        return tuple(sorted(_POOL_FACTORIES))


def _pop_alias(config, aliases, label):
    present = [alias for alias in aliases if alias in config]
    if not present:
        raise ValueError(f"{label} is required for the uccsd operator pool.")
    if len(present) > 1:
        raise ValueError(
            f"{label} was provided more than once through aliases: "
            + ", ".join(present)
            + "."
        )
    return config.pop(present[0])


def _qaoa_factory(**config):
    from .qaoa import get_operator_pool as get_qaoa_operator_pool

    return get_qaoa_operator_pool("qaoa", **config)


def _uccsd_factory(**config):
    normalized = dict(config)
    num_qubits = _pop_alias(
        normalized,
        ("num_qubits", "n_qubits", "num-qubits", "n-qubits"),
        "num_qubits",
    )
    num_electrons = _pop_alias(
        normalized,
        ("num_electrons", "n_electrons", "num-electrons", "n-electrons"),
        "num_electrons",
    )
    spin = normalized.pop("spin", 0)
    if normalized:
        unexpected = ", ".join(sorted(str(key) for key in normalized))
        raise ValueError(
            f"unsupported uccsd operator-pool configuration: {unexpected}."
        )
    return get_uccsd_operator_pool(num_electrons, num_qubits, spin)


def _normalize_generated_pool(name: str, generated):
    if QuantumOperator is None:
        raise RuntimeError(
            "Canonical 'rocq' package is required to build operator pools."
        )
    if isinstance(generated, (str, bytes, Mapping)):
        raise TypeError(
            f"operator pool factory '{name}' must return an iterable of operators."
        )
    try:
        pool = list(generated)
    except TypeError as exc:
        raise TypeError(
            f"operator pool factory '{name}' must return an iterable of operators."
        ) from exc
    if any(not isinstance(operator, QuantumOperator) for operator in pool):
        raise TypeError(
            f"operator pool factory '{name}' returned a non-QuantumOperator value."
        )
    return pool


def get_operator_pool(name: str, **config):
    """Generate a registered operator pool from keyword configuration."""

    normalized_name = _pool_name(name)
    with _POOL_LOCK:
        factory = _POOL_FACTORIES.get(normalized_name)
    if factory is None:
        available = ", ".join(get_available_operator_pools())
        raise ValueError(
            f"unknown operator pool '{normalized_name}'; available pools: {available}."
        )
    return _normalize_generated_pool(normalized_name, factory(**config))


register_operator_pool("qaoa", _qaoa_factory)
register_operator_pool("uccsd", _uccsd_factory)


__all__ = [
    "get_available_operator_pools",
    "get_operator_pool",
    "operator_pool",
    "register_operator_pool",
]
