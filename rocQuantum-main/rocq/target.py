"""Context-local target selection for the canonical :mod:`rocq` runtime.

Targets in this module describe the local execution backend selected by the
high-level runtime.  They do not imply native multi-QPU scheduling or hardware
availability; every currently registered target exposes one logical QPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union


@dataclass(frozen=True)
class Target:
    """A named local runtime target.

    ``backend`` is the canonical backend name understood by
    :func:`rocq.backends.get_backend`.  ``num_qpus()`` deliberately returns
    one for the current local runtime; it is capability metadata, not a
    hardware discovery result.  The remaining metadata mirrors the commonly
    inspected CUDA-Q target surface while retaining rocQuantum's existing
    ``name`` and ``backend`` attributes.
    """

    name: str
    backend: str
    _num_qpus: int = 1
    description: str = ""
    precision: str = "fp32"
    simulator: str = ""
    remote: bool = False
    emulated: bool = False
    platform: str = "default"

    def num_qpus(self) -> int:
        """Return the number of logical QPUs exposed by this target."""

        return self._num_qpus

    def is_remote(self) -> bool:
        """Return whether execution is delegated to a remote service."""

        return self.remote

    def is_emulated(self) -> bool:
        """Return whether a physical target is running in emulation mode."""

        return self.emulated

    def get_precision(self) -> str:
        """Return the simulator precision as ``"fp32"`` or ``"fp64"``."""

        return self.precision


_TARGETS: Dict[str, Target] = {
    "state_vector": Target(
        "state_vector",
        "state_vector",
        description="Native ROCm state-vector simulator backed by hipStateVec.",
        precision="fp32",
        simulator="hipstatevec",
    ),
    "qpp-cpu": Target(
        "qpp-cpu",
        "qpp-cpu",
        description="NumPy CPU reference state-vector simulator.",
        precision="fp64",
        simulator="numpy",
    ),
    "density_matrix": Target(
        "density_matrix",
        "density_matrix",
        description="Native ROCm density-matrix simulator.",
        precision="fp32",
        simulator="rocq-hip-density-matrix",
    ),
    "stabilizer": Target(
        "stabilizer",
        "stabilizer",
        description="Local Clifford stabilizer simulator.",
        precision="fp64",
        simulator="rocq-stabilizer",
    ),
    "tableau": Target(
        "tableau",
        "tableau",
        description="Alias for the local Clifford stabilizer simulator.",
        precision="fp64",
        simulator="rocq-stabilizer",
    ),
    "clifford": Target(
        "clifford",
        "clifford",
        description="Alias for the local Clifford stabilizer simulator.",
        precision="fp64",
        simulator="rocq-stabilizer",
    ),
}
_DEFAULT_TARGET = _TARGETS["state_vector"]
_ACTIVE_TARGET: ContextVar[Target] = ContextVar(
    "rocq_active_target", default=_DEFAULT_TARGET
)


def _validate_target_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("target name must be a non-empty string.")
    return name.strip().lower()


def get_targets() -> List[Target]:
    """Return registered target objects in stable order."""

    return list(_TARGETS.values())


def has_target(name: str) -> bool:
    """Return whether ``name`` identifies a registered target."""

    try:
        normalized = _validate_target_name(name)
    except ValueError:
        return False
    return normalized in _TARGETS


def get_target(name: Optional[str] = None) -> Target:
    """Return the active target, or a registered target by name."""

    if name is None:
        return _ACTIVE_TARGET.get()
    normalized = _validate_target_name(name)
    try:
        return _TARGETS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown target '{name}'. Available targets are: {list(_TARGETS)}."
        ) from exc


def _resolve_target(target_or_name: Union[str, Target]) -> Target:
    if isinstance(target_or_name, Target):
        registered = get_target(target_or_name.name)
        if registered != target_or_name:
            raise ValueError(
                "Target objects passed to set_target() must be registered targets."
            )
        return registered
    return get_target(target_or_name)


def set_target(target_or_name: Union[str, Target]) -> None:
    """Set a registered target for the current context."""

    selected = _resolve_target(target_or_name)
    _ACTIVE_TARGET.set(selected)


def reset_target() -> None:
    """Restore the default ``state_vector`` target."""

    _ACTIVE_TARGET.set(_DEFAULT_TARGET)


def num_qpus() -> int:
    """Return the active target's logical QPU count.

    The current runtime has no multi-QPU scheduler, so registered targets
    report exactly one logical QPU.
    """

    return get_target().num_qpus()


@contextmanager
def target(target_or_name: Union[str, Target]) -> Iterator[Target]:
    """Temporarily select a target and restore the prior target on exit."""

    selected = _resolve_target(target_or_name)
    token = _ACTIVE_TARGET.set(selected)
    try:
        yield selected
    finally:
        _ACTIVE_TARGET.reset(token)


def resolve_backend_name(backend: Optional[str]) -> str:
    """Resolve an explicit backend or the current target's backend name."""

    if backend is None:
        return get_target().backend
    return backend


__all__ = [
    "Target",
    "get_targets",
    "has_target",
    "get_target",
    "set_target",
    "reset_target",
    "num_qpus",
    "target",
]
