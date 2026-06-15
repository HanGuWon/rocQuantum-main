# rocquantum/core.py

"""
This module serves as the central management hub for backend clients 
in the rocQuantum framework.
"""

import importlib
import os
from typing import Dict, Optional, Type

from .backends.base import RocqBackend

_EXPERIMENTAL_PROVIDER_ENV_VAR = "ROCQ_ENABLE_EXPERIMENTAL_PROVIDERS"

_CONCRETE_BACKENDS: Dict[str, str] = {
    "ionq": "rocquantum.backends.ionq.IonQBackend",
    "infleqtion": "rocquantum.backends.infleqtion.InfleqtionBackend",
    "pasqal": "rocquantum.backends.pasqal.PasqalBackend",
    "quantinuum": "rocquantum.backends.quantinuum.QuantinuumBackend",
    "qristal": "rocquantum.backends.qristal.QuantumBrillianceBackend",
    "rigetti": "rocquantum.backends.rigetti.RigettiBackend",
}

_SKELETON_BACKENDS: Dict[str, str] = {
    "iqm": "rocquantum.backends.iqm.IQMBackend",
    "xanadu": "rocquantum.backends.xanadu.XanaduBackend",
    "quera": "rocquantum.backends.quera.QuEraBackend",
    "orca": "rocquantum.backends.orca.OrcaBackend",
    "seeqc": "rocquantum.backends.seeqc.SeeqcBackend",
    "quantum_machines": "rocquantum.backends.quantum_machines.QuantumMachinesBackend",
    "alice_bob": "rocquantum.backends.alice_bob.AliceBobBackend",
}

_AVAILABLE_BACKENDS: Dict[str, str] = {**_CONCRETE_BACKENDS, **_SKELETON_BACKENDS}
_ACTIVE_BACKEND: Optional[RocqBackend] = None


def _experimental_providers_enabled() -> bool:
    return os.environ.get(_EXPERIMENTAL_PROVIDER_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def list_backends(include_experimental: bool = False) -> Dict[str, Dict[str, object]]:
    """Return provider backends exposed by the public target-selection API."""
    names = sorted(_AVAILABLE_BACKENDS)
    result: Dict[str, Dict[str, object]] = {}
    for name in names:
        is_skeleton = name in _SKELETON_BACKENDS
        if is_skeleton and not include_experimental:
            continue
        result[name] = {
            "import_path": _AVAILABLE_BACKENDS[name],
            "status": "unsupported_stub" if is_skeleton else "client",
            "requires_experimental_opt_in": is_skeleton,
        }
    return result


def set_target(name: str, *, allow_experimental: bool = False, **kwargs) -> None:
    """Selects, instantiates, and authenticates a quantum backend."""
    global _ACTIVE_BACKEND
    if name not in _AVAILABLE_BACKENDS:
        public_names = list(list_backends().keys())
        raise ValueError(
            f"Backend '{name}' not recognized. Available: {public_names}. "
            "Use list_backends(include_experimental=True) to inspect unsupported skeleton providers."
        )

    if name in _SKELETON_BACKENDS and not (allow_experimental or _experimental_providers_enabled()):
        raise ValueError(
            f"Backend '{name}' is an unsupported skeleton provider, not a production connector. "
            f"Pass allow_experimental=True or set {_EXPERIMENTAL_PROVIDER_ENV_VAR}=1 only for "
            "contract tests or integration development."
        )
    
    import_path = _AVAILABLE_BACKENDS[name]
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        backend_class: Type[RocqBackend] = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import backend class '{import_path}': {e}")

    instance = backend_class(**kwargs)
    instance.authenticate()
    _ACTIVE_BACKEND = instance

def get_active_backend() -> RocqBackend:
    """Retrieves the currently active backend instance."""
    if _ACTIVE_BACKEND is None:
        raise RuntimeError("No active backend. Call set_target() first.")
    return _ACTIVE_BACKEND
