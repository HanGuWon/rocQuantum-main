from .backend import RocQuantumBackend

try:
    from qiskit.providers import ProviderV1 as _ProviderBase
except ImportError:  # Qiskit 1.x/2.x no longer requires ProviderV1 for local providers.
    _ProviderBase = object


class RocQuantumProvider(_ProviderBase):
    """
    Provider for the rocQuantum simulator backend.
    """
    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.name = "rocquantum_provider"
        # Instantiate and store the backend instance
        self._backends = {"rocq_simulator": RocQuantumBackend(provider=self)}

    def backends(self, name=None, **kwargs):
        """Return a list of backends."""
        if name is not None:
            backend = self._backends.get(name)
            return [backend] if backend is not None else []
        return list(self._backends.values())

    def get_backend(self, name=None, **kwargs):
        """Return a single backend by name, matching Qiskit provider ergonomics."""
        matches = self.backends(name=name, **kwargs)
        if not matches:
            raise KeyError(f"No rocQuantum backend named {name!r}.")
        if len(matches) > 1:
            raise ValueError(f"More than one backend matches {name!r}.")
        return matches[0]
