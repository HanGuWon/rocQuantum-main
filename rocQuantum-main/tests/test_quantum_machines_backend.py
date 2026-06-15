import pytest

from rocquantum.backends.base import UnsupportedBackend
from rocquantum.backends.quantum_machines import QuantumMachinesBackend


def test_quantum_machines_backend_is_explicit_unsupported_stub():
    backend = QuantumMachinesBackend()

    assert isinstance(backend, UnsupportedBackend)
    with pytest.raises(NotImplementedError):
        backend.authenticate()
