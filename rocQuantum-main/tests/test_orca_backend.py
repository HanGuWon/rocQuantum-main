import pytest

from rocquantum.backends.base import UnsupportedBackend
from rocquantum.backends.orca import OrcaBackend


def test_orca_backend_is_explicit_unsupported_stub():
    backend = OrcaBackend()

    assert isinstance(backend, UnsupportedBackend)
    with pytest.raises(NotImplementedError):
        backend.authenticate()
