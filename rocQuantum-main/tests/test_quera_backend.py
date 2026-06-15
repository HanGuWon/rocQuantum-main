import pytest

from rocquantum.backends.base import UnsupportedBackend
from rocquantum.backends.quera import QuEraBackend


def test_quera_backend_is_explicit_unsupported_stub():
    backend = QuEraBackend()

    assert isinstance(backend, UnsupportedBackend)
    with pytest.raises(NotImplementedError):
        backend.authenticate()
