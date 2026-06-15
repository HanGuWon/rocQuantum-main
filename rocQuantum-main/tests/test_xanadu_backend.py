import pytest

from rocquantum.backends.base import UnsupportedBackend
from rocquantum.backends.xanadu import XanaduBackend


def test_xanadu_backend_is_explicit_unsupported_stub():
    backend = XanaduBackend()

    assert isinstance(backend, UnsupportedBackend)
    with pytest.raises(NotImplementedError):
        backend.authenticate()
