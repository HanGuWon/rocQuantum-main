import pytest

from rocquantum.backends.base import UnsupportedBackend
from rocquantum.backends.seeqc import SeeqcBackend


def test_seeqc_backend_is_explicit_unsupported_stub():
    backend = SeeqcBackend()

    assert isinstance(backend, UnsupportedBackend)
    with pytest.raises(NotImplementedError):
        backend.authenticate()
