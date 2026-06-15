import pytest

from rocquantum.backends.base import UnsupportedBackend
from rocquantum.backends.iqm import IQMBackend


def test_iqm_backend_is_explicit_unsupported_stub():
    backend = IQMBackend()

    assert isinstance(backend, UnsupportedBackend)
    with pytest.raises(NotImplementedError):
        backend.authenticate()
