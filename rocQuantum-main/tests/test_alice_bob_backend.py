import pytest

from rocquantum.backends.alice_bob import AliceBobBackend
from rocquantum.backends.base import UnsupportedBackend


def test_alice_bob_backend_is_explicit_unsupported_stub():
    backend = AliceBobBackend()

    assert isinstance(backend, UnsupportedBackend)
    with pytest.raises(NotImplementedError):
        backend.authenticate()
