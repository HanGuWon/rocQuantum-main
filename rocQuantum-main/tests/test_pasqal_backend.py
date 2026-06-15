import pytest

from rocquantum.backends.base import BackendAuthenticationError
from rocquantum.backends.pasqal import PasqalBackend


def test_pasqal_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("PASQAL_API_KEY", raising=False)
    backend = PasqalBackend()

    with pytest.raises(BackendAuthenticationError, match="PASQAL_API_KEY"):
        backend.authenticate()


def test_pasqal_payload_shape_is_stable():
    backend = PasqalBackend(backend_name="pasqal_qpu")

    payload = backend._build_payload("OPENQASM 3.0;", shots=11)

    assert payload == {
        "target": "pasqal_qpu",
        "shots": 11,
        "body": {
            "language": "OPENQASM",
            "program": "OPENQASM 3.0;",
        },
    }
