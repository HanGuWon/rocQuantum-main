import pytest

from rocquantum.backends.base import BackendAuthenticationError
from rocquantum.backends.infleqtion import InfleqtionBackend


def test_infleqtion_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("SUPERSTAQ_API_KEY", raising=False)
    backend = InfleqtionBackend()

    with pytest.raises(BackendAuthenticationError, match="SUPERSTAQ_API_KEY"):
        backend.authenticate()


def test_infleqtion_payload_shape_is_stable():
    backend = InfleqtionBackend(backend_name="cq_hilbert_qpu")

    payload = backend._build_payload("OPENQASM 3.0;", shots=7)

    assert payload == {
        "target": "cq_hilbert_qpu",
        "shots": 7,
        "body": {
            "language": "OPENQASM",
            "program": "OPENQASM 3.0;",
        },
    }
