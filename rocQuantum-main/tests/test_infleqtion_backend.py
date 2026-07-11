import pytest

from rocquantum.backends.infleqtion import INFLEQTION_API_ENDPOINT, InfleqtionBackend


def test_infleqtion_defaults_are_stable():
    backend = InfleqtionBackend()

    assert backend.backend_name == "infleqtion"
    assert backend.api_endpoint == INFLEQTION_API_ENDPOINT


@pytest.mark.parametrize("api_key", [None, ""])
def test_infleqtion_missing_api_key_fails_fast(monkeypatch, api_key):
    from rocquantum.backends.base import BackendAuthenticationError

    if api_key is None:
        monkeypatch.delenv("SUPERSTAQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("SUPERSTAQ_API_KEY", api_key)
    backend = InfleqtionBackend()

    with pytest.raises(BackendAuthenticationError) as error:
        backend.authenticate()

    assert str(error.value) == (
        "Authentication failed: The 'SUPERSTAQ_API_KEY' environment variable is not set. "
        "Please set it to your Superstaq API key."
    )


def test_infleqtion_authentication_builds_api_key_header(monkeypatch, capsys):
    monkeypatch.setenv("SUPERSTAQ_API_KEY", "superstaq-secret")
    backend = InfleqtionBackend()

    backend.authenticate()

    assert backend._get_auth_headers() == {"Authorization": "ApiKey superstaq-secret"}
    assert capsys.readouterr().out == "Authentication successful.\n"


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
