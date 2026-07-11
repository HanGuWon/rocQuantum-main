import pytest

from rocquantum.backends.pasqal import PASQAL_API_ENDPOINT, PasqalBackend


def test_pasqal_defaults_are_stable():
    backend = PasqalBackend()

    assert backend.backend_name == "pasqal"
    assert backend.api_endpoint == PASQAL_API_ENDPOINT


@pytest.mark.parametrize("api_key", [None, ""])
def test_pasqal_missing_api_key_fails_fast(monkeypatch, api_key):
    from rocquantum.backends.base import BackendAuthenticationError

    if api_key is None:
        monkeypatch.delenv("PASQAL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("PASQAL_API_KEY", api_key)
    backend = PasqalBackend()

    with pytest.raises(BackendAuthenticationError) as error:
        backend.authenticate()

    assert str(error.value) == (
        "Authentication failed: The 'PASQAL_API_KEY' environment variable is not set. "
        "Please set it to your Pasqal API key."
    )


def test_pasqal_authentication_builds_api_key_header(monkeypatch, capsys):
    monkeypatch.setenv("PASQAL_API_KEY", "pasqal-secret")
    backend = PasqalBackend()

    backend.authenticate()

    assert backend._get_auth_headers() == {"Authorization": "ApiKey pasqal-secret"}
    assert capsys.readouterr().out == "Authentication successful.\n"


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
