# rocquantum/backends/ionq.py

"""
This module provides a concrete implementation of the RocqBackend for the 
IonQ quantum computing platform.

It enables communication with the IonQ REST API (Version 0.3) to manage
the lifecycle of quantum jobs, including authentication, submission, 
status monitoring, and result retrieval.
"""

from .base import BackendAuthenticationError, RocqBackend, _ApiKeyOpenQasmBackend

# The base URL for the IonQ API, version 0.3
IONQ_API_V0_3_ENDPOINT = "https://api.ionq.co/v0.3"


class IonQBackend(_ApiKeyOpenQasmBackend):
    """
    A client for interacting with the IonQ quantum computing hardware.

    This class implements the RocqBackend interface and provides a concrete
    method for executing quantum circuits on IonQ's QPUs through their
    public REST API.
    """

    api_key_environment_variable = "IONQ_API_KEY"
    api_key_provider_name = "IonQ"

    def __init__(self, backend_name: str = "qpu", api_endpoint: str = IONQ_API_V0_3_ENDPOINT):
        """
        Initializes the IonQ backend client.

        Args:
            backend_name (str): The specific name of the IonQ backend to target.
                                Defaults to 'qpu'. Other examples include 
                                'qpu.aria-1' or 'simulator'.
            api_endpoint (str): The base URL for the IonQ API. Defaults to the
                                standard v0.3 endpoint.
        """
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
