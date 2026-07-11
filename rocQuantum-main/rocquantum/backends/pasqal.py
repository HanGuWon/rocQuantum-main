# rocquantum/backends/pasqal.py

"""
This module provides a concrete implementation of the RocqBackend for the 
Pasqal quantum computing platform.

It enables communication with the Pasqal REST API to manage
the lifecycle of quantum jobs, including authentication, submission, 
status monitoring, and result retrieval.
"""

from .base import BackendAuthenticationError, RocqBackend, _ApiKeyOpenQasmBackend

# The base URL for the Pasqal API
PASQAL_API_ENDPOINT = "https://api.pasqal.cloud"


class PasqalBackend(_ApiKeyOpenQasmBackend):
    """
    A client for interacting with the Pasqal quantum computing hardware.

    This class implements the RocqBackend interface and provides a concrete
    method for executing quantum circuits on Pasqal's QPUs.
    """

    api_key_environment_variable = "PASQAL_API_KEY"
    api_key_provider_name = "Pasqal"

    def __init__(self, backend_name: str = "pasqal", api_endpoint: str = PASQAL_API_ENDPOINT):
        """
        Initializes the Pasqal backend client.

        Args:
            backend_name (str): The specific name of the Pasqal backend to target.
            api_endpoint (str): The base URL for the Pasqal API.
        """
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
