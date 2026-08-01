# rocquantum/backends/infleqtion.py

"""
This module provides a concrete implementation of the RocqBackend for the 
Infleqtion quantum computing platform using the Superstaq API.

It enables communication with the Infleqtion REST API to manage
the lifecycle of quantum jobs, including authentication, submission, 
status monitoring, and result retrieval.
"""

from .base import BackendAuthenticationError, RocqBackend, _ApiKeyOpenQasmBackend

# The base URL for the Infleqtion Superstaq API
INFLEQTION_API_ENDPOINT = "https://api.superstaq.infleqtion.com"


class InfleqtionBackend(_ApiKeyOpenQasmBackend):
    """
    A client for interacting with the Infleqtion quantum computing hardware
    via the Superstaq API.

    This class implements the RocqBackend interface and provides a concrete
    method for executing quantum circuits on Infleqtion's QPUs.
    """

    api_key_environment_variable = "SUPERSTAQ_API_KEY"
    api_key_provider_name = "Superstaq"

    def __init__(self, backend_name: str = "infleqtion", api_endpoint: str = INFLEQTION_API_ENDPOINT):
        """
        Initializes the Infleqtion backend client.

        Args:
            backend_name (str): The specific name of the Infleqtion backend to target.
            api_endpoint (str): The base URL for the Superstaq API.
        """
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
