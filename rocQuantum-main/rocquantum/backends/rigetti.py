# rocquantum/backends/rigetti.py

"""
This module provides a concrete implementation of the RocqBackend for the 
Rigetti quantum computing platform, accessed via Amazon Braket.

It uses the boto3 library to interact with the AWS Braket service, 
encapsulating the logic for authentication, job submission, and result
retrieval for Rigetti QPUs.
"""

import os

from typing import Dict, Any

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - exercised in environments without boto3
    boto3 = None

    class ClientError(Exception):
        """Fallback type used when botocore is not installed."""

from .base import (
    RocqBackend,
    BackendAuthenticationError,
    JobSubmissionError,
)

# Default AWS region for Braket if not specified
DEFAULT_AWS_REGION = "us-west-1"

# ARN for the Rigetti Aspen-M-3 QPU on AWS Braket
RIGETTI_ASPEN_M_3_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"


class RigettiBackend(RocqBackend):
    """
    A client for interacting with Rigetti quantum hardware via AWS Braket.

    This class implements the RocqBackend interface but overrides the default
    API-based methods to use the boto3 SDK for AWS communication.
    """

    def __init__(self, backend_name: str = "rigetti", aws_region: str = DEFAULT_AWS_REGION,
                 s3_output: str = None):
        """
        Initializes the Rigetti backend client.

        Args:
            backend_name (str): The name of the backend.
            aws_region (str): The AWS region where the Braket service is hosted.
            s3_output (str): S3 URI for Braket task output. Falls back to
                ``ROCQ_RIGETTI_S3_OUTPUT`` env var if not provided.
        """
        super().__init__(backend_name=backend_name, api_endpoint="")  # API endpoint not used
        self.aws_region = aws_region
        self.braket_client = None
        self.s3_output = s3_output or os.getenv("ROCQ_RIGETTI_S3_OUTPUT")

    def authenticate(self) -> None:
        """
        Authenticates with AWS Braket using credentials configured for boto3.

        boto3 automatically searches for credentials in environment variables,
        shared credential files (~/.aws/credentials), or IAM roles.

        Raises:
            BackendAuthenticationError: If the AWS session cannot be created
                                        due to missing credentials or other
                                        configuration issues.
        """
        if boto3 is None:
            raise BackendAuthenticationError(
                "AWS Braket support requires the optional 'boto3' and "
                "'botocore' dependencies. Install rocQuantum with the "
                "Rigetti backend extras or install boto3 directly."
            )

        try:
            session = boto3.Session(region_name=self.aws_region)
            if session.get_credentials() is None:
                raise BackendAuthenticationError(
                    "AWS Braket authentication failed: no AWS credentials were "
                    "found. Configure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
                    "a shared credentials file, or an IAM role."
                )

            self.braket_client = session.client("braket")
            # A simple check to confirm client was created
            if self.braket_client is None:
                 raise ClientError({}, "Boto3 client creation failed")
            print("AWS Braket client created successfully.")
        except BackendAuthenticationError:
            raise
        except ClientError as e:
            raise BackendAuthenticationError(
                f"AWS Braket authentication failed: {e}. Ensure your AWS "
                "credentials (e.g., AWS_ACCESS_KEY_ID) are configured correctly."
            )

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Returns an empty dictionary as authentication is handled by the SDK.
        """
        return {}

    def submit_job(self, circuit: str, shots: int) -> str:
        """
        Submits a quantum circuit (as a QASM string) to a Rigetti QPU on AWS Braket.

        This method overrides the base class's REST API implementation to use
        the boto3 SDK.

        Args:
            circuit (str): The OpenQASM 3.0 representation of the circuit.
            shots (int): The number of times to execute the circuit.

        Returns:
            str: The `quantumTaskArn` which serves as the unique job ID.

        Raises:
            JobSubmissionError: If the job submission via boto3 fails.
        """
        if self.braket_client is None:
            raise JobSubmissionError("Braket client not initialized. Please call authenticate() first.")

        if not self.s3_output:
            raise JobSubmissionError(
                "Rigetti S3 output location not configured. Set the "
                "'ROCQ_RIGETTI_S3_OUTPUT' environment variable or pass "
                "s3_output= to the RigettiBackend constructor."
            )
        s3_output_location = self.s3_output

        try:
            response = self.braket_client.create_quantum_task(
                action={
                    "type": "OPENQASM",
                    "source": circuit,
                },
                shots=shots,
                deviceArn=RIGETTI_ASPEN_M_3_ARN,
                s3OutputLocation=s3_output_location,
            )
        except ClientError as e:
            raise JobSubmissionError(f"Failed to submit job to AWS Braket: {e}")

        quantum_task_arn = response.get("quantumTaskArn")
        if not quantum_task_arn:
            raise JobSubmissionError("AWS Braket response did not contain a quantumTaskArn.")
            
        return quantum_task_arn
