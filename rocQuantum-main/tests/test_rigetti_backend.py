# tests/test_rigetti_backend.py

import os
import pytest
import rocquantum.backends.rigetti as rigetti_module
from rocquantum.backends.rigetti import RigettiBackend
from rocquantum.backends.base import JobSubmissionError, BackendAuthenticationError

# Define a simple QASM circuit for testing
TEST_QASM_CIRCUIT = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

@pytest.fixture
def rigetti_backend():
    """Pytest fixture to initialize the RigettiBackend."""
    return RigettiBackend()

@pytest.mark.skipif(
    "AWS_ACCESS_KEY_ID" not in os.environ,
    reason="AWS credentials (AWS_ACCESS_KEY_ID) not found in environment variables."
)
def test_rigetti_job_submission(rigetti_backend):
    """
    Tests the full authentication and job submission flow for the Rigetti backend.
    
    This test requires AWS credentials to be configured in the environment.
    It submits a simple QASM circuit and asserts that a valid quantum task ARN
    (job ID) is returned.
    """
    pytest.importorskip("boto3", reason="Rigetti backend requires optional boto3 dependency.")

    try:
        # Authenticate with AWS Braket
        rigetti_backend.authenticate()
        assert rigetti_backend.braket_client is not None, "Braket client should be initialized after authentication."

        # Submit the job
        shots = 10
        job_id = rigetti_backend.submit_job(TEST_QASM_CIRCUIT, shots)

        # Assert that a valid job ID (quantumTaskArn) is returned
        assert isinstance(job_id, str)
        assert job_id.startswith("arn:aws:braket:")
        assert "quantum-task" in job_id

    except (BackendAuthenticationError, JobSubmissionError) as e:
        pytest.fail(f"Rigetti backend test failed during authentication or submission: {e}")

def test_rigetti_authentication_fails_without_credentials(monkeypatch):
    """
    Tests that BackendAuthenticationError is raised if AWS credentials are not set.
    """
    if rigetti_module.boto3 is not None:
        class _NoCredentialsSession:
            def __init__(self, *args, **kwargs):
                pass

            def get_credentials(self):
                return None

        monkeypatch.setattr(rigetti_module.boto3, "Session", _NoCredentialsSession)

    backend = RigettiBackend()
    with pytest.raises(BackendAuthenticationError):
        backend.authenticate()
