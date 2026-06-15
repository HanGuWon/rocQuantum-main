# rocquantum/backends/qristal.py

"""
This module provides a Type B (Direct, Provider-Managed Integration) backend
for the Quantum Brilliance Qristal SDK.

This backend executes jobs locally by invoking the Qristal SDK's command-line
tool. It demonstrates a synchronous, file-based execution model, contrasting
with the API-based Type A backends.
"""

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import suppress
from typing import Any, Dict, Optional

from .base import BackendAuthenticationError, RocqBackend, JobSubmissionError, ResultRetrievalError
from rocquantum.circuit import QuantumCircuit

_QRISTAL_CLI_ENV_VAR = "ROCQ_QRISTAL_CLI"
_DEFAULT_QRISTAL_CLI = "qristal"


class QuantumBrillianceBackend(RocqBackend):
    """
    A backend for executing circuits locally via the Qristal SDK.

    This class overrides the default job submission logic to run a local
    subprocess, making it a Type B backend.
    """

    def __init__(
        self,
        backend_name: str = "qristal",
        api_endpoint: str = "local",
        cli_executable: Optional[str] = None,
    ):
        """Initializes the Qristal local backend."""
        super().__init__(backend_name=backend_name, api_endpoint=api_endpoint)
        self.cli_executable = cli_executable or os.getenv(_QRISTAL_CLI_ENV_VAR, _DEFAULT_QRISTAL_CLI)
        self._cli_path = None
        self._local_results: Dict[str, Dict[str, str]] = {}

    def _resolve_cli(self) -> str:
        resolved = shutil.which(self.cli_executable)
        if resolved is None:
            raise BackendAuthenticationError(
                f"Qristal SDK CLI '{self.cli_executable}' was not found. "
                f"Install the Qristal SDK or set {_QRISTAL_CLI_ENV_VAR} to the CLI path before "
                "selecting the qristal backend."
            )
        self._cli_path = resolved
        return resolved

    def authenticate(self) -> None:
        """Validate that the local Qristal CLI is available."""
        self._resolve_cli()

    def _get_auth_headers(self) -> Dict[str, str]:
        """Not applicable for a local backend."""
        return {}

    def _build_payload(self, circuit_representation: str, shots: int) -> Dict[str, Any]:
        """Not applicable for the direct execution model."""
        raise NotImplementedError("Payload building is not used for Type B backends.")

    def submit_job(self, circuit: QuantumCircuit, shots: int) -> str:
        """
        Executes a quantum circuit synchronously using the Qristal CLI.

        This method writes the circuit to a temporary QASM file and runs it
        using a subprocess.

        Args:
            circuit (QuantumCircuit): The QuantumCircuit object to execute.
            shots (int): The number of shots to run.

        Returns:
            str: A unique local job identifier.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise JobSubmissionError("Qristal backend requires a QuantumCircuit object, not a QASM string.")
        if not isinstance(shots, int) or shots <= 0:
            raise JobSubmissionError("Qristal backend requires shots to be a positive integer.")

        try:
            cli_path = self._cli_path or self._resolve_cli()
        except BackendAuthenticationError as e:
            raise JobSubmissionError(str(e)) from e
        qasm_string = circuit.to_qasm()
        tmp_filepath = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False) as tmp_file:
            tmp_file.write(qasm_string)
            tmp_filepath = tmp_file.name

        command = [cli_path, "--run", tmp_filepath, "--shots", str(shots)]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            stdout = result.stdout
        except FileNotFoundError:
            raise JobSubmissionError(
                f"Job submission failed: Qristal CLI '{cli_path}' was not found. "
                f"Set {_QRISTAL_CLI_ENV_VAR} or install the Qristal SDK."
            )
        except subprocess.CalledProcessError as e:
            raise JobSubmissionError(
                f"Qristal job execution failed with exit code {e.returncode}: {e.stderr or e.stdout}"
            )
        finally:
            if tmp_filepath is not None:
                with suppress(OSError):
                    os.unlink(tmp_filepath)

        job_id = f"local-run-{uuid.uuid4()}"
        self._local_results[job_id] = {"stdout": stdout}
        return job_id

    def get_job_status(self, job_id: str) -> str:
        """For a synchronous local run, the status is always 'completed'."""
        if job_id in self._local_results:
            return 'completed'
        raise ResultRetrievalError(f"Local job ID '{job_id}' not found.")

    def get_job_result(self, job_id: str) -> Dict[str, int]:
        """
        Parses the stored stdout from the subprocess to extract the histogram.
        """
        if job_id not in self._local_results:
            raise ResultRetrievalError(f"Local job ID '{job_id}' not found.")

        stdout = self._local_results[job_id]["stdout"]
        
        try:
            return self._parse_histogram(stdout)
        except (ValueError, StopIteration, json.JSONDecodeError, IndexError) as e:
            raise ResultRetrievalError(
                f"Failed to parse histogram from Qristal output. Error: {e}. "
                f"Full output:\n{stdout}"
            )

    def _parse_histogram(self, stdout: str) -> Dict[str, int]:
        histogram_line = next(line for line in stdout.splitlines() if "Histogram:" in line)
        json_str = histogram_line.split("Histogram:", 1)[1].strip()
        histogram = json.loads(json_str)
        if not isinstance(histogram, dict):
            raise ValueError("Qristal histogram payload is not a JSON object.")
        return {str(bitstring): int(count) for bitstring, count in histogram.items()}
