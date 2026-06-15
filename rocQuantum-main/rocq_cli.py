# rocq_cli.py

"""
rocQuantum Command-Line Interface (CLI)

A simple tool for demonstrating the rocQuantum framework's capabilities
by running a standard Bell State circuit on a specified backend.
"""

import os
import argparse
import sys
import time

from rocquantum.circuit import QuantumCircuit
from rocquantum.core import get_active_backend, list_backends, set_target
from rocquantum.backends.base import (
    BackendAuthenticationError,
    JobSubmissionError,
    ResultRetrievalError,
)

def create_bell_circuit() -> QuantumCircuit:
    """Creates a standard 2-qubit Bell State circuit."""
    qc = QuantumCircuit(num_qubits=2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def check_environment_vars(backend_name: str):
    """Checks for required environment variables for a given backend."""
    if backend_name == 'ionq' and not os.getenv('IONQ_API_KEY'):
        raise EnvironmentError("Error: IONQ_API_KEY environment variable not set.")
    if backend_name == 'quantinuum' and not os.getenv('CUDAQ_QUANTINUUM_CREDENTIALS'):
        raise EnvironmentError("Error: CUDAQ_QUANTINUUM_CREDENTIALS environment variable not set.")
    if backend_name == 'rigetti' and not os.getenv('AWS_ACCESS_KEY_ID'):
        raise EnvironmentError("Error: AWS credentials (e.g., AWS_ACCESS_KEY_ID) not set.")
    if backend_name == 'rigetti' and not os.getenv('ROCQ_RIGETTI_S3_OUTPUT'):
        raise EnvironmentError(
            "Error: ROCQ_RIGETTI_S3_OUTPUT environment variable not set. "
            "Set it to an S3 URI for Braket task output, e.g. "
            "'s3://your-bucket/braket-results'."
        )
    # Add other backend checks here as needed


def print_backend_capabilities(include_experimental: bool = False) -> None:
    """Print backend status metadata before target selection."""
    backends = list_backends(include_experimental=include_experimental)
    print("Backend\tStatus\tNotes")
    for name, info in sorted(backends.items()):
        notes = []
        if info.get("requires_local_runtime"):
            notes.append(f"requires {info.get('runtime', 'local runtime')}")
        if info.get("requires_experimental_opt_in"):
            notes.append("requires experimental opt-in")
        if info.get("safe_to_submit_jobs") is False:
            notes.append("job submission disabled")
        if info.get("unsupported_reason"):
            notes.append(str(info["unsupported_reason"]))
        print(f"{name}\t{info['status']}\t{', '.join(notes) if notes else '-'}")


def main():
    """Parses arguments and runs the quantum circuit."""
    parser = argparse.ArgumentParser(
        description="Run a Bell State circuit on a rocQuantum backend."
    )
    parser.add_argument(
        "command",
        choices=["run", "list-backends"],
        help="The command to execute.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=False,
        help="The name of the backend to run on (e.g., 'ionq', 'qristal', 'rigetti').",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Include unsupported skeleton providers in backend capability output.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100,
        help="The number of shots to execute.",
    )
    args = parser.parse_args()

    if args.command == "list-backends":
        print_backend_capabilities(include_experimental=args.include_experimental)
        return

    if args.command == "run":
        if not args.backend:
            parser.error("--backend is required for the 'run' command.")
        try:
            print(f"--> Preparing to run on backend: {args.backend}")
            
            # 1. Check for credentials
            check_environment_vars(args.backend)
            
            # 2. Create circuit
            bell_circuit = create_bell_circuit()
            print("--> Bell State circuit created.")

            # 3. Set target and get backend
            if args.backend == 'ionq':
                set_target(args.backend, backend_name='simulator')
            else:
                set_target(args.backend)
            backend = get_active_backend()
            print(f"--> Target set to '{backend.backend_name}'.")

            # 4. Submit job
            print(f"--> Submitting job with {args.shots} shots...")
            
            # Handle different submission requirements
            if args.backend in ['ionq', 'rigetti', 'pasqal', 'infleqtion']: # Type A/C
                job_id = backend.submit_job(bell_circuit.to_qasm(), shots=args.shots)
            else: # Type B (local SDK)
                job_id = backend.submit_job(bell_circuit, shots=args.shots)
            
            print(f"--> Job submitted successfully. Job ID: {job_id}")

            # 5. Poll for and display results
            print("--> Waiting for results...")
            while True:
                status = backend.get_job_status(job_id)
                print(f"    Current job status: {status}")
                if status == 'completed':
                    results = backend.get_job_result(job_id)
                    print("\n--> Execution complete!")
                    print(f"    Results: {results}")
                    break
                elif status in ['failed', 'cancelled']:
                    print(f"\n--> Job {status}. Cannot retrieve results.")
                    break
                # For synchronous backends like Qristal, this loop runs once.
                if status == 'completed':
                    break
                time.sleep(2)

        except (EnvironmentError, BackendAuthenticationError, JobSubmissionError, ResultRetrievalError, ImportError) as e:
            print(f"\n[ERROR] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
