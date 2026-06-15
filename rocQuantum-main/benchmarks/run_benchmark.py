"""Framework QFT benchmarks for rocQuantum's PennyLane and Qiskit adapters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUBITS = tuple(range(10, 22, 2))
DEFAULT_TRIALS = 5
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks"
DEFAULT_PENNYLANE_DEVICE = "lightning.rocq"


def setup_paths() -> list[str]:
    """Make the in-tree integration packages importable before installation."""

    paths = [
        PROJECT_ROOT,
        PROJECT_ROOT / "integrations" / "pennylane-rocq",
        PROJECT_ROOT / "integrations" / "qiskit-rocquantum-provider",
        PROJECT_ROOT / "build",
    ]
    inserted: list[str] = []
    for path in reversed(paths):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
            inserted.append(resolved)
    return list(reversed(inserted))


setup_paths()


def _import_pennylane():
    try:
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("PennyLane is required for the PennyLane benchmark.") from exc
    return qml


def _import_qiskit():
    try:
        from qiskit import QuantumCircuit, transpile
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("Qiskit is required for the Qiskit benchmark.") from exc
    return QuantumCircuit, transpile


def make_pennylane_device(device_name: str, num_qubits: int):
    qml = _import_pennylane()
    try:
        return qml.device(device_name, wires=num_qubits)
    except Exception as exc:
        if device_name not in {
            "rocquantum.qpu",
            "rocq.pennylane",
            "lightning.rocq",
            "lightning.rocm",
        }:
            raise
        try:
            from pennylane_rocq import (
                LightningRocmDevice,
                LightningRocqDevice,
                RocQDevice,
                RocqDevice,
            )
        except ImportError:
            raise exc
        classes = {
            "rocquantum.qpu": RocQDevice,
            "rocq.pennylane": RocqDevice,
            "lightning.rocq": LightningRocqDevice,
            "lightning.rocm": LightningRocmDevice,
        }
        return classes[device_name](wires=num_qubits)


def make_qiskit_cpu_backend():
    try:
        from qiskit_aer import AerSimulator

        return AerSimulator(), "AerSimulator"
    except ImportError:
        from qiskit.providers.basic_provider import BasicSimulator

        return BasicSimulator(), "BasicSimulator"


def parse_qubits(raw: str) -> tuple[int, ...]:
    value = raw.strip()
    if not value:
        raise argparse.ArgumentTypeError("qubit list cannot be empty")
    if ":" in value:
        parts = [int(part) for part in value.split(":")]
        if len(parts) not in {2, 3}:
            raise argparse.ArgumentTypeError("range syntax must be start:stop[:step]")
        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) == 3 else 1
        qubits = tuple(range(start, stop, step))
    else:
        qubits = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not qubits or any(qubit <= 0 for qubit in qubits):
        raise argparse.ArgumentTypeError("qubits must be positive integers")
    return qubits


def generate_pennylane_qft(num_qubits: int, *, device: Any | None = None, device_name: str = DEFAULT_PENNYLANE_DEVICE):
    """Create a PennyLane QNode for QFT on the requested device."""

    qml = _import_pennylane()
    qnode_device = device if device is not None else make_pennylane_device(device_name, num_qubits)

    def qft_rotations(wires):
        for i in range(len(wires)):
            for j in range(i):
                qml.CRZ(np.pi / 2 ** (i - j), wires=[wires[j], wires[i]])

    def swap_qubits(wires):
        for i in range(len(wires) // 2):
            qml.SWAP(wires=[wires[i], wires[len(wires) - 1 - i]])

    @qml.qnode(qnode_device)
    def circuit():
        wires = range(num_qubits)
        for i in wires:
            qml.Hadamard(wires=i)
        qft_rotations(wires=range(num_qubits))
        swap_qubits(wires=range(num_qubits))
        return qml.state()

    return circuit


def generate_qiskit_qft(num_qubits: int):
    """Create a Qiskit QuantumCircuit for QFT."""

    QuantumCircuit, _ = _import_qiskit()
    circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circuit.h(i)
        for j in range(i + 1, num_qubits):
            circuit.cp(np.pi / 2 ** (j - i), j, i)
    for i in range(num_qubits // 2):
        circuit.swap(i, num_qubits - 1 - i)
    return circuit


def _average_runtime(callback, trials: int) -> float:
    if trials <= 0:
        raise ValueError("trials must be a positive integer")
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        callback()
        times.append(time.perf_counter() - start_time)
    return float(np.mean(times))


def _speedup(cpu_time: float, rocq_time: float) -> float | None:
    if rocq_time <= 0.0:
        return None
    return float(cpu_time / rocq_time)


def run_pennylane_benchmark(
    *,
    qubits: Iterable[int] = DEFAULT_QUBITS,
    trials: int = DEFAULT_TRIALS,
    rocq_device_name: str = DEFAULT_PENNYLANE_DEVICE,
) -> dict[str, Any]:
    print("\n" + "=" * 40)
    print(" PennyLane Performance Benchmark: QFT ")
    print("=" * 40)

    results: dict[str, Any] = {
        "qubits": [],
        "rocq_time": [],
        "cpu_time": [],
        "speedup": [],
        "rocq_device": rocq_device_name,
        "cpu_device": "default.qubit",
    }

    for num_qubits in qubits:
        print(f"\nRunning benchmark for {num_qubits} qubits...")

        rocq_circuit = generate_pennylane_qft(num_qubits, device_name=rocq_device_name)
        rocq_avg_time = _average_runtime(rocq_circuit, trials)

        cpu_circuit = generate_pennylane_qft(num_qubits, device_name="default.qubit")
        cpu_avg_time = _average_runtime(cpu_circuit, trials)
        speedup = _speedup(cpu_avg_time, rocq_avg_time)

        speedup_label = f"{speedup:7.3f}x" if speedup is not None else "n/a"
        print(
            f"  Qubits: {num_qubits:2} | rocQuantum: {rocq_avg_time:7.3f}s | "
            f"CPU (default.qubit): {cpu_avg_time:7.3f}s | Speedup: {speedup_label}"
        )
        results["qubits"].append(int(num_qubits))
        results["rocq_time"].append(rocq_avg_time)
        results["cpu_time"].append(cpu_avg_time)
        results["speedup"].append(speedup)

    return results


def run_qiskit_benchmark(*, qubits: Iterable[int] = DEFAULT_QUBITS, trials: int = DEFAULT_TRIALS) -> dict[str, Any]:
    print("\n" + "=" * 40)
    print(" Qiskit Performance Benchmark: QFT ")
    print("=" * 40)

    _, transpile = _import_qiskit()
    from qiskit_rocquantum_provider import RocQuantumProvider

    results: dict[str, Any] = {
        "qubits": [],
        "rocq_time": [],
        "cpu_time": [],
        "speedup": [],
        "rocq_backend": "rocq_simulator",
    }

    rocq_backend = RocQuantumProvider().get_backend("rocq_simulator")
    cpu_backend, cpu_backend_name = make_qiskit_cpu_backend()
    results["cpu_backend"] = cpu_backend_name

    for num_qubits in qubits:
        print(f"\nRunning benchmark for {num_qubits} qubits...")
        circuit = generate_qiskit_qft(num_qubits)

        transpiled_rocq = transpile(circuit, rocq_backend)
        rocq_avg_time = _average_runtime(lambda: rocq_backend.run(transpiled_rocq, shots=1).result(), trials)

        transpiled_cpu = transpile(circuit, cpu_backend)
        cpu_avg_time = _average_runtime(lambda: cpu_backend.run(transpiled_cpu, shots=1).result(), trials)
        speedup = _speedup(cpu_avg_time, rocq_avg_time)

        speedup_label = f"{speedup:7.3f}x" if speedup is not None else "n/a"
        print(
            f"  Qubits: {num_qubits:2} | rocQuantum: {rocq_avg_time:7.3f}s | "
            f"CPU ({cpu_backend_name}): {cpu_avg_time:7.3f}s | Speedup: {speedup_label}"
        )
        results["qubits"].append(int(num_qubits))
        results["rocq_time"].append(rocq_avg_time)
        results["cpu_time"].append(cpu_avg_time)
        results["speedup"].append(speedup)

    return results


def write_results_json(results: dict[str, Any], output_dir: Path, *, trials: int, qubits: Iterable[int]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "benchmark": "framework_qft",
        "trials": int(trials),
        "qubits": [int(qubit) for qubit in qubits],
        "frameworks": results,
    }
    output_path = output_dir / "framework-benchmark-results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nBenchmark JSON saved to '{output_path}'")
    return output_path


def run_or_skip_native_binding(framework_name: str, callback) -> dict[str, Any]:
    try:
        return callback()
    except ImportError as exc:
        reason = str(exc)
        if "rocquantum_bind" not in reason:
            raise
        print(f"Skipping {framework_name} benchmark: {reason}")
        return {
            "status": "skipped",
            "reason": reason,
        }


def plot_results(results: dict[str, Any], framework_name: str, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(results["qubits"], results["rocq_time"], "o-", label="rocQuantum Simulator")
    plt.plot(results["qubits"], results["cpu_time"], "s-", label=f"Default {framework_name} CPU Simulator")

    plt.xlabel("Number of Qubits")
    plt.ylabel("Execution Time (seconds)")
    plt.yscale("log")
    plt.title(f"{framework_name} QFT Benchmark: rocQuantum vs. CPU")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    filename = output_dir / f"benchmark_results_{framework_name.lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Benchmark plot saved to '{filename}'")
    return filename


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework",
        choices=("all", "pennylane", "qiskit"),
        default="all",
        help="Framework benchmark to run.",
    )
    parser.add_argument(
        "--qubits",
        type=parse_qubits,
        default=DEFAULT_QUBITS,
        help="Comma list or start:stop[:step] range of qubit counts.",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Trials per qubit count.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and optional PNG outputs.",
    )
    parser.add_argument(
        "--pennylane-device",
        default=DEFAULT_PENNYLANE_DEVICE,
        help="rocQuantum PennyLane device entry point to benchmark.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib PNG generation.")
    parser.add_argument(
        "--require-plots",
        action="store_true",
        help="Fail instead of continuing when matplotlib plot generation is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trials <= 0:
        raise ValueError("trials must be a positive integer")

    qubits = tuple(args.qubits)
    results: dict[str, Any] = {}
    if args.framework in {"all", "pennylane"}:
        results["PennyLane"] = run_or_skip_native_binding(
            "PennyLane",
            lambda: run_pennylane_benchmark(
                qubits=qubits,
                trials=args.trials,
                rocq_device_name=args.pennylane_device,
            ),
        )
    if args.framework in {"all", "qiskit"}:
        results["Qiskit"] = run_or_skip_native_binding(
            "Qiskit",
            lambda: run_qiskit_benchmark(qubits=qubits, trials=args.trials),
        )

    write_results_json(results, args.output_dir, trials=args.trials, qubits=qubits)

    if not args.no_plots:
        for framework_name, framework_results in results.items():
            if framework_results.get("status") == "skipped":
                print(f"Skipping {framework_name} plot generation: benchmark was skipped.")
                continue
            try:
                plot_results(framework_results, framework_name, args.output_dir)
            except (ImportError, OSError) as exc:
                if args.require_plots:
                    raise
                print(f"Skipping {framework_name} plot generation: {exc}")

    print("\nBenchmark run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
