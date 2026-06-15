"""Contracts for the framework-level benchmark helper."""

from __future__ import annotations

import builtins
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "benchmarks" / "run_benchmark.py"


def _load_benchmark_module(name: str = "framework_benchmark_contract"):
    spec = importlib.util.spec_from_file_location(name, BENCHMARK)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_framework_benchmark_import_is_lightweight_and_uses_source_paths():
    module = _load_benchmark_module()

    sys_path = {str(Path(path).resolve()) for path in sys.path if path}
    assert str((PROJECT_ROOT / "integrations" / "pennylane-rocq").resolve()) in sys_path
    assert str((PROJECT_ROOT / "integrations" / "qiskit-rocquantum-provider").resolve()) in sys_path
    assert module.DEFAULT_PENNYLANE_DEVICE == "lightning.rocq"

    source = BENCHMARK.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any(
        isinstance(node, ast.Import) and any(alias.name == "matplotlib.pyplot" for alias in node.names)
        for node in top_level_imports
    )
    assert ".device =" not in source


def test_framework_benchmark_parse_qubits_accepts_lists_and_ranges():
    module = _load_benchmark_module("framework_benchmark_parse")

    assert module.parse_qubits("2,4,6") == (2, 4, 6)
    assert module.parse_qubits("2:8:2") == (2, 4, 6)

    with pytest.raises(Exception):
        module.parse_qubits("0")


def test_framework_benchmark_json_writer_emits_machine_readable_results(tmp_path):
    module = _load_benchmark_module("framework_benchmark_json")

    output_path = module.write_results_json(
        {
            "PennyLane": {
                "qubits": [2],
                "rocq_time": [0.5],
                "cpu_time": [1.0],
                "speedup": [2.0],
            }
        },
        tmp_path,
        trials=1,
        qubits=(2,),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "framework_qft"
    assert payload["trials"] == 1
    assert payload["frameworks"]["PennyLane"]["speedup"] == [2.0]


def test_framework_benchmark_main_writes_skipped_json_without_native_binding(tmp_path, monkeypatch):
    module = _load_benchmark_module("framework_benchmark_native_skip")

    def missing_binding(**kwargs):
        raise ImportError("rocquantum_bind is required for framework integrations")

    monkeypatch.setattr(module, "run_qiskit_benchmark", missing_binding)

    assert (
        module.main(
            [
                "--framework",
                "qiskit",
                "--qubits",
                "1",
                "--trials",
                "1",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    payload = json.loads((tmp_path / "framework-benchmark-results.json").read_text(encoding="utf-8"))
    assert payload["frameworks"]["Qiskit"]["status"] == "skipped"
    assert "rocquantum_bind" in payload["frameworks"]["Qiskit"]["reason"]


def test_framework_benchmark_qiskit_cpu_backend_falls_back_without_aer(monkeypatch):
    pytest.importorskip("qiskit.providers.basic_provider")
    module = _load_benchmark_module("framework_benchmark_qiskit_fallback")
    real_import = builtins.__import__

    def reject_aer(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "qiskit_aer":
            raise ImportError("forced missing qiskit-aer")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_aer)
    backend, label = module.make_qiskit_cpu_backend()

    assert label == "BasicSimulator"
    assert backend.__class__.__name__ == "BasicSimulator"


def test_framework_benchmark_pennylane_qft_uses_provided_device():
    qml = pytest.importorskip("pennylane")
    module = _load_benchmark_module("framework_benchmark_pennylane_qft")

    device = qml.device("default.qubit", wires=2)
    circuit = module.generate_pennylane_qft(2, device=device)

    assert getattr(circuit, "device", None) is device
    state = np.asarray(circuit())
    assert state.shape == (4,)
