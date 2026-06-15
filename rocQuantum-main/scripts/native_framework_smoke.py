#!/usr/bin/env python3
"""Run a tiny native ROCm framework-integration smoke suite."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = PROJECT_ROOT / "integrations"
for path in [
    PROJECT_ROOT,
    INTEGRATIONS / "pennylane-rocq",
    INTEGRATIONS / "qiskit-rocquantum-provider",
    INTEGRATIONS / "cirq-rocm",
]:
    sys.path.insert(0, str(path))


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _rocm_device_probe() -> dict[str, Any]:
    if Path("/dev/kfd").exists():
        return {"has_rocm_device": True, "device_probe": "actual_dev_kfd"}
    if _env_truthy("ROCQ_NATIVE_SMOKE_ASSUME_ROCM_DEVICE"):
        return {"has_rocm_device": True, "device_probe": "assumed_rocm_device"}
    return {"has_rocm_device": False, "device_probe": "missing_dev_kfd"}


def _require_rocm_device() -> dict[str, Any]:
    probe = _rocm_device_probe()
    if not probe["has_rocm_device"]:
        raise RuntimeError("Native framework smoke requires a ROCm runtime device at /dev/kfd.")
    return probe


def _assert_bell_state(statevector: object) -> None:
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex128)
    if state.shape != expected.shape or not np.allclose(state, expected, atol=1e-5):
        raise AssertionError(f"Bell state mismatch: got {state!r}")


def smoke_native_binding() -> None:
    binding = importlib.import_module("rocquantum_bind")
    simulator = binding.QuantumSimulator(2)
    simulator.apply_gate("H", [0], [])
    simulator.apply_gate("CNOT", [0, 1], [])
    _assert_bell_state(simulator.get_statevector())
    print("native binding smoke: ok")


def smoke_pennylane() -> None:
    qml = importlib.import_module("pennylane")
    pennylane_rocq = importlib.import_module("pennylane_rocq")

    dev = pennylane_rocq.RocqDevice(wires=2, shots=None)

    @qml.qnode(dev)
    def bell_state():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    _assert_bell_state(bell_state())
    print("PennyLane smoke: ok")


def smoke_qiskit() -> None:
    qiskit = importlib.import_module("qiskit")
    provider_module = importlib.import_module("qiskit_rocquantum_provider.provider")

    circuit = qiskit.QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    backend = provider_module.RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=32, sampling=True).result()
    counts = result.get_counts()
    if sum(int(value) for value in counts.values()) != 32:
        raise AssertionError(f"Qiskit smoke returned invalid shot count: {counts!r}")
    print("Qiskit smoke: ok")


def smoke_cirq() -> None:
    cirq = importlib.import_module("cirq")
    simulator_module = importlib.import_module("cirq_rocm")

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = simulator_module.RocQuantumSimulator()
    result = simulator.simulate(circuit)
    _assert_bell_state(result.final_state_vector)
    print("Cirq smoke: ok")


def _write_report(report: dict[str, Any], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"native framework smoke JSON: {output_path}")


def _markdown_bool(value: bool) -> str:
    return "yes" if value else "no"


def format_native_smoke_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native Framework Smoke",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- ROCm device detected: {_markdown_bool(bool(report.get('has_rocm_device')))}",
        f"- ROCm device probe: `{report.get('device_probe', '')}`",
        f"- Native ROCm evidence: {_markdown_bool(bool(report.get('native_rocm_evidence')))}",
    ]
    if report.get("native_rocm_evidence_required"):
        lines.append("- Native ROCm evidence required: yes")
    if report.get("native_rocm_evidence_failure"):
        lines.append(f"- Native ROCm evidence gate: failed ({report['native_rocm_evidence_failure']})")
    if report.get("reason"):
        lines.append(f"- Reason: {report['reason']}")
    lines.extend(
        [
            "",
            "| Check | Status | Evidence |",
            "| --- | --- | --- |",
        ]
    )
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        lines.append(
            f"| `{result.get('name', '')}` | `{result.get('status', '')}` | "
            f"`{result.get('evidence_kind', '')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_markdown_report(report: dict[str, Any], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_native_smoke_markdown(report), encoding="utf-8")
    print(f"native framework smoke Markdown: {output_path}")


def _run_step(name: str, callback: Callable[[], None]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        callback()
    except Exception as exc:  # noqa: BLE001 - top-level smoke report boundary
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"{name} smoke: failed: {exc}", file=sys.stderr)
        return {
            "name": name,
            "status": "failed",
            "elapsed_ms": round(elapsed_ms, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "name": name,
        "status": "passed",
        "elapsed_ms": round(elapsed_ms, 3),
    }


def _native_evidence_kind(status: str, device_probe: str) -> str:
    if status == "skipped":
        return "skip"
    if status == "passed" and device_probe == "actual_dev_kfd":
        return "native_rocm"
    if device_probe == "actual_dev_kfd":
        return "native_rocm_failed"
    if device_probe == "assumed_rocm_device":
        return "assumed_rocm_device"
    return "non_native_or_failed"


def _make_report(
    status: str,
    results: list[dict[str, Any]],
    device_probe: dict[str, Any] | None = None,
    require_native_rocm_evidence: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    probe = dict(device_probe or _rocm_device_probe())
    actual_rocm_device = probe.get("device_probe") == "actual_dev_kfd"
    evidence_count = 0
    annotated_results = []
    for result in results:
        annotated = dict(result)
        step_is_evidence = actual_rocm_device and annotated.get("status") == "passed"
        annotated["native_rocm_evidence"] = bool(step_is_evidence)
        annotated["evidence_kind"] = "native_rocm" if step_is_evidence else _native_evidence_kind(status, probe["device_probe"])
        if step_is_evidence:
            evidence_count += 1
        annotated_results.append(annotated)
    suite_is_evidence = bool(results) and status == "passed" and evidence_count == len(results)
    report = {
        "schema_version": 1,
        "suite": "native_framework_smoke",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requires_rocm_device": True,
        "has_rocm_device": bool(probe.get("has_rocm_device")),
        "device_probe": probe["device_probe"],
        "native_rocm_evidence": suite_is_evidence,
        "native_rocm_evidence_count": evidence_count,
        "native_rocm_evidence_required": require_native_rocm_evidence,
        "evidence_kind": "native_rocm" if suite_is_evidence else _native_evidence_kind(status, probe["device_probe"]),
        "results": annotated_results,
    }
    if require_native_rocm_evidence and not suite_is_evidence:
        report["native_rocm_evidence_failure"] = (
            "no passed native ROCm framework smoke suite was produced"
        )
    report.update(extra)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable native smoke report.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional path for a Markdown native smoke summary.",
    )
    parser.add_argument(
        "--allow-missing-device-skip",
        action="store_true",
        help="Return success with a skipped report when /dev/kfd is unavailable.",
    )
    parser.add_argument(
        "--require-native-rocm-evidence",
        action="store_true",
        default=_env_truthy("ROCQ_NATIVE_SMOKE_REQUIRE_NATIVE_EVIDENCE"),
        help="Return non-zero unless the full smoke suite passes on an actual /dev/kfd ROCm device.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        device_probe = _require_rocm_device()
    except RuntimeError as exc:
        status = "skipped" if args.allow_missing_device_skip else "failed"
        report = _make_report(
            status,
            [],
            _rocm_device_probe(),
            require_native_rocm_evidence=args.require_native_rocm_evidence,
            reason=str(exc),
        )
        _write_report(report, args.json_output)
        _write_markdown_report(report, args.markdown_output)
        if args.allow_missing_device_skip and not report.get("native_rocm_evidence_failure"):
            print(f"native framework smoke: skipped: {exc}")
            return 0
        print(f"native framework smoke: failed: {exc}", file=sys.stderr)
        return 1

    steps = [
        ("native_binding", smoke_native_binding),
        ("pennylane", smoke_pennylane),
        ("qiskit", smoke_qiskit),
        ("cirq", smoke_cirq),
    ]
    results = [_run_step(name, callback) for name, callback in steps]
    status = "passed" if all(result["status"] == "passed" for result in results) else "failed"
    report = _make_report(
        status,
        results,
        device_probe,
        require_native_rocm_evidence=args.require_native_rocm_evidence,
    )
    _write_report(report, args.json_output)
    _write_markdown_report(report, args.markdown_output)
    if report.get("native_rocm_evidence_failure"):
        return 1
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
