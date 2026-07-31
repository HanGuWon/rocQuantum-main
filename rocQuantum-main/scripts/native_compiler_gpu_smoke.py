#!/usr/bin/env python3
"""Require a real MLIR -> HipStateVec Bell-state execution on a ROCm GPU."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def validate_bell_state(statevector: Iterable[complex], *, atol: float = 1e-5) -> dict[str, float]:
    """Validate the exact two-qubit Bell state and return numerical evidence."""

    state = [complex(amplitude) for amplitude in statevector]
    expected = [1 / math.sqrt(2), 0j, 0j, 1 / math.sqrt(2)]
    if len(state) != len(expected):
        raise AssertionError(
            f"Bell state must contain {len(expected)} amplitudes, got {len(state)}."
        )
    if any(
        not math.isfinite(amplitude.real) or not math.isfinite(amplitude.imag)
        for amplitude in state
    ):
        raise AssertionError(f"Bell state contains a non-finite amplitude: {state!r}")

    max_error = max(abs(actual - wanted) for actual, wanted in zip(state, expected))
    norm = sum(abs(amplitude) ** 2 for amplitude in state)
    if max_error > atol:
        raise AssertionError(
            f"Bell state mismatch (max error {max_error:.6g} > {atol:.6g}): {state!r}"
        )
    if abs(norm - 1.0) > atol:
        raise AssertionError(f"Bell state norm mismatch: {norm:.12g}")
    return {"max_amplitude_error": max_error, "state_norm": norm}


def _write_json(report: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(report: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Native Compiler GPU Smoke",
        "",
        f"- Status: `{report['status']}`",
        f"- Evidence: `{report['evidence_kind']}`",
        f"- Backend: `{report.get('backend', '')}`",
        f"- Elapsed: `{report.get('elapsed_ms', 0):.3f} ms`",
    ]
    if "max_amplitude_error" in report:
        lines.append(
            f"- Maximum Bell-state amplitude error: `{report['max_amplitude_error']:.6g}`"
        )
    if "state_norm" in report:
        lines.append(f"- State norm: `{report['state_norm']:.12g}`")
    if report.get("error"):
        lines.append(f"- Error: `{report['error']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_bell_kernel(rocq_module: Any) -> Any:
    @rocq_module.kernel
    def bell_program() -> None:
        qubits = rocq_module.qvec(2)
        rocq_module.h(qubits[0])
        rocq_module.cnot(qubits[0], qubits[1])

    return bell_program


def run_native_compiler_gpu_smoke() -> dict[str, Any]:
    """Execute the strict public compiler API without diagnostic fallbacks."""

    if not Path("/dev/kfd").exists():
        raise RuntimeError("A real ROCm device at /dev/kfd is required.")

    import rocq
    import rocquantum_bind

    if not bool(getattr(rocquantum_bind, "MLIR_COMPILER_ENABLED", False)):
        raise RuntimeError("The native binding was built without MLIR compiler support.")
    if not bool(
        getattr(rocquantum_bind, "MLIR_COMPILER_GPU_EXECUTION_ENABLED", False)
    ):
        raise RuntimeError(
            "The native binding was built without MLIR compiler GPU execution support."
        )

    bell_program = _make_bell_kernel(rocq)
    started = time.perf_counter()
    state = rocq.compile_and_execute(
        bell_program,
        compiler_backend="hip_statevec",
        strict=True,
    )
    metrics = validate_bell_state(state)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": 1,
        "suite": "native_compiler_gpu_smoke",
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend": "hip_statevec",
        "requires_rocm_device": True,
        "has_rocm_device": True,
        "strict": True,
        "native_mlir_compiler": True,
        "native_gpu_execution": True,
        "evidence_kind": "native_rocm",
        "elapsed_ms": round(elapsed_ms, 3),
        **metrics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_native_compiler_gpu_smoke()
    except Exception as exc:  # noqa: BLE001 - fail-closed executable boundary
        report = {
            "schema_version": 1,
            "suite": "native_compiler_gpu_smoke",
            "status": "failed",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "backend": "hip_statevec",
            "requires_rocm_device": True,
            "has_rocm_device": Path("/dev/kfd").exists(),
            "strict": True,
            "evidence_kind": "native_rocm_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(report, args.json_output)
        _write_markdown(report, args.markdown_output)
        print(f"native compiler GPU smoke failed: {exc}", file=sys.stderr)
        return 1

    _write_json(report, args.json_output)
    _write_markdown(report, args.markdown_output)
    print("native compiler GPU smoke: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
