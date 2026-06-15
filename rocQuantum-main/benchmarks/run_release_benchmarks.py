"""Run release benchmark binaries and collect JSON artifacts.

The runner is intentionally tolerant of missing native binaries or missing ROCm
devices so CPU-only CI can still publish a truthful skip manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "benchmark_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _has_rocm_device() -> bool:
    if os.environ.get("ROCQ_BENCHMARK_ASSUME_ROCM_DEVICE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if os.name == "nt":
        return False
    return Path("/dev/kfd").exists()


def _resolve_executable(build_dir: Path, executable: str) -> Path:
    candidate = build_dir / executable
    if candidate.exists():
        return candidate
    if os.name == "nt" and candidate.with_suffix(candidate.suffix + ".exe").exists():
        return candidate.with_suffix(candidate.suffix + ".exe")
    return candidate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_bool(value: bool) -> str:
    return "yes" if value else "no"


def _speedup_thresholds(entry: dict[str, Any]) -> dict[str, float]:
    raw = entry.get("speedup_thresholds")
    if not isinstance(raw, dict):
        return {}
    thresholds: dict[str, float] = {}
    for metric, value in raw.items():
        if isinstance(metric, str) and isinstance(value, (int, float)) and value > 0:
            thresholds[metric] = float(value)
    return thresholds


def _speedups_by_metric(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    speedups = result.get("speedups")
    if not isinstance(speedups, list):
        return {}
    return {
        speedup["metric"]: speedup
        for speedup in speedups
        if isinstance(speedup, dict) and isinstance(speedup.get("metric"), str)
    }


def extract_case_speedups(
    payload: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Extract host-fallback-over-RCCL speedups from benchmark case timings."""
    thresholds = thresholds or {}
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return []

    by_name = {
        case.get("name"): case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("name"), str)
    }
    rccl = by_name.get("rccl")
    host_fallback = by_name.get("host_fallback")
    if not isinstance(rccl, dict) or not isinstance(host_fallback, dict):
        return []

    metrics = sorted(
        key
        for key in set(rccl).intersection(host_fallback)
        if key.endswith("_ms") and isinstance(rccl.get(key), (int, float)) and isinstance(host_fallback.get(key), (int, float))
    )
    speedups: list[dict[str, Any]] = []
    for metric in metrics:
        optimized_ms = float(rccl[metric])
        baseline_ms = float(host_fallback[metric])
        if optimized_ms <= 0.0 or baseline_ms <= 0.0:
            continue
        speedup = baseline_ms / optimized_ms
        entry = {
            "metric": metric,
            "optimized_case": "rccl",
            "baseline_case": "host_fallback",
            "optimized_ms": optimized_ms,
            "baseline_ms": baseline_ms,
            "speedup": speedup,
            "faster_than_baseline": speedup >= 1.0,
        }
        if metric in thresholds:
            minimum = thresholds[metric]
            entry["minimum_speedup"] = minimum
            entry["passes_threshold"] = speedup >= minimum
        speedups.append(entry)
    return speedups


def _format_speedup_entry(entry: dict[str, Any]) -> str:
    text = (
        f"`{entry['metric']}`: {entry['speedup']:.3f}x "
        f"({entry['baseline_case']} / {entry['optimized_case']})"
    )
    if "minimum_speedup" in entry:
        status = "pass" if entry.get("passes_threshold") else "fail"
        text += f" threshold {entry['minimum_speedup']:.3f}x={status}"
    if "baseline_speedup" in entry:
        status = "pass" if entry.get("passes_trend_gate") else "fail"
        text += (
            f" trend baseline {entry['baseline_speedup']:.3f}x"
            f" min {entry['minimum_trend_speedup']:.3f}x={status}"
        )
    return text


def apply_speedup_trend_gate(
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    max_speedup_regression: float,
) -> None:
    if max_speedup_regression < 0.0 or max_speedup_regression >= 1.0:
        raise ValueError("max_speedup_regression must be >= 0 and < 1")

    baseline_results = {
        result.get("id"): result
        for result in baseline_summary.get("results", [])
        if isinstance(result, dict) and isinstance(result.get("id"), str)
    }
    for result in summary.get("results", []):
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            continue
        baseline_result = baseline_results.get(result["id"])
        if not isinstance(baseline_result, dict):
            continue

        baseline_speedups = _speedups_by_metric(baseline_result)
        trend_regressions: list[dict[str, Any]] = []
        for metric, speedup in _speedups_by_metric(result).items():
            baseline_speedup = baseline_speedups.get(metric)
            if not isinstance(baseline_speedup, dict):
                continue
            current_value = speedup.get("speedup")
            baseline_value = baseline_speedup.get("speedup")
            if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                continue
            if current_value <= 0.0 or baseline_value <= 0.0:
                continue

            minimum = float(baseline_value) * (1.0 - max_speedup_regression)
            speedup["baseline_speedup"] = float(baseline_value)
            speedup["max_speedup_regression"] = max_speedup_regression
            speedup["minimum_trend_speedup"] = minimum
            speedup["passes_trend_gate"] = float(current_value) >= minimum
            if not speedup["passes_trend_gate"]:
                trend_regressions.append(speedup)

        if trend_regressions:
            result["status"] = "failed"
            result["trend_regressions"] = trend_regressions
            reason = "one or more speedup trend gates regressed versus baseline"
            if result.get("failure_reason"):
                result["failure_reason"] = f"{result['failure_reason']}; {reason}"
            else:
                result["failure_reason"] = reason


def format_benchmark_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Release Benchmark Summary",
        "",
        f"- ROCm device detected: {_markdown_bool(bool(summary.get('has_rocm_device')))}",
        f"- Output directory: `{summary.get('output_dir', '')}`",
        "",
        "| Benchmark | Status | Speedups | Output |",
        "| --- | --- | --- | --- |",
    ]
    for result in summary.get("results", []):
        if not isinstance(result, dict):
            continue
        speedups = result.get("speedups")
        if isinstance(speedups, list) and speedups:
            speedup_text = "<br>".join(
                _format_speedup_entry(entry)
                for entry in speedups
                if isinstance(entry, dict)
                and {"metric", "speedup", "baseline_case", "optimized_case"}.issubset(entry)
            )
        else:
            speedup_text = str(result.get("reason") or result.get("analysis_warning") or "-")
        lines.append(
            f"| `{result.get('id', '')}` | `{result.get('status', '')}` | "
            f"{speedup_text} | `{result.get('output', '')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _skip_result(entry: dict[str, Any], output_path: Path, reason: str, executable: Path) -> dict[str, Any]:
    payload = {
        "benchmark": entry["id"],
        "status": "skipped",
        "reason": reason,
        "expected_executable": str(executable),
        "created_at_utc": _utc_now(),
    }
    _write_json(output_path, payload)
    return {
        "id": entry["id"],
        "category": entry.get("category"),
        "status": "skipped",
        "reason": reason,
        "output": str(output_path),
        "executable": str(executable),
        "duration_seconds": 0.0,
    }


def _run_entry(entry: dict[str, Any], build_dir: Path, output_dir: Path, has_device: bool) -> dict[str, Any]:
    output_path = output_dir / entry["output"]
    executable = _resolve_executable(build_dir, entry["executable"])

    if entry.get("requires_rocm_device", False) and not has_device:
        return _skip_result(entry, output_path, "ROCm runtime device /dev/kfd is not available", executable)
    if not executable.exists():
        return _skip_result(entry, output_path, "benchmark executable was not found", executable)

    args = [str(executable)]
    for arg in entry.get("args", []):
        args.append(str(output_path) if arg == "{output}" else str(arg))

    stdout_path = output_dir / f"{entry['id']}.stdout.txt"
    stderr_path = output_dir / f"{entry['id']}.stderr.txt"
    start = time.perf_counter()
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    duration = time.perf_counter() - start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    status = "passed" if completed.returncode == 0 else "failed"
    if not output_path.exists():
        _write_json(
            output_path,
            {
                "benchmark": entry["id"],
                "status": status,
                "returncode": completed.returncode,
                "reason": "benchmark did not write its declared JSON output",
                "created_at_utc": _utc_now(),
            },
        )

    result = {
        "id": entry["id"],
        "category": entry.get("category"),
        "status": status,
        "returncode": completed.returncode,
        "output": str(output_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "executable": str(executable),
        "duration_seconds": duration,
    }
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        speedups = extract_case_speedups(payload, thresholds=_speedup_thresholds(entry))
        if speedups:
            result["speedups"] = speedups
            threshold_failures = [
                speedup
                for speedup in speedups
                if speedup.get("passes_threshold") is False
            ]
            if threshold_failures:
                result["status"] = "failed"
                result["threshold_failures"] = threshold_failures
                result["failure_reason"] = "one or more configured speedup thresholds were not met"
    except (OSError, json.JSONDecodeError) as exc:
        result["analysis_warning"] = f"could not analyze benchmark output: {exc}"
    return result


def run(
    manifest_path: Path,
    build_dir: Path,
    output_dir: Path,
    baseline_summary_path: Path | None = None,
    max_speedup_regression: float = 0.20,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    has_device = _has_rocm_device()
    results = [
        _run_entry(entry, build_dir=build_dir, output_dir=output_dir, has_device=has_device)
        for entry in manifest["benchmarks"]
    ]
    summary = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "manifest": str(manifest_path),
        "build_dir": str(build_dir),
        "output_dir": str(output_dir),
        "has_rocm_device": has_device,
        "results": results,
    }
    if baseline_summary_path is not None:
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        summary["baseline_summary"] = str(baseline_summary_path)
        summary["max_speedup_regression"] = max_speedup_regression
        apply_speedup_trend_gate(summary, baseline_summary, max_speedup_regression)
    markdown_path = output_dir / "benchmark-summary.md"
    summary["markdown_summary"] = str(markdown_path)
    _write_json(output_dir / "benchmark-summary.json", summary)
    markdown_path.write_text(format_benchmark_summary_markdown(summary), encoding="utf-8")
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    baseline_default = os.environ.get("ROCQ_BENCHMARK_BASELINE_SUMMARY", "").strip()
    regression_default = float(os.environ.get("ROCQ_BENCHMARK_MAX_SPEEDUP_REGRESSION", "0.20"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build-dir", type=Path, default=PROJECT_ROOT / "build-ci")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmark-artifacts")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return non-zero when any discovered benchmark executable fails.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path(baseline_default) if baseline_default else None,
        help="Optional previous benchmark-summary.json used to fail speedup trend regressions.",
    )
    parser.add_argument(
        "--max-speedup-regression",
        type=float,
        default=regression_default,
        help="Allowed fractional speedup drop versus --baseline-summary before failing, default 0.20.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = run(
        manifest_path=args.manifest.resolve(),
        build_dir=args.build_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        baseline_summary_path=args.baseline_summary.resolve() if args.baseline_summary else None,
        max_speedup_regression=args.max_speedup_regression,
    )
    failed = [result for result in summary["results"] if result["status"] == "failed"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_error and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
