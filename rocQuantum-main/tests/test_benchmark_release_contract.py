"""Contracts for release benchmark artifact generation."""

from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
MANIFEST = os.path.join(PROJECT_ROOT, "benchmarks", "benchmark_manifest.json")
RUNNER = os.path.join(PROJECT_ROOT, "benchmarks", "run_release_benchmarks.py")
DISTRIBUTED_BENCHMARK = os.path.join(
    PROJECT_ROOT,
    "benchmarks",
    "distributed_reduction_benchmark.cpp",
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_release_benchmarks", RUNNER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestBenchmarkReleaseContract(unittest.TestCase):
    def test_manifest_covers_release_benchmark_axes(self):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        ids = {entry["id"] for entry in manifest["benchmarks"]}
        by_id = {entry["id"]: entry for entry in manifest["benchmarks"]}
        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("statevec_core_paths", ids)
        self.assertIn("distributed_reduction_rccl_vs_host", ids)
        self.assertIn("tensornet_contraction", ids)
        self.assertIn("densitymat_channel_sampling", ids)
        self.assertEqual(
            by_id["distributed_reduction_rccl_vs_host"]["speedup_thresholds"],
            {
                "expectation_ms": 1.0,
                "dense_expectation_ms": 1.0,
                "sparse_moments_ms": 1.0,
                "generic_matrix_ms": 1.0,
            },
        )

        covered = {cover for entry in manifest["benchmarks"] for cover in entry["covers"]}
        for required in {
            "statevec_fastpath_vs_fallback",
            "gate_fusion_vs_unfused",
            "sampling_observe",
            "distributed_rccl_vs_host_fallback",
            "distributed_sparse_moments",
            "tensor_contraction",
            "densitymat_channel",
            "densitymat_sampling",
        }:
            self.assertIn(required, covered)

    def test_release_runner_emits_skip_artifacts_without_native_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = os.path.join(tmp, "missing-build")
            output_dir = os.path.join(tmp, "artifacts")
            completed = subprocess.run(
                [
                    sys.executable,
                    RUNNER,
                    "--build-dir",
                    build_dir,
                    "--output-dir",
                    output_dir,
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary_path = os.path.join(output_dir, "benchmark-summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            self.assertEqual(summary["schema_version"], 1)
            self.assertTrue(summary["results"])
            self.assertTrue(all(result["status"] == "skipped" for result in summary["results"]))
            for result in summary["results"]:
                self.assertTrue(os.path.exists(result["output"]))
            markdown_path = os.path.join(output_dir, "benchmark-summary.md")
            self.assertTrue(os.path.exists(markdown_path))
            with open(markdown_path, "r", encoding="utf-8") as f:
                markdown = f.read()
            self.assertIn("# Release Benchmark Summary", markdown)
            self.assertIn("ROCm device detected", markdown)
            self.assertIn("skipped", markdown)

    def test_release_runner_extracts_distributed_speedup_metrics(self):
        runner = _load_runner_module()
        speedups = runner.extract_case_speedups(
            {
                "cases": [
                    {
                        "name": "rccl",
                        "status": 0,
                        "expectation_ms": 2.0,
                        "dense_expectation_ms": 4.0,
                        "sparse_moments_ms": 5.0,
                        "sampling_ms": 10.0,
                    },
                    {
                        "name": "host_fallback",
                        "status": 0,
                        "expectation_ms": 8.0,
                        "dense_expectation_ms": 20.0,
                        "sparse_moments_ms": 25.0,
                        "sampling_ms": 8.0,
                    },
                ]
            },
            thresholds={"expectation_ms": 3.0, "sampling_ms": 1.0},
        )

        by_metric = {entry["metric"]: entry for entry in speedups}
        self.assertEqual(by_metric["expectation_ms"]["baseline_case"], "host_fallback")
        self.assertEqual(by_metric["expectation_ms"]["optimized_case"], "rccl")
        self.assertAlmostEqual(by_metric["expectation_ms"]["speedup"], 4.0)
        self.assertEqual(by_metric["expectation_ms"]["minimum_speedup"], 3.0)
        self.assertTrue(by_metric["expectation_ms"]["passes_threshold"])
        self.assertAlmostEqual(by_metric["dense_expectation_ms"]["speedup"], 5.0)
        self.assertAlmostEqual(by_metric["sparse_moments_ms"]["speedup"], 5.0)
        self.assertFalse(by_metric["sampling_ms"]["faster_than_baseline"])
        self.assertFalse(by_metric["sampling_ms"]["passes_threshold"])

    def test_release_runner_adds_speedups_to_summary_when_output_has_cases(self):
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_benchmark = tmp_path / "fake_benchmark.py"
            fake_benchmark.write_text(
                "import json, sys\n"
                "payload = {\n"
                "    'cases': [\n"
                "        {'name': 'rccl', 'status': 0, 'expectation_ms': 2.0, 'sparse_moments_ms': 3.0},\n"
                "        {'name': 'host_fallback', 'status': 0, 'expectation_ms': 10.0, 'sparse_moments_ms': 9.0},\n"
                "    ]\n"
                "}\n"
                "with open(sys.argv[1], 'w', encoding='utf-8') as f:\n"
                "    json.dump(payload, f)\n",
                encoding="utf-8",
            )
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmarks": [
                            {
                                "id": "fake_distributed",
                                "category": "distributed",
                                "executable": sys.executable,
                                "output": "fake-distributed.json",
                                "args": [str(fake_benchmark), "{output}"],
                                "requires_rocm_device": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = runner.run(
                manifest_path=manifest_path,
                build_dir=tmp_path / "build",
                output_dir=tmp_path / "artifacts",
            )
            markdown_path = tmp_path / "artifacts" / "benchmark-summary.md"
            markdown = markdown_path.read_text(encoding="utf-8")

        result = summary["results"][0]
        by_metric = {entry["metric"]: entry for entry in result["speedups"]}
        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(by_metric["expectation_ms"]["speedup"], 5.0)
        self.assertAlmostEqual(by_metric["sparse_moments_ms"]["speedup"], 3.0)
        self.assertIn("`expectation_ms`: 5.000x", markdown)
        self.assertIn("`sparse_moments_ms`: 3.000x", markdown)

    def test_release_runner_fails_configured_speedup_thresholds(self):
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_benchmark = tmp_path / "fake_benchmark.py"
            fake_benchmark.write_text(
                "import json, sys\n"
                "payload = {'cases': [\n"
                "    {'name': 'rccl', 'status': 0, 'expectation_ms': 5.0},\n"
                "    {'name': 'host_fallback', 'status': 0, 'expectation_ms': 6.0},\n"
                "]}\n"
                "with open(sys.argv[1], 'w', encoding='utf-8') as f:\n"
                "    json.dump(payload, f)\n",
                encoding="utf-8",
            )
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmarks": [
                            {
                                "id": "fake_distributed",
                                "category": "distributed",
                                "executable": sys.executable,
                                "output": "fake-distributed.json",
                                "args": [str(fake_benchmark), "{output}"],
                                "requires_rocm_device": False,
                                "speedup_thresholds": {"expectation_ms": 2.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = runner.run(
                manifest_path=manifest_path,
                build_dir=tmp_path / "build",
                output_dir=tmp_path / "artifacts",
            )

        result = summary["results"][0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "one or more configured speedup thresholds were not met")
        self.assertEqual(result["threshold_failures"][0]["metric"], "expectation_ms")
        self.assertFalse(result["threshold_failures"][0]["passes_threshold"])

    def test_release_runner_fails_speedup_trend_regressions(self):
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_benchmark = tmp_path / "fake_benchmark.py"
            fake_benchmark.write_text(
                "import json, sys\n"
                "payload = {'cases': [\n"
                "    {'name': 'rccl', 'status': 0, 'expectation_ms': 4.0},\n"
                "    {'name': 'host_fallback', 'status': 0, 'expectation_ms': 8.0},\n"
                "]}\n"
                "with open(sys.argv[1], 'w', encoding='utf-8') as f:\n"
                "    json.dump(payload, f)\n",
                encoding="utf-8",
            )
            baseline_path = tmp_path / "baseline-summary.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "results": [
                            {
                                "id": "fake_distributed",
                                "status": "passed",
                                "speedups": [
                                    {
                                        "metric": "expectation_ms",
                                        "optimized_case": "rccl",
                                        "baseline_case": "host_fallback",
                                        "speedup": 4.0,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmarks": [
                            {
                                "id": "fake_distributed",
                                "category": "distributed",
                                "executable": sys.executable,
                                "output": "fake-distributed.json",
                                "args": [str(fake_benchmark), "{output}"],
                                "requires_rocm_device": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = runner.run(
                manifest_path=manifest_path,
                build_dir=tmp_path / "build",
                output_dir=tmp_path / "artifacts",
                baseline_summary_path=baseline_path,
                max_speedup_regression=0.25,
            )
            markdown = (tmp_path / "artifacts" / "benchmark-summary.md").read_text(encoding="utf-8")

        result = summary["results"][0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["failure_reason"],
            "one or more speedup trend gates regressed versus baseline",
        )
        self.assertEqual(result["trend_regressions"][0]["metric"], "expectation_ms")
        self.assertAlmostEqual(result["trend_regressions"][0]["baseline_speedup"], 4.0)
        self.assertAlmostEqual(result["trend_regressions"][0]["minimum_trend_speedup"], 3.0)
        self.assertFalse(result["trend_regressions"][0]["passes_trend_gate"])
        self.assertIn("trend baseline 4.000x min 3.000x=fail", markdown)

    def test_release_runner_updates_bounded_history_artifact(self):
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_benchmark = tmp_path / "fake_benchmark.py"
            fake_benchmark.write_text(
                "import json, sys\n"
                "rccl_ms = float(sys.argv[2])\n"
                "payload = {'cases': [\n"
                "    {'name': 'rccl', 'status': 0, 'expectation_ms': rccl_ms},\n"
                "    {'name': 'host_fallback', 'status': 0, 'expectation_ms': 8.0},\n"
                "]}\n"
                "with open(sys.argv[1], 'w', encoding='utf-8') as f:\n"
                "    json.dump(payload, f)\n",
                encoding="utf-8",
            )
            history_path = tmp_path / "benchmark-history.json"

            summary = {}
            for index, rccl_ms in enumerate([4.0, 2.0, 1.0]):
                manifest_path = tmp_path / f"manifest-{index}.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "benchmarks": [
                                {
                                    "id": "fake_distributed",
                                    "category": "distributed",
                                    "executable": sys.executable,
                                    "output": "fake-distributed.json",
                                    "args": [str(fake_benchmark), "{output}", str(rccl_ms)],
                                    "requires_rocm_device": False,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                summary = runner.run(
                    manifest_path=manifest_path,
                    build_dir=tmp_path / "build",
                    output_dir=tmp_path / f"artifacts-{index}",
                    history_path=history_path,
                    history_limit=2,
                )

            history = json.loads(history_path.read_text(encoding="utf-8"))
            markdown = (tmp_path / "artifacts-2" / "benchmark-summary.md").read_text(encoding="utf-8")

        self.assertEqual(summary["history"], str(history_path))
        self.assertEqual(summary["history_entries"], 2)
        self.assertEqual(history["schema_version"], 1)
        self.assertEqual(history["history_limit"], 2)
        self.assertEqual(len(history["runs"]), 2)
        first_retained = history["runs"][0]["results"][0]["speedups"][0]
        latest = history["runs"][1]["results"][0]["speedups"][0]
        self.assertAlmostEqual(first_retained["speedup"], 4.0)
        self.assertAlmostEqual(latest["speedup"], 8.0)
        self.assertIn("History entries: 2", markdown)

    def test_cmake_exposes_release_benchmark_targets(self):
        statevec_cmake = os.path.join(PROJECT_ROOT, "rocquantum", "src", "hipStateVec", "CMakeLists.txt")
        tensornet_cmake = os.path.join(PROJECT_ROOT, "rocquantum", "src", "hipTensorNet", "CMakeLists.txt")
        density_cmake = os.path.join(PROJECT_ROOT, "rocquantum", "src", "hipDensityMat", "CMakeLists.txt")

        with open(statevec_cmake, "r", encoding="utf-8") as f:
            statevec = f.read()
        with open(tensornet_cmake, "r", encoding="utf-8") as f:
            tensornet = f.read()
        with open(density_cmake, "r", encoding="utf-8") as f:
            density = f.read()

        self.assertIn("benchmark_hipStateVec_core_paths", statevec)
        self.assertIn("benchmark_hipStateVec_distributed_reductions", statevec)
        self.assertIn("benchmark_hipTensorNet_contraction", tensornet)
        self.assertIn("benchmark_hipDensityMat_channel_sampling", density)

    def test_distributed_benchmark_covers_rccl_dense_sparse_and_generic_matrix_paths(self):
        with open(DISTRIBUTED_BENCHMARK, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("dense_expectation_ms", source)
        self.assertIn("sparse_moments_ms", source)
        self.assertIn("generic_matrix_ms", source)
        self.assertIn("rocsvGetExpectationMatrix", source)
        self.assertIn("rocsvGetSparseMatrixMoments", source)
        self.assertIn("upload_rank_local_z_csr", source)
        self.assertIn("rocsvApplyMatrix", source)
        self.assertIn("ROCQ_DISTRIBUTED_COMM", source)
        self.assertIn("ROCQ_DISTRIBUTED_FALLBACK_MODE", source)

    def test_ci_uploads_benchmark_artifacts(self):
        workflow_paths = [
            os.path.join(REPO_ROOT, ".github", "workflows", "rocm-linux-build.yml"),
            os.path.join(REPO_ROOT, ".github", "workflows", "rocm-ci.yml"),
            os.path.join(REPO_ROOT, ".github", "workflows", "rocm-nightly.yml"),
        ]
        combined = ""
        for workflow_path in workflow_paths:
            with open(workflow_path, "r", encoding="utf-8") as f:
                combined += f.read()

        self.assertIn("benchmarks/run_release_benchmarks.py", combined)
        self.assertIn("benchmark-artifacts", combined)
        self.assertIn("benchmark-summary.md", combined)
        self.assertIn("benchmark-history.json", combined)
        self.assertIn("GITHUB_STEP_SUMMARY", combined)
        self.assertIn("--fail-on-error", combined)
        self.assertIn("--history-path", combined)
        self.assertGreaterEqual(combined.count("set -o pipefail"), 3)
        self.assertIn("actions/cache/restore@v4", combined)
        self.assertIn("actions/cache/save@v4", combined)
        self.assertIn("benchmark-baseline-rocm-runtime", combined)
        self.assertIn("benchmark-baseline-rocm-nightly", combined)
        self.assertIn("BASELINE_ARGS=(--baseline-summary", combined)
        self.assertIn("actions/upload-artifact@v4", combined)

    def test_readme_describes_release_benchmark_registry(self):
        with open(os.path.join(PROJECT_ROOT, "README.md"), "r", encoding="utf-8") as f:
            readme = f.read()

        self.assertIn("benchmarks/benchmark_manifest.json", readme)
        self.assertIn("benchmarks/run_release_benchmarks.py", readme)
        self.assertIn("benchmark-summary.json", readme)
        self.assertIn("benchmark-summary.md", readme)
        self.assertIn("benchmark-history.json", readme)
        self.assertIn("speedup ratios", readme)
        self.assertIn("--baseline-summary", readme)
        self.assertIn("self-hosted ROCm workflows restore the previous benchmark summary and bounded history", readme)
        self.assertIn("dense expectation, sparse moments, and generic matrix", readme)


if __name__ == "__main__":
    unittest.main()
