"""Contracts for release benchmark artifact generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
MANIFEST = os.path.join(PROJECT_ROOT, "benchmarks", "benchmark_manifest.json")
RUNNER = os.path.join(PROJECT_ROOT, "benchmarks", "run_release_benchmarks.py")
DISTRIBUTED_BENCHMARK = os.path.join(
    PROJECT_ROOT,
    "benchmarks",
    "distributed_reduction_benchmark.cpp",
)


class TestBenchmarkReleaseContract(unittest.TestCase):
    def test_manifest_covers_release_benchmark_axes(self):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        ids = {entry["id"] for entry in manifest["benchmarks"]}
        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("statevec_core_paths", ids)
        self.assertIn("distributed_reduction_rccl_vs_host", ids)
        self.assertIn("tensornet_contraction", ids)
        self.assertIn("densitymat_channel_sampling", ids)

        covered = {cover for entry in manifest["benchmarks"] for cover in entry["covers"]}
        for required in {
            "statevec_fastpath_vs_fallback",
            "gate_fusion_vs_unfused",
            "sampling_observe",
            "distributed_rccl_vs_host_fallback",
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

    def test_distributed_benchmark_covers_rccl_dense_and_generic_matrix_paths(self):
        with open(DISTRIBUTED_BENCHMARK, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("dense_expectation_ms", source)
        self.assertIn("generic_matrix_ms", source)
        self.assertIn("rocsvGetExpectationMatrix", source)
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
        self.assertIn("actions/upload-artifact@v4", combined)

    def test_readme_describes_release_benchmark_registry(self):
        with open(os.path.join(PROJECT_ROOT, "README.md"), "r", encoding="utf-8") as f:
            readme = f.read()

        self.assertIn("benchmarks/benchmark_manifest.json", readme)
        self.assertIn("benchmarks/run_release_benchmarks.py", readme)
        self.assertIn("benchmark-summary.json", readme)
        self.assertIn("dense expectation and generic matrix", readme)


if __name__ == "__main__":
    unittest.main()
