"""Keep shipped examples aligned with the canonical public Python API."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILES = [ROOT / "example.py"] + sorted(
    path
    for path in (ROOT / "examples").rglob("*.py")
    if not path.name.startswith("_")
)
FORBIDDEN_LEGACY_REFERENCES = (
    "rocq.Simulator",
    "rocq.api",
    "rocq.grad",
    "rocq_hip",
    "DensityMatrixState",
)


def test_examples_do_not_reference_removed_python_surfaces():
    offenders = {}
    for path in EXAMPLE_FILES:
        source = path.read_text(encoding="utf-8")
        matches = [name for name in FORBIDDEN_LEGACY_REFERENCES if name in source]
        if matches:
            offenders[str(path.relative_to(ROOT))] = matches

    assert offenders == {}


@pytest.mark.parametrize(
    "example_path",
    EXAMPLE_FILES,
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_example_runs_with_cpu_mock_fallback(example_path):
    env = os.environ.copy()
    env["ROCQ_ENABLE_MOCK_BACKENDS"] = "1"
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), current_pythonpath) if value
    )

    result = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        f"{example_path.relative_to(ROOT)} failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
