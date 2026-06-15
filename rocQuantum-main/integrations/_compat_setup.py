"""Shared helpers for legacy integration setup.py compatibility installers."""

from __future__ import annotations

from pathlib import Path
import re

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10 fallback.
    tomllib = None  # type: ignore[assignment]


def root_project_version(setup_file: str) -> str:
    """Return the root pyproject.toml version for an integration setup.py file."""
    pyproject = Path(setup_file).resolve().parents[2] / "pyproject.toml"
    if tomllib is not None:
        with pyproject.open("rb") as f:
            return str(tomllib.load(f)["project"]["version"])

    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if not match:
        raise RuntimeError("Could not read rocQuantum version from pyproject.toml.")
    return match.group(1)


def compatibility_long_description(adapter_name: str) -> str:
    return (
        f"{adapter_name} is shipped as part of the root rocquantum package. "
        "This setup.py is kept only as a compatibility installer for adapter-only "
        "development; the supported project install path is `pip install .` from "
        "the repository root, optionally with extras such as `rocquantum[all]`."
    )
