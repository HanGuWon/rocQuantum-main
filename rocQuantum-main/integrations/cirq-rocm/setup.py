# cirq-rocm/setup.py
from pathlib import Path
import sys

from setuptools import setup, find_packages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _compat_setup import compatibility_long_description, root_project_version

setup(
    name="cirq-rocm",
    version=root_project_version(__file__),
    author="rocQuantum Developers",
    description="Compatibility installer for the rocQuantum Cirq adapter.",
    long_description=compatibility_long_description("cirq-rocm"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "cirq-core>=1.0",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
