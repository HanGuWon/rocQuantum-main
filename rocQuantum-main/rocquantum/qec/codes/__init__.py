"""Built-in QEC code models and registry extension points."""

from .base import (
    Code,
    CodeMetadata,
    code,
    get_available_codes,
    get_code,
    register_code,
)
from .repetition_code import RepetitionCode, ThreeQubitRepetitionCode
from .steane import SteaneCode

__all__ = [
    "Code",
    "CodeMetadata",
    "RepetitionCode",
    "SteaneCode",
    "ThreeQubitRepetitionCode",
    "code",
    "get_available_codes",
    "get_code",
    "register_code",
]
