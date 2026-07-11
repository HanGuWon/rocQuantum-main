from __future__ import annotations

from numbers import Integral

import numpy as np

from .rocq_device import RocQDevice


LIGHTNING_COMPAT_OPTION_DEFAULTS = {
    "batch_obs": False,
    "mpi": False,
    "mcmc": False,
    "kernel_name": None,
    "num_burnin": 0,
    "c_dtype": np.complex128,
    "seed": "global",
}


def _normalize_bool_option(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _normalize_nonnegative_integer_option(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative.")
    return normalized


def _normalize_complex_dtype_option(value):
    try:
        return np.dtype(value).type
    except (TypeError, ValueError) as exc:
        raise ValueError("c_dtype must be a valid NumPy dtype.") from exc


def _normalize_lightning_options(options):
    unknown_options = sorted(
        str(name) for name in options if name not in LIGHTNING_COMPAT_OPTION_DEFAULTS
    )
    if unknown_options:
        option_names = ", ".join(unknown_options)
        raise ValueError(f"Unsupported rocQuantum PennyLane device option(s): {option_names}")

    normalized = dict(LIGHTNING_COMPAT_OPTION_DEFAULTS)
    normalized.update(options)
    for name in ("batch_obs", "mpi", "mcmc"):
        normalized[name] = _normalize_bool_option(name, normalized[name])
    normalized["num_burnin"] = _normalize_nonnegative_integer_option(
        "num_burnin",
        normalized["num_burnin"],
    )
    normalized["c_dtype"] = _normalize_complex_dtype_option(normalized["c_dtype"])
    if normalized["kernel_name"] is not None and not isinstance(normalized["kernel_name"], str):
        raise ValueError("kernel_name must be a string or None.")

    unsupported_non_defaults = []
    for name, default in LIGHTNING_COMPAT_OPTION_DEFAULTS.items():
        if name == "seed":
            matches_default = isinstance(normalized[name], str) and normalized[name] == default
        else:
            matches_default = normalized[name] == default
        if not matches_default:
            unsupported_non_defaults.append(name)
    if unsupported_non_defaults:
        option_names = ", ".join(unsupported_non_defaults)
        raise NotImplementedError(
            "rocQuantum's PennyLane Lightning-compatible aliases do not yet support "
            f"non-default Lightning option(s): {option_names}."
        )
    return normalized


class RocqDevice(RocQDevice):
    """Compatibility alias for the historical rocq.pennylane entry point."""

    short_name = "rocq.pennylane"


class LightningRocqDevice(RocQDevice):
    """PennyLane Lightning-style AMD GPU entry point backed by rocQuantum."""

    name = "rocQuantum Lightning-compatible AMD GPU Device"
    short_name = "lightning.rocq"

    @classmethod
    def capabilities(cls):
        capabilities = dict(super().capabilities())
        capabilities.update(
            {
                "lightning_compatible_aliases": ("lightning.rocq", "lightning.rocm"),
                "unsupported_non_default_lightning_options": tuple(
                    LIGHTNING_COMPAT_OPTION_DEFAULTS.keys()
                ),
            }
        )
        return capabilities

    def __init__(self, wires, shots=None, **kwargs):
        self._lightning_compat_options = _normalize_lightning_options(kwargs)
        super().__init__(wires=wires, shots=shots)


class LightningRocmDevice(LightningRocqDevice):
    """Alias matching the ROCm platform name for Lightning-style discovery."""

    short_name = "lightning.rocm"
