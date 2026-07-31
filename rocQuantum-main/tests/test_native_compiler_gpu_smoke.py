"""Host-side numerical contracts for the fail-closed compiler GPU smoke."""

from __future__ import annotations

import math

import pytest

from scripts.native_compiler_gpu_smoke import validate_bell_state


def test_validate_bell_state_accepts_exact_state():
    amplitude = 1 / math.sqrt(2)
    metrics = validate_bell_state([amplitude, 0j, 0j, amplitude])

    assert metrics["max_amplitude_error"] == 0.0
    assert metrics["state_norm"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ([1.0, 0j], "must contain 4 amplitudes"),
        ([complex(float("nan"), 0.0), 0j, 0j, 0j], "non-finite"),
        ([1.0, 0j, 0j, 0j], "Bell state mismatch"),
    ],
)
def test_validate_bell_state_rejects_non_evidence(state, message):
    with pytest.raises(AssertionError, match=message):
        validate_bell_state(state)
