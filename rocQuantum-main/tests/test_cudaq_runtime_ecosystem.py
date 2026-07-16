from concurrent.futures import Future, ThreadPoolExecutor
import importlib
from unittest import mock

import numpy as np
import pytest

import rocq

kernel_module = importlib.import_module("rocq.kernel")


class _FakeBackend:
    def __init__(self):
        self.ops = []

    def run_ops(self, ops, noise_model=None):
        self.ops = list(ops)

    def get_state(self):
        return "state"

    def sample(self, shots, qubits=None):
        return {"0": shots}

    def expectation(self, operator):
        return 0.75


@pytest.fixture(autouse=True)
def _restore_default_target():
    rocq.reset_target()
    yield
    rocq.reset_target()


@rocq.kernel
def _one_qubit_kernel():
    q = rocq.qvec(1)
    rocq.h(q[0])


def test_target_registry_context_and_explicit_backend_compatibility():
    assert [target.name for target in rocq.get_targets()] == [
        "state_vector",
        "density_matrix",
        "stabilizer",
        "tableau",
        "clifford",
    ]
    assert rocq.has_target("density_matrix")
    assert not rocq.has_target("missing")
    assert rocq.get_target().name == "state_vector"
    assert rocq.num_qpus() == 1
    assert rocq.get_target().num_qpus() == 1

    with rocq.target("density_matrix") as selected:
        assert selected is rocq.get_target()
        assert selected.backend == "density_matrix"
        with rocq.target("stabilizer"):
            assert rocq.get_target().name == "stabilizer"
        assert rocq.get_target().name == "density_matrix"
    assert rocq.get_target().name == "state_vector"

    fake_backend = _FakeBackend()
    rocq.set_target("density_matrix")
    with mock.patch.object(kernel_module, "get_backend", return_value=fake_backend) as factory:
        assert rocq.execute(_one_qubit_kernel) == "state"
        factory.assert_called_once_with("density_matrix", 1, enable_fusion=None)

    with mock.patch.object(kernel_module, "get_backend", return_value=fake_backend) as factory:
        rocq.execute(_one_qubit_kernel, backend="stabilizer")
        factory.assert_called_once_with("stabilizer", 1, enable_fusion=None)


@pytest.mark.parametrize("invalid", ["", "unknown", None, True])
def test_target_lookup_fails_closed(invalid):
    if invalid is None:
        assert rocq.get_target() == rocq.get_target("state_vector")
        return
    with pytest.raises(ValueError):
        rocq.set_target(invalid)


def test_async_result_preserves_submission_context_and_adds_get():
    fake_backend = _FakeBackend()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with rocq.target("density_matrix"):
            with mock.patch.object(
                kernel_module, "get_backend", return_value=fake_backend
            ) as factory:
                result = rocq.sample_async(
                    _one_qubit_kernel,
                    5,
                    executor=executor,
                )
                assert isinstance(result, Future)
                assert isinstance(result, rocq.AsyncResult)
                assert result.get(timeout=5) == {"0": 5}
                factory.assert_called_once_with(
                    "density_matrix", 1, enable_fusion=None
                )

    assert rocq.get_target().name == "state_vector"


def test_result_types_preserve_builtin_compatibility_and_validate_values():
    samples = rocq.SampleResult({"00": 3, "11": 2})
    assert isinstance(samples, dict)
    assert samples == {"00": 3, "11": 2}
    assert samples.total_shots == 5
    assert samples.most_probable() == "00"
    assert samples.probability("11") == pytest.approx(0.4)
    assert rocq.SampleResult({}).probability("0") == 0.0

    observed = rocq.ObserveResult(0.25 + 0.0j)
    assert isinstance(observed, float)
    assert observed == 0.25
    assert observed.expectation() == 0.25
    assert rocq.ObserveResult(np.complex64(0.5 + 0.0j)) == 0.5

    with pytest.raises(TypeError):
        rocq.SampleResult({"0": True})
    with pytest.raises(ValueError):
        rocq.ObserveResult(0.25 + 0.1j)


def test_sample_and_observe_return_compatibility_result_types():
    fake_backend = _FakeBackend()
    with mock.patch.object(kernel_module, "get_backend", return_value=fake_backend):
        samples = rocq.sample(_one_qubit_kernel, 4)
        observed = rocq.observe(_one_qubit_kernel, object())

    assert isinstance(samples, rocq.SampleResult)
    assert isinstance(samples, dict)
    assert isinstance(observed, rocq.ObserveResult)
    assert isinstance(observed, float)


@rocq.kernel
def _parameterized_kernel(theta):
    q = rocq.qvec(1)
    rocq.ry(theta, q[0])


def test_sample_supports_cudaq_shots_count_call_shape_and_default():
    fake_backend = _FakeBackend()
    with mock.patch.object(kernel_module, "get_backend", return_value=fake_backend):
        parameterized = rocq.sample(
            _parameterized_kernel,
            0.25,
            shots_count=7,
        )
        defaulted = rocq.sample(_one_qubit_kernel)

    assert parameterized == {"0": 7}
    assert defaulted == {"0": 1000}


@pytest.mark.parametrize("entry_point", [
    rocq.execute_async,
    rocq.get_state_async,
    rocq.sample_async,
    rocq.observe_async,
    rocq.compile_and_execute_async,
])
def test_async_entry_points_fail_closed_for_unavailable_qpu_ids(entry_point):
    positional = []
    if entry_point is rocq.sample_async:
        positional = [1]
    elif entry_point is rocq.observe_async:
        positional = [object()]
    with pytest.raises(ValueError, match="qpu_id must be 0"):
        entry_point(_one_qubit_kernel, *positional, qpu_id=1)


@rocq.kernel
def _resource_kernel():
    q = rocq.qvec(3)
    rocq.h(q[0])
    rocq.x(q[1])
    rocq.cnot(q[0], q[2])


def test_tools_estimate_draw_and_translate_supported_subset():
    resources = rocq.estimate_resources(_resource_kernel)
    assert resources.num_qubits == 3
    assert resources.num_gates == 3
    assert resources.depth == 2
    assert dict(resources.gate_counts) == {"cnot": 1, "h": 1, "x": 1}
    assert resources.count("H") == 1
    assert resources == _resource_kernel.estimate_resources()

    drawing = _resource_kernel.draw()
    assert "kernel _resource_kernel (3 qubits)" in drawing
    assert "h q[0]" in drawing
    assert "cnot q[0], q[2]" in drawing
    assert drawing == rocq.draw(_resource_kernel)

    mlir = rocq.translate(_resource_kernel, "mlir")
    assert '"quantum.h"' in mlir
    qasm = _resource_kernel.translate("openqasm2")
    assert qasm.startswith("OPENQASM 2.0;")
    assert "qreg q[3];" in qasm
    assert "cx q[0],q[2];" in qasm


def test_translate_rejects_unknown_formats_and_unsupported_qasm_ops():
    with pytest.raises(ValueError, match="Unsupported translation format"):
        rocq.translate(_one_qubit_kernel, "json")

    @rocq.kernel
    def unsupported_qasm():
        q = rocq.qvec(2)
        rocq.crx(0.5, q[0], q[1])

    with pytest.raises(NotImplementedError, match="does not support gate 'crx'"):
        rocq.translate(unsupported_qasm, "openqasm2")


@pytest.mark.parametrize(
    ("channel", "backend_type"),
    [
        (rocq.BitFlipChannel(0.1), "bit_flip"),
        (rocq.PhaseFlipChannel(0.1), "phase_flip"),
        (rocq.DepolarizationChannel(0.1), "depolarizing"),
        (rocq.AmplitudeDampingChannel(0.1), "amplitude_damping"),
    ],
)
def test_cudaq_style_channel_objects_normalize_to_existing_backend_specs(
    channel, backend_type
):
    model = rocq.NoiseModel()
    model.add_channel("x", [0], channel)
    spec = model.get_channels()[0]
    assert spec == {
        "type": backend_type,
        "prob": 0.1,
        "qubits": [0],
        "op": "x",
        "kraus_matrices": None,
    }


def test_kraus_and_phase_damping_channels_are_cptp_and_backend_compatible():
    phase_damping = rocq.PhaseDampingChannel(0.2)
    matrices = phase_damping.kraus_matrices
    completeness = sum(matrix.conj().T @ matrix for matrix in matrices)
    np.testing.assert_allclose(completeness, np.eye(2))

    model = rocq.NoiseModel()
    model.add_all_qubit_channel("h", phase_damping)
    spec = model.get_channels()[0]
    assert spec["type"] == "kraus"
    assert spec["prob"] == 1.0
    assert spec["qubits"] is None
    assert spec["op"] == "h"
    np.testing.assert_allclose(spec["kraus_matrices"], matrices)

    identity_channel = rocq.KrausChannel([np.eye(2)])
    object_style = rocq.NoiseModel()
    object_style.add_channel(identity_channel, on_qubits=0, after_op="x")
    assert object_style.get_channels()[0]["qubits"] == [0]

    with pytest.raises(ValueError, match="requires exactly 1 target"):
        object_style.add_channel("x", [0, 1], identity_channel)


@pytest.mark.parametrize(
    "operators",
    [
        [],
        [np.zeros((2, 3))],
        [np.eye(3)],
        [np.zeros((2, 2))],
        [np.array([[np.nan, 0], [0, 1]])],
    ],
)
def test_kraus_channel_validation_rejects_invalid_operators(operators):
    with pytest.raises((TypeError, ValueError)):
        rocq.KrausChannel(operators)


def test_legacy_noise_model_api_remains_unchanged():
    model = rocq.NoiseModel()
    model.add_channel("bit_flip", 0.25, on_qubits=[1], after_op="X")
    assert model.get_channels()[0] == {
        "type": "bit_flip",
        "prob": 0.25,
        "qubits": [1],
        "op": "x",
        "kraus_matrices": None,
    }
