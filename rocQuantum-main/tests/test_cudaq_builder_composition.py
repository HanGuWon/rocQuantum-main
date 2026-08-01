import math

import numpy as np
import pytest

import rocq
from rocq.builder import MeasurementHandle, QuakeValue, make_kernel


def test_device_qubit_argument_call_substitutes_symbolic_classical_arguments():
    rotation, target_argument, angle_argument = make_kernel(QuakeValue, float)
    rotation.ry(angle_argument / 2.0, target_argument)
    rotation.rz(-angle_argument, target_argument)

    caller, theta = make_kernel(float)
    target = caller.qalloc()[0]
    caller.call(rotation, target, 2.0 * theta)

    context = caller.build(0.25)

    assert [(op.name, op.targets) for op in context.ops] == [
        ("ry", [0]),
        ("rz", [0]),
    ]
    assert context.ops[0].params == {"theta": pytest.approx(0.25)}
    assert context.ops[1].params == {"phi": pytest.approx(-0.5)}
    with pytest.raises(RuntimeError, match="device-style kernel"):
        rotation.build(target, 0.5)


def test_apply_call_substitutes_typed_sequence_index_expressions():
    rotation, target_argument, angles_argument = make_kernel(
        QuakeValue, list[float]
    )
    rotation.rx(angles_argument[1], target_argument)
    caller, angles = make_kernel(list[float])
    target = caller.qalloc()[0]
    caller.apply_call(rotation, target, angles)

    context = caller.build([0.1, 0.375])

    assert context.ops[0].params == {"theta": pytest.approx(0.375)}
    with pytest.raises(ValueError, match="index 1"):
        caller.build([0.1])


def test_apply_call_validates_signature_types_and_expression_ownership():
    callee, target_argument, angle_argument = make_kernel(QuakeValue, float)
    callee.rx(angle_argument, target_argument)
    caller = make_kernel()
    target = caller.qalloc()[0]

    with pytest.raises(TypeError, match="expects 2"):
        caller.apply_call(callee, target)
    with pytest.raises(TypeError, match="finite real"):
        caller.apply_call(callee, target, "not-an-angle")

    first, first_angle = make_kernel(float)
    second = make_kernel()
    second_target = second.qalloc()[0]
    with pytest.raises(ValueError, match="symbolic classical expression"):
        second.rx(first_angle, second_target)

    two_qubit, first_target, second_target_argument = make_kernel(
        QuakeValue, QuakeValue
    )
    two_qubit.cx(first_target, second_target_argument)
    with pytest.raises(ValueError, match="after kernel argument binding"):
        caller.apply_call(two_qubit, target, target)
    assert caller.build().ops == []


def test_apply_call_inlines_callee_local_allocation(monkeypatch):
    monkeypatch.setenv("ROCQ_ENABLE_MOCK_BACKENDS", "1")
    prepare_one = make_kernel()
    local = prepare_one.qalloc()[0]
    prepare_one.x(local)

    caller = make_kernel()
    caller.apply_call(prepare_one)

    assert caller.num_qubits == 1
    np.testing.assert_allclose(
        np.abs(caller.execute(backend="state_vector")),
        [0.0, 1.0],
        atol=1e-7,
    )


def test_adjoint_synthesis_reverses_and_inverts_a_composed_kernel(monkeypatch):
    monkeypatch.setenv("ROCQ_ENABLE_MOCK_BACKENDS", "1")
    transform, target_argument, angle_argument = make_kernel(QuakeValue, float)
    transform.h(target_argument)
    transform.rz(angle_argument, target_argument)
    transform.t(target_argument)
    transform.rx(angle_argument / 3.0, target_argument)

    caller, theta = make_kernel(float)
    target = caller.qalloc()[0]
    caller.apply_call(transform, target, theta)
    caller.adjoint(transform, target, theta)

    context = caller.build(0.73)
    assert [op.name for op in context.ops] == [
        "h",
        "rz",
        "t",
        "rx",
        "rx",
        "tdg",
        "rz",
        "h",
    ]
    np.testing.assert_allclose(
        caller.execute(0.73, backend="state_vector"),
        [1.0, 0.0],
        atol=2e-7,
    )


def test_control_synthesis_executes_x_and_h_device_kernels(monkeypatch):
    monkeypatch.setenv("ROCQ_ENABLE_MOCK_BACKENDS", "1")
    x_kernel, x_target = make_kernel(QuakeValue)
    x_kernel.x(x_target)
    h_kernel, h_target = make_kernel(QuakeValue)
    h_kernel.h(h_target)

    bell = make_kernel()
    bell_qubits = bell.qalloc(2)
    bell.h(bell_qubits[0])
    bell.control(x_kernel, bell_qubits[0], bell_qubits[1])
    np.testing.assert_allclose(
        np.abs(bell.execute(backend="state_vector")),
        [1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)],
        atol=2e-7,
    )

    controlled_h = make_kernel()
    controlled_h_qubits = controlled_h.qalloc(2)
    controlled_h.x(controlled_h_qubits[0])
    controlled_h.control(
        h_kernel,
        controlled_h_qubits[0],
        controlled_h_qubits[1],
    )
    np.testing.assert_allclose(
        controlled_h.execute(backend="state_vector"),
        [0.0, 1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)],
        atol=2e-7,
    )


def test_multi_control_synthesis_and_unsupported_phase_are_fail_closed(monkeypatch):
    monkeypatch.setenv("ROCQ_ENABLE_MOCK_BACKENDS", "1")
    z_kernel, z_target = make_kernel(QuakeValue)
    z_kernel.z(z_target)

    caller = make_kernel()
    qubits = caller.qalloc(3)
    caller.x(qubits)
    caller.control(z_kernel, qubits[:2], qubits[2])
    state = caller.execute(backend="state_vector")
    np.testing.assert_allclose(state[:-1], np.zeros(7), atol=1e-7)
    assert state[-1] == pytest.approx(-1.0)

    phase_kernel, phase_target = make_kernel(QuakeValue)
    phase_kernel.s(phase_target)
    rejected = make_kernel()
    rejected_qubits = rejected.qalloc(3)
    with pytest.raises(NotImplementedError, match="Multi-controlled phase"):
        rejected.control(phase_kernel, rejected_qubits[:2], rejected_qubits[2])
    assert rejected.build().ops == []
    assert rejected.num_qubits == 3


def test_terminal_measurement_selects_sample_qubits_and_blocks_dynamic_use(monkeypatch):
    monkeypatch.setenv("ROCQ_ENABLE_MOCK_BACKENDS", "1")
    builder = make_kernel()
    qubits = builder.qalloc(2)
    builder.x(qubits[1])
    handle = builder.mz(qubits[1], register_name="result")

    assert isinstance(handle, MeasurementHandle)
    assert isinstance(handle, rocq.MeasurementHandle)
    assert handle.qubits == (1,)
    assert handle.basis == "z"
    assert handle.register_name == "result"
    with pytest.raises(TypeError, match="mid-circuit classical control"):
        bool(handle)
    assert builder.sample(shots_count=8, backend="state_vector") == {"1": 8}
    with pytest.raises(ValueError, match="cannot override"):
        builder.sample(shots_count=1, backend="state_vector", qubits=[0])
    with pytest.raises(RuntimeError, match="after a terminal measurement"):
        builder.h(qubits[0])
    with pytest.raises(RuntimeError, match="after a terminal measurement"):
        builder.qalloc()
    drawing = builder.draw()
    assert "mz q[1] -> result" in drawing
    with pytest.raises(NotImplementedError, match="measurement collapse"):
        builder.execute(backend="state_vector")
    mlir = builder.mlir()
    assert '"quantum.mz"(%q1)' in mlir
    assert 'registerName = "result"' in mlir
    assert "!quantum.result" in mlir
    with pytest.raises(NotImplementedError, match="target IR"):
        builder.translate("openqasm2")


def test_terminal_x_and_y_measurements_lower_basis_changes_before_sampling():
    builder = make_kernel()
    qubits = builder.qalloc(3)
    z_result = builder.mz(qubits[0], "z")
    x_result = builder.mx(qubits[1], "x")
    y_result = builder.my(qubits[2], "y")

    assert [handle.basis for handle in (z_result, x_result, y_result)] == [
        "z",
        "x",
        "y",
    ]
    assert [op.name for op in builder.build().ops] == ["h", "sdg", "h"]
    assert [
        qubit
        for handle in builder.measurement_handles
        for qubit in handle.qubits
    ] == [0, 1, 2]
    with pytest.raises(ValueError, match="terminal measurement twice"):
        builder.mz(qubits[0])
    with pytest.raises(ValueError, match="already used"):
        builder.mz([], "x")


def test_measurement_mlir_expands_and_escapes_unique_result_labels():
    builder = make_kernel()
    qubits = builder.qalloc(3)
    builder.mz(qubits[:2], 'readout"')
    builder.mz(qubits[2])

    mlir = builder.mlir()
    assert 'registerName = "readout\\\"[0]"' in mlir
    assert 'registerName = "readout\\\"[1]"' in mlir
    assert 'registerName = "result[0]"' in mlir
    assert mlir.count('"quantum.mz"') == 3

    rejected = make_kernel()
    rejected_qubit = rejected.qalloc()[0]
    with pytest.raises(ValueError, match="NUL"):
        rejected.mz(rejected_qubit, "bad\x00label")


def test_measurement_bearing_device_kernel_is_not_silently_composed():
    measured = make_kernel()
    measured_qubit = measured.qalloc()[0]
    measured.mz(measured_qubit)
    caller = make_kernel()

    with pytest.raises(NotImplementedError, match="measurement-aware classical IR"):
        caller.apply_call(measured)


def test_device_kernel_exp_pauli_keeps_distinct_formal_qubits():
    device, first, second = make_kernel(QuakeValue, QuakeValue)
    device.exp_pauli(0.25, [first, second], "XX")

    caller = make_kernel()
    qubits = caller.qalloc(2)
    caller.apply_call(device, qubits[0], qubits[1])

    assert [operation.name for operation in caller.build().ops] == [
        "h",
        "h",
        "cnot",
        "rz",
        "cnot",
        "h",
        "h",
    ]
