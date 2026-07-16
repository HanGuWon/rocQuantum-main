import math

import numpy as np
import pytest

from rocq.builder import KernelBuilder, KernelExpression, make_kernel


def test_make_kernel_scalar_expression_specializes_and_executes(monkeypatch):
    monkeypatch.setenv("ROCQ_ENABLE_MOCK_BACKENDS", "1")
    builder, theta = make_kernel(float)
    qubits = builder.qalloc(1)
    builder.ry(theta / 2.0 + theta / 2.0, qubits[0])

    state = builder.execute(math.pi, backend="state_vector")

    assert np.allclose(np.abs(state), [0.0, 1.0], atol=1e-12)
    context = builder.build(math.pi / 3.0)
    assert context.ops[0].params["theta"] == pytest.approx(math.pi / 3.0)


def test_make_kernel_sequence_argument_indexing_and_mlir_specialization():
    builder, angles = make_kernel(list[float])
    qubits = builder.qalloc(2)
    builder.h(qubits)
    builder.crz(-2.0 * angles[1], qubits[0], qubits[1])

    mlir = builder.mlir([0.1, 0.25])

    assert mlir.count('"quantum.h"') == 2
    assert '"quantum.crz"' in mlir
    assert "angle = -0.5 : f64" in mlir


def test_make_kernel_multiple_typed_arguments_are_strict():
    builder, integer, flag = make_kernel(int, bool)
    builder.qalloc(1)

    assert isinstance(builder, KernelBuilder)
    assert isinstance(integer, KernelExpression)
    assert isinstance(flag, KernelExpression)
    builder.build(3, True)
    with pytest.raises(TypeError, match="expects 2"):
        builder.build(3)
    with pytest.raises(TypeError, match="argument 0 must be int"):
        builder.build(True, False)
    with pytest.raises(TypeError, match="argument 1 must be bool"):
        builder.build(3, 1)


def test_make_kernel_rejects_unsupported_types_and_invalid_allocations():
    with pytest.raises(TypeError, match="argument types"):
        make_kernel(str)
    builder = make_kernel()
    with pytest.raises(ValueError, match="positive integer"):
        builder.qalloc(0)
    with pytest.raises(ValueError, match="positive integer"):
        builder.qalloc(True)


def test_builder_qubits_cannot_cross_kernel_ownership():
    first = make_kernel()
    second = make_kernel()
    first_qubit = first.qalloc()[0]
    second_qubit = second.qalloc()[0]

    with pytest.raises(ValueError, match="cannot be shared"):
        first.cx(first_qubit, second_qubit)


def test_builder_expression_reports_index_and_arithmetic_errors():
    builder, values = make_kernel(list[float])
    qubit = builder.qalloc()[0]
    builder.rx(values[2], qubit)
    with pytest.raises(ValueError, match="index 2"):
        builder.build([0.5])

    division, value = make_kernel(float)
    division_qubit = division.qalloc()[0]
    division.rz(value / 0.0, division_qubit)
    with pytest.raises(ValueError, match="truediv"):
        division.build(0.5)


def test_builder_exp_pauli_uses_cudaq_positive_exponent(monkeypatch):
    monkeypatch.setenv("ROCQ_ENABLE_MOCK_BACKENDS", "1")
    builder, angle = make_kernel(float)
    qubit = builder.qalloc(1)
    builder.exp_pauli(angle, qubit, "X")

    state = builder.execute(math.pi / 2.0, backend="state_vector")

    assert np.allclose(np.abs(state), [0.0, 1.0], atol=1e-12)


def test_builder_reuses_canonical_tools():
    builder = make_kernel()
    qubits = builder.qalloc(2)
    builder.h(qubits[0])
    builder.cx(qubits[0], qubits[1])

    resources = builder.estimate_resources()
    drawing = builder.draw()

    assert resources.num_qubits == 2
    assert resources.num_gates == 2
    assert "h" in drawing.lower()
    assert "cnot" in drawing.lower()


def test_default_qalloc_register_is_single_target_compatible(monkeypatch):
    monkeypatch.setenv("ROCQ_ENABLE_MOCK_BACKENDS", "1")
    builder = make_kernel()
    control = builder.qalloc()
    target = builder.qalloc()

    builder.h(control)
    builder.cx(control, target)

    state = builder.execute(backend="state_vector")
    np.testing.assert_allclose(
        np.abs(state),
        [1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)],
        atol=1e-12,
    )

    register = builder.qalloc(2)
    with pytest.raises(ValueError, match="multi-qubit register"):
        builder.cx(register, target)
