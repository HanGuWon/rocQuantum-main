import math
import os
from unittest import mock

import numpy as np
import pytest

import rocq
from rocq.dynamics import IntermediateResultSave, Schedule, evolve, evolve_async
from rocq.gates import exp_pauli
from rocq.operator import HermitianOperator, PauliOperator, operator_to_matrix


def test_operator_to_matrix_uses_runtime_little_endian_qubit_order():
    x0 = operator_to_matrix(PauliOperator("X0"), num_qubits=2)
    x1 = operator_to_matrix(PauliOperator("X1"), num_qubits=2)

    np.testing.assert_allclose(x0 @ np.array([1, 0, 0, 0]), [0, 1, 0, 0])
    np.testing.assert_allclose(x1 @ np.array([1, 0, 0, 0]), [0, 0, 1, 0])


def test_operator_to_matrix_supports_mixed_sum_and_local_embedding():
    local_z = HermitianOperator([[1, 0], [0, -1]], targets=[1])
    matrix = operator_to_matrix(0.5 * PauliOperator("X0") + 2.0 * local_z, 2)
    expected = 0.5 * np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    expected += 2.0 * np.kron(np.array([[1, 0], [0, -1]]), np.eye(2))
    np.testing.assert_allclose(matrix, expected)


def test_operator_to_matrix_rejects_register_that_is_too_small():
    with pytest.raises(ValueError, match="too small"):
        operator_to_matrix(PauliOperator("Z2"), num_qubits=2)


def test_exp_pauli_matches_cudaq_positive_exponent_convention():
    @rocq.kernel
    def rotate():
        q = rocq.qvec(2)
        exp_pauli(math.pi / 2.0, q, "XI")

    with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
        state = rocq.get_state(rotate, backend="state_vector")
    np.testing.assert_allclose(state, [0.0, 1.0j, 0.0, 0.0], atol=2e-6)


def test_exp_pauli_variadic_form_respects_target_order():
    @rocq.kernel
    def rotate():
        q = rocq.qvec(3)
        exp_pauli(math.pi / 2.0, "XZ", q[0], q[2])

    with mock.patch.dict(os.environ, {"ROCQ_ENABLE_MOCK_BACKENDS": "1"}):
        counts = rocq.sample(rotate, 8, backend="state_vector")
    assert counts == {"001": 8}


@pytest.mark.parametrize(
    "word,targets,error",
    [
        ("XA", [0, 1], "only I, X, Y, and Z"),
        ("X", [0, 1], "length must match"),
        ("XX", [0, 0], "distinct"),
    ],
)
def test_exp_pauli_fails_closed_on_invalid_words_and_targets(word, targets, error):
    @rocq.kernel
    def invalid():
        q = rocq.qvec(2)
        exp_pauli(0.2, word, *(q[index] for index in targets))

    with pytest.raises((ValueError, RuntimeError), match=error):
        invalid.build()


def test_schedule_is_immutable_from_caller_and_requires_increasing_times():
    source = [0.0, 0.5, 1.0]
    schedule = Schedule(source, parameter_names=["t"])
    source[1] = 99.0
    np.testing.assert_allclose(schedule.steps, [0.0, 0.5, 1.0])
    assert schedule.parameter_names == ("t",)

    with pytest.raises(ValueError, match="strictly increasing"):
        Schedule([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        Schedule([0.0, float("nan")])


def test_closed_system_evolution_matches_one_qubit_rabi_oracle():
    hamiltonian = 0.5 * PauliOperator("X0")
    result = evolve(
        hamiltonian,
        Schedule([0.0, math.pi / 2.0, math.pi]),
        observables=[PauliOperator("Z0")],
        store_intermediate_results=IntermediateResultSave.ALL,
    )

    np.testing.assert_allclose(result.final_state, [0.0, -1.0j], atol=1e-10)
    np.testing.assert_allclose(result.expectation(0).real, [1.0, 0.0, -1.0], atol=1e-10)
    assert result.state_kind == "state_vector"
    assert len(result.intermediate_states) == 3


def test_time_dependent_hamiltonian_is_evaluated_on_the_schedule():
    calls = []

    def hamiltonian(time):
        calls.append(time)
        return 0.5 * PauliOperator("X0")

    result = evolve(
        hamiltonian,
        [0.0, math.pi],
        num_qubits=1,
        max_step=0.2,
    )
    np.testing.assert_allclose(result.final_state, [0.0, -1.0j], atol=1e-10)
    assert calls
    assert all(0.0 <= time <= math.pi for time in calls)


def test_density_matrix_unitary_evolution_preserves_density_contract():
    initial = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    result = evolve(
        0.5 * PauliOperator("X0"),
        [0.0, math.pi],
        initial_state=initial,
    )
    np.testing.assert_allclose(result.final_state, [[0.0, 0.0], [0.0, 1.0]], atol=1e-10)
    assert result.state_kind == "density_matrix"


def test_lindblad_amplitude_damping_matches_analytic_population():
    gamma = 0.7
    lowering = math.sqrt(gamma) * np.array([[0.0, 1.0], [0.0, 0.0]])
    initial_excited = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    result = evolve(
        np.zeros((2, 2), dtype=np.complex128),
        [0.0, 1.0],
        initial_state=initial_excited,
        collapse_operators=[lowering],
        observables=[PauliOperator("Z0")],
        max_step=0.01,
    )

    expected_excited = math.exp(-gamma)
    assert result.final_state[1, 1].real == pytest.approx(expected_excited, abs=2e-9)
    assert np.trace(result.final_state) == pytest.approx(1.0, abs=1e-12)
    assert result.expectation(0)[-1].real == pytest.approx(1.0 - 2.0 * expected_excited, abs=2e-9)


def test_evolve_validates_state_hamiltonian_and_options():
    with pytest.raises(ValueError, match="unit norm"):
        evolve(PauliOperator("Z0"), [0.0, 1.0], initial_state=[1.0, 1.0])
    with pytest.raises(ValueError, match="Hermitian"):
        evolve([[0.0, 1.0], [0.0, 0.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="positive and finite"):
        evolve(PauliOperator("Z0"), [0.0, 1.0], max_step=0.0)
    with pytest.raises(ValueError, match="store_intermediate_results"):
        evolve(PauliOperator("Z0"), [0.0, 1.0], store_intermediate_results="everything")


def test_evolve_default_none_keeps_only_final_results():
    result = evolve(
        0.5 * PauliOperator("X0"),
        [0.0, math.pi / 2.0, math.pi],
        observables=[PauliOperator("Z0")],
    )

    np.testing.assert_allclose(result.times, [math.pi])
    np.testing.assert_allclose(result.expectation(0).real, [-1.0], atol=1e-10)
    assert len(result.intermediate_states) == 1
    np.testing.assert_allclose(result.intermediate_states[0], result.final_state)


@pytest.mark.parametrize(
    "store_mode",
    [
        IntermediateResultSave.EXPECTATION_VALUE,
        IntermediateResultSave.EXPECTATION_VALUES,
        "expectation_value",
        "expectation_values",
    ],
)
def test_evolve_expectation_value_modes_keep_history_but_only_final_state(store_mode):
    result = evolve(
        PauliOperator("Z0"),
        [0.0, 0.5, 1.0],
        store_intermediate_results=store_mode,
    )

    assert len(result.intermediate_states) == 1
    assert result.expectation_values.shape == (3, 0)


def test_evolve_async_returns_canonical_async_result():
    result = evolve_async(
        0.5 * PauliOperator("X0"),
        [0.0, math.pi],
        observables=[PauliOperator("Z0")],
        store_intermediate_results=IntermediateResultSave.EXPECTATION_VALUE,
        max_step=0.1,
    ).get(timeout=5.0)

    assert result.expectation()[0].real == pytest.approx(1.0)
    assert result.expectation()[-1].real == pytest.approx(-1.0, abs=1e-10)

    with pytest.raises(ValueError, match="qpu_id must be 0"):
        evolve_async(PauliOperator("Z0"), [0.0, 1.0], qpu_id=1)
