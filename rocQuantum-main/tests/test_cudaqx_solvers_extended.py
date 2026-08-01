"""CPU exact-oracle tests for the extended CUDA-QX-style solver layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from types import SimpleNamespace

import numpy as np
import pytest


os.environ.setdefault("ROCQ_ENABLE_MOCK_BACKENDS", "1")

import rocq
from rocq.operator import PauliOperator, operator_to_matrix
from rocquantum.solvers import (
    MolecularHamiltonian,
    ObserveExecutionType,
    ObserveIteration,
    adapt_vqe,
    create_molecule,
    get_num_qaoa_parameters,
    get_operator_pool,
    jordan_wigner,
    qaoa,
    vqe,
)
from rocquantum.solvers.qaoa import _make_generic_qaoa_kernel
from rocquantum.solvers.vqe_solver import VQE_Solver, _normalize_optimizer


class HoldOptimizer:
    def minimize(self, fun, x0, args=()):
        parameters = np.asarray(x0, dtype=float)
        return SimpleNamespace(fun=fun(parameters, *args), x=parameters.copy())


class OneParameterGridOptimizer:
    def minimize(self, fun, x0, args=()):
        parameters = np.asarray(x0, dtype=float)
        if parameters.size != 1:
            raise AssertionError("test optimizer only supports one parameter")
        candidates = np.linspace(-np.pi / 2.0, np.pi / 2.0, 161)
        energies = [fun(np.asarray([value]), *args) for value in candidates]
        index = int(np.argmin(energies))
        return SimpleNamespace(
            fun=float(energies[index]),
            x=np.asarray([candidates[index]], dtype=float),
        )


@rocq.kernel
def single_rx(theta):
    q = rocq.qvec(1)
    rocq.rx(theta, q[0])


def test_functional_vqe_returns_immutable_trace_and_fails_closed_for_shots():
    energy, parameters, trace = vqe(
        single_rx,
        PauliOperator("Z0"),
        [0.25],
        optimizer=HoldOptimizer(),
    )

    assert energy == pytest.approx(np.cos(0.25), abs=1.0e-7)
    np.testing.assert_allclose(parameters, [0.25])
    assert len(trace) == 1
    assert trace[0] == ObserveIteration(
        (0.25,),
        energy,
        ObserveExecutionType.function,
    )
    with pytest.raises(FrozenInstanceError):
        trace[0].result = 0.0
    with pytest.raises(NotImplementedError, match="shots-based VQE"):
        vqe(
            single_rx,
            PauliOperator("Z0"),
            [0.25],
            optimizer=HoldOptimizer(),
            shots=100,
        )


def test_functional_vqe_optimizer_and_numeric_validation_is_strict():
    assert _normalize_optimizer("cobyla").options["method"] == "COBYLA"
    assert _normalize_optimizer("lbfgs").options["method"] == "L-BFGS-B"
    with pytest.raises(ValueError, match="optimizer must be"):
        _normalize_optimizer("adam")
    with pytest.raises(ValueError, match="max_iterations"):
        _normalize_optimizer(HoldOptimizer(), max_iterations=5)
    with pytest.raises(ValueError, match="finite numeric"):
        vqe(single_rx, PauliOperator("Z0"), [True], optimizer=HoldOptimizer())
    with pytest.raises(ValueError, match="finite"):
        vqe(single_rx, PauliOperator("Z0"), [np.nan], optimizer=HoldOptimizer())

    scipy_minimize = pytest.importorskip("scipy.optimize").minimize
    callable_optimizer = _normalize_optimizer(
        scipy_minimize,
        optimizer_options={"method": "COBYLA", "options": {"maxiter": 20}},
    )
    callable_result = callable_optimizer.minimize(
        lambda values: float((values[0] - 0.5) ** 2),
        np.asarray([0.0]),
    )
    assert np.isfinite(callable_result.fun)


def test_functional_vqe_lbfgs_records_gradient_expectation_probes():
    energy, parameters, trace = vqe(
        single_rx,
        PauliOperator("Z0"),
        [0.2],
        optimizer="lbfgs",
        gradient="parameter_shift",
        max_iterations=20,
    )

    assert energy < -0.999999
    assert parameters[0] == pytest.approx(np.pi, abs=1.0e-5)
    assert ObserveExecutionType.function in {entry.type for entry in trace}
    assert ObserveExecutionType.gradient in {entry.type for entry in trace}


def test_functional_vqe_accepts_cudaqx_parameter_vector_callable():
    def vector_callable(x):
        single_rx(x[0])

    energy, parameters, trace = vqe(
        vector_callable,
        PauliOperator("Z0"),
        [0.25],
        optimizer=HoldOptimizer(),
    )

    assert energy == pytest.approx(np.cos(0.25), abs=1.0e-7)
    np.testing.assert_allclose(parameters, [0.25])
    assert len(trace) == 1


def test_functional_vqe_forwards_documented_scipy_keywords():
    captured = {}

    def callback(current_parameters):
        return current_parameters

    def scipy_minimize(*, fun, x0, args=(), **kwargs):
        captured.update(kwargs)
        parameters = np.asarray(x0, dtype=float)
        return SimpleNamespace(fun=fun(parameters, *args), x=parameters.copy())

    energy, parameters, _ = vqe(
        lambda x: single_rx(x[0]),
        PauliOperator("Z0"),
        [0.25],
        optimizer=scipy_minimize,
        callback=callback,
        method="L-BFGS-B",
        jac="3-point",
        tol=1.0e-4,
        options={"disp": False},
    )

    assert energy == pytest.approx(np.cos(0.25), abs=1.0e-7)
    np.testing.assert_allclose(parameters, [0.25])
    assert captured == {
        "callback": callback,
        "jac": "3-point",
        "method": "L-BFGS-B",
        "options": {"disp": False},
        "tol": 1.0e-4,
    }

    with pytest.raises(ValueError, match="conflicts"):
        vqe(
            lambda x: single_rx(x[0]),
            PauliOperator("Z0"),
            [0.25],
            optimizer=scipy_minimize,
            method="L-BFGS-B",
            optimizer_options={"method": "BFGS"},
        )


def test_parameter_shift_falls_back_for_scaled_host_parameterization():
    @rocq.kernel
    def scaled_ansatz(theta):
        q = rocq.qvec(1)
        rocq.ry(2.0 * theta, q[0])

    theta = 0.2
    solver = VQE_Solver()
    with pytest.warns(RuntimeWarning, match="central-difference"):
        gradient = solver.estimate_gradient(
            [theta],
            PauliOperator("Z0"),
            scaled_ansatz,
            1,
            method="parameter_shift",
        )

    assert gradient[0] == pytest.approx(-2.0 * np.sin(2.0 * theta), abs=1.0e-7)


def test_qaoa_parameter_counts_and_operator_pool_match_cudaqx_contract():
    problem = PauliOperator("Z0") + 0.5 * PauliOperator("Z0 Z1")
    reference = PauliOperator("X0") + PauliOperator("X1")

    assert get_num_qaoa_parameters(problem, 3) == 6
    assert (
        get_num_qaoa_parameters(
            problem,
            2,
            reference_operator=reference,
            full_parameterization=True,
            num_qubits=2,
        )
        == 8
    )
    assert (
        get_num_qaoa_parameters(
            problem,
            2,
            counterdiabatic=True,
            num_qubits=2,
        )
        == 8
    )

    pool = get_operator_pool("qaoa", num_qubits=2)
    words = {operator.pauli_string.replace(" ", "") for operator in pool}
    assert len(pool) == 12
    assert words == {
        "X0",
        "X1",
        "Y0",
        "Y1",
        "X0X1",
        "Y0Y1",
        "Y0Z1",
        "Z0Y1",
        "X0Y1",
        "Y0X1",
        "X0Z1",
        "Z0X1",
    }
    uccsd_pool = get_operator_pool(
        "uccsd", num_qubits=4, num_electrons=2
    )
    assert len(uccsd_pool) == 3
    with pytest.raises(ValueError, match="unknown operator pool"):
        get_operator_pool("not-a-pool", num_qubits=2)
    with pytest.raises(ValueError, match="non-negative integer"):
        get_operator_pool("qaoa", num_qubits=True)


def test_generic_qaoa_state_matches_dense_positive_exponent_oracle():
    gamma = 0.37
    beta = -0.21
    problem = PauliOperator("Z0")
    reference = PauliOperator("X0")
    ansatz = _make_generic_qaoa_kernel(
        problem,
        reference,
        num_qubits=1,
        layers=1,
        full_parameterization=False,
        counterdiabatic=False,
    )

    actual = rocq.get_state(
        ansatz,
        np.asarray([gamma, beta]),
        backend="state_vector",
    )
    identity = np.eye(2, dtype=np.complex128)
    x_matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    z_matrix = np.diag([1.0, -1.0]).astype(np.complex128)
    plus = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    problem_evolution = np.cos(gamma) * identity + 1.0j * np.sin(gamma) * z_matrix
    mixer_evolution = np.cos(beta) * identity + 1.0j * np.sin(beta) * x_matrix
    expected = mixer_evolution @ problem_evolution @ plus
    np.testing.assert_allclose(actual, expected, atol=1.0e-10)


def test_generic_qaoa_returns_final_sample_counts_and_validates_real_terms():
    result = qaoa(
        PauliOperator("Z0"),
        1,
        [0.0, 0.0],
        optimizer=HoldOptimizer(),
        shots=37,
    )
    assert result.optimal_value == pytest.approx(0.0, abs=1.0e-12)
    assert result.optimal_parameters == (0.0, 0.0)
    assert sum(result.optimal_config.values()) == 37
    assert isinstance(result.optimal_config, rocq.SampleResult)
    optimal_value, optimal_parameters, optimal_config = result
    assert optimal_value == result.optimal_value
    assert optimal_parameters == result.optimal_parameters
    assert optimal_config is result.optimal_config
    assert len(result) == 3
    assert result[2].most_probable() in {"0", "1"}

    with pytest.raises(ValueError, match="real Pauli coefficients"):
        qaoa(
            1.0j * PauliOperator("Z0"),
            1,
            [0.0, 0.0],
            optimizer=HoldOptimizer(),
            shots=10,
        )
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        qaoa(
            PauliOperator("Z0"),
            1,
            [0.0, 0.0],
            optimizer=HoldOptimizer(),
            shots=True,
        )


def test_adapt_vqe_selects_largest_finite_difference_gradient_and_improves():
    energy, parameters, selected = adapt_vqe(
        None,
        PauliOperator("X0"),
        [PauliOperator("Y0")],
        optimizer=OneParameterGridOptimizer(),
        max_iter=1,
        num_qubits=1,
        finite_difference_step=1.0e-5,
    )

    assert energy < -0.999
    assert parameters.size == 1
    assert len(selected) == 1
    assert selected[0].pauli_string == "Y0"
    with pytest.raises(NotImplementedError, match="shots-based ADAPT-VQE"):
        adapt_vqe(
            None,
            PauliOperator("X0"),
            [PauliOperator("Y0")],
            shots=10,
        )
    with pytest.raises(ValueError, match="dynamic_start"):
        adapt_vqe(
            None,
            PauliOperator("X0"),
            [PauliOperator("Y0")],
            dynamic_start="reuse",
        )


def test_adapt_vqe_accepts_one_register_initial_kernel_and_target_energy_stop():
    @rocq.kernel
    def prepare_one(q):
        rocq.x(q[0])

    energy, parameters, selected = adapt_vqe(
        prepare_one,
        PauliOperator("Z0"),
        [PauliOperator("Y0")],
        threshold_energy=-0.9,
        num_qubits=1,
        max_iter=1,
    )
    assert energy == pytest.approx(-1.0)
    assert parameters.size == 0
    assert selected == []


def _fermionic_creation(mode: int, num_modes: int) -> np.ndarray:
    dimension = 1 << num_modes
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        if (column >> mode) & 1:
            continue
        parity = bin(column & ((1 << mode) - 1)).count("1")
        row = column | (1 << mode)
        matrix[row, column] = -1.0 if parity % 2 else 1.0
    return matrix


def test_jordan_wigner_matches_independent_tiny_dense_fermion_oracle():
    one_body = np.asarray(
        [[0.3, 0.7 - 0.2j], [0.7 + 0.2j, -0.4]],
        dtype=np.complex128,
    )
    two_body = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    two_body[0, 1, 1, 0] = 0.25
    core = -0.1
    operator = jordan_wigner(one_body, two_body, core)
    actual = operator_to_matrix(operator, num_qubits=2)

    creation = [_fermionic_creation(mode, 2) for mode in range(2)]
    annihilation = [matrix.conj().T for matrix in creation]
    expected = core * np.eye(4, dtype=np.complex128)
    for p in range(2):
        for q in range(2):
            expected += one_body[p, q] * creation[p] @ annihilation[q]
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    expected += (
                        two_body[p, q, r, s]
                        * creation[p]
                        @ creation[q]
                        @ annihilation[r]
                        @ annihilation[s]
                    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_molecular_hamiltonian_preserves_integrals_and_pyscf_fails_closed(
    monkeypatch,
):
    molecular = MolecularHamiltonian.from_integrals(
        np.asarray([[1.0]]),
        core_energy=0.2,
        n_electrons=1,
        energies={"hf_energy": -0.7},
    )
    assert molecular.n_electrons == 1
    assert molecular.n_orbitals == 1
    assert not molecular.hpq.flags.writeable
    assert not molecular.hpqrs.flags.writeable
    assert molecular.energies == {"hf_energy": -0.7}
    positional_core = jordan_wigner(np.asarray([[1.0]]), 0.2)
    np.testing.assert_allclose(
        operator_to_matrix(positional_core),
        operator_to_matrix(molecular.hamiltonian),
    )
    tolerance_name = jordan_wigner(np.asarray([[1.0]]), tolerance=1.0e-12)
    tolerance_alias = jordan_wigner(np.asarray([[1.0]]), tol=1.0e-12)
    np.testing.assert_allclose(
        operator_to_matrix(tolerance_alias),
        operator_to_matrix(tolerance_name),
    )
    with pytest.raises(ValueError, match="must agree"):
        jordan_wigner(
            np.asarray([[1.0]]),
            tolerance=1.0e-10,
            tol=1.0e-12,
        )
    with pytest.raises(ValueError, match="rank-2"):
        jordan_wigner([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        jordan_wigner(np.asarray([[np.nan]]))
    import rocquantum.solvers.chemistry as chemistry_module

    def missing_pyscf(module_name):
        raise ModuleNotFoundError(
            f"No module named '{module_name}'", name="pyscf"
        )

    monkeypatch.setattr(chemistry_module, "import_module", missing_pyscf)
    with pytest.raises(ImportError, match=r"rocquantum\[chemistry\]"):
        create_molecule(
            [("H", (0.0, 0.0, 0.0))],
            "sto-3g",
            0,
            0,
        )
