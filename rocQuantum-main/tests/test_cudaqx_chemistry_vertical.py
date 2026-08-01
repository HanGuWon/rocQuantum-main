"""CPU vertical tests for the CUDA-QX chemistry and UCCSD subset."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


import rocq
from rocq.operator import PauliOperator, iter_pauli_terms, operator_to_matrix
from rocquantum.solvers import (
    MolecularHamiltonian,
    adapt_vqe,
    create_molecule,
    get_available_operator_pools,
    get_operator_pool,
    operator_pool,
    stateprep,
)
import rocquantum.solvers.operator_pools as operator_pools_module


class OneParameterGridOptimizer:
    def minimize(self, fun, x0, args=()):
        parameters = np.asarray(x0, dtype=float)
        if parameters.size != 1:
            raise AssertionError("test optimizer only supports one parameter")
        candidates = np.linspace(-np.pi, np.pi, 321)
        energies = [fun(np.asarray([value]), *args) for value in candidates]
        index = int(np.argmin(energies))
        return SimpleNamespace(
            fun=float(energies[index]),
            x=np.asarray([candidates[index]], dtype=float),
        )


def _install_fake_pyscf(monkeypatch):
    captured = {}
    raw_eri = np.arange(16, dtype=float).reshape((2, 2, 2, 2)) / 20.0

    class FakeMolecule:
        nelectron = 2

        def energy_nuc(self):
            return 0.7

    class FakeMeanField:
        def __init__(self, molecule):
            captured["mean_field_molecule"] = molecule
            self.mo_coeff = np.eye(2)
            self.converged = True
            self.max_cycle = None
            self.init_guess = None

        def kernel(self):
            captured["max_cycle"] = self.max_cycle
            captured["init_guess"] = self.init_guess
            return -1.1

        def get_hcore(self):
            return np.asarray([[-1.0, 0.2], [0.2, -0.4]])

    parent = ModuleType("pyscf")
    parent.__path__ = []
    ao2mo = ModuleType("pyscf.ao2mo")
    fci = ModuleType("pyscf.fci")
    gto = ModuleType("pyscf.gto")
    scf = ModuleType("pyscf.scf")

    def make_molecule(**kwargs):
        captured["gto"] = kwargs
        return FakeMolecule()

    def transform_eri(molecule, coefficients, compact):
        captured["ao2mo"] = (molecule, np.asarray(coefficients), compact)
        return raw_eri.reshape(-1)

    class FakeFCI:
        def __init__(self, mean_field):
            captured["fci_mean_field"] = mean_field

        def kernel(self):
            return -1.2, object()

    gto.M = make_molecule
    scf.RHF = FakeMeanField
    scf.ROHF = FakeMeanField
    ao2mo.kernel = transform_eri
    fci.FCI = FakeFCI
    for module in (parent, ao2mo, fci, gto, scf):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return captured, raw_eri


def _dense_terms(operator, num_qubits):
    terms = {}
    for coefficient, paulis in iter_pauli_terms(operator):
        word = ["I"] * num_qubits
        for pauli, qubit in paulis:
            word[int(qubit)] = str(pauli).upper()
        terms["".join(word)] = complex(coefficient)
    return terms


def test_create_molecule_fake_pyscf_preserves_cudaqx_integral_convention(
    monkeypatch,
):
    captured, raw_eri = _install_fake_pyscf(monkeypatch)

    molecule = create_molecule(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.75))],
        "sto-3g",
        0,
        0,
        casci=True,
        cycles=77,
        initguess="atom",
    )

    assert molecule.n_electrons == 2
    assert molecule.n_orbitals == 2
    assert molecule.hpq.shape == (4, 4)
    assert molecule.hpqrs.shape == (4, 4, 4, 4)
    assert molecule.energies == {
        "nuclear_energy": pytest.approx(0.7),
        "hf_energy": pytest.approx(-1.1),
        "fci_energy": pytest.approx(-1.2),
    }
    np.testing.assert_allclose(
        molecule.hpq,
        np.asarray(
            [
                [-1.0, 0.0, 0.2, 0.0],
                [0.0, -1.0, 0.0, 0.2],
                [0.2, 0.0, -0.4, 0.0],
                [0.0, 0.2, 0.0, -0.4],
            ]
        ),
    )
    expected_spatial = raw_eri.transpose(0, 2, 3, 1)
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    expected = 0.5 * expected_spatial[p, q, r, s]
                    assert molecule.hpqrs[2 * p, 2 * q, 2 * r, 2 * s] == expected
                    assert (
                        molecule.hpqrs[
                            2 * p + 1,
                            2 * q + 1,
                            2 * r + 1,
                            2 * s + 1,
                        ]
                        == expected
                    )
                    assert (
                        molecule.hpqrs[
                            2 * p,
                            2 * q + 1,
                            2 * r + 1,
                            2 * s,
                        ]
                        == expected
                    )
                    assert (
                        molecule.hpqrs[
                            2 * p + 1,
                            2 * q,
                            2 * r,
                            2 * s + 1,
                        ]
                        == expected
                    )
    assert captured["max_cycle"] == 77
    assert captured["init_guess"] == "atom"
    assert captured["ao2mo"][2] is False
    assert not molecule.hpq.flags.writeable
    assert not molecule.hpqrs.flags.writeable


def test_create_molecule_accepts_xyz_path(monkeypatch, tmp_path):
    captured, _ = _install_fake_pyscf(monkeypatch)
    xyz_path = tmp_path / "h2.xyz"
    xyz_path.write_text(
        "2\nH2 test geometry\nH 0.0 0.0 0.0\nH 0.0 0.0 0.75\n",
        encoding="utf-8",
    )

    molecule = create_molecule(xyz_path, "sto-3g", 0, 0)

    assert molecule.n_orbitals == 2
    assert captured["gto"]["atom"] == [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.75)),
    ]


def test_create_molecule_rejects_unsupported_workflows_before_import():
    geometry = [("H", (0.0, 0.0, 0.0))]
    with pytest.raises(NotImplementedError, match="Active-space"):
        create_molecule(
            geometry,
            "sto-3g",
            0,
            0,
            nele_cas=1,
            norb_cas=1,
        )
    with pytest.raises(NotImplementedError, match="Unrestricted"):
        create_molecule(geometry, "sto-3g", 0, 0, UR=True)
    with pytest.raises(NotImplementedError, match="Jordan-Wigner"):
        create_molecule(
            geometry,
            "sto-3g",
            0,
            0,
            fermion_to_spin="bravyi_kitaev",
        )
    with pytest.raises(ValueError, match="coordinates"):
        create_molecule([("H", (0.0, 0.0))], "sto-3g", 0, 0)


def test_optional_real_pyscf_h2_sto3g_known_energy_and_shapes():
    if importlib.util.find_spec("pyscf") is None:
        pytest.skip("optional PySCF dependency is not installed")

    molecule = create_molecule(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))],
        "sto-3g",
        0,
        0,
        casci=True,
    )

    assert molecule.n_electrons == 2
    assert molecule.n_orbitals == 2
    assert molecule.hpq.shape == (4, 4)
    assert molecule.hpqrs.shape == (4, 4, 4, 4)
    assert molecule.energies["hf_energy"] == pytest.approx(-1.117, abs=0.01)
    assert molecule.energies["fci_energy"] == pytest.approx(-1.137, abs=0.01)
    dense = operator_to_matrix(molecule.hamiltonian, num_qubits=4)
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-10)
    two_electron_basis = [
        basis_state
        for basis_state in range(16)
        if bin(basis_state).count("1") == 2
    ]
    sector = dense[np.ix_(two_electron_basis, two_electron_basis)]
    assert np.linalg.eigvalsh(sector)[0] == pytest.approx(
        molecule.energies["fci_energy"],
        abs=1.0e-8,
    )

    from scipy.optimize import minimize
    from rocquantum.solvers import vqe

    @rocq.kernel
    def h2_uccsd(parameters):
        qubits = rocq.qvec(4)
        rocq.x(qubits[0])
        rocq.x(qubits[1])
        stateprep.uccsd(qubits, parameters, 2, 0)

    energy, _, _ = vqe(
        h2_uccsd,
        molecule.hamiltonian,
        [-0.2, -0.2, -0.2],
        optimizer=minimize,
        method="L-BFGS-B",
        jac="3-point",
        tol=1.0e-8,
        options={"maxiter": 200},
        backend="qpp-cpu",
    )
    assert energy == pytest.approx(molecule.energies["fci_energy"], abs=1.0e-6)


def test_uccsd_excitations_parameter_count_and_pool_match_cudaqx_h2():
    assert stateprep.get_uccsd_excitations(2, 4, 0) == (
        [[0, 2]],
        [[1, 3]],
        [[0, 1, 3, 2]],
        [],
        [],
    )
    assert stateprep.get_num_uccsd_parameters(2, 4, 0) == 3
    assert stateprep.get_num_uccsd_parameters(3, 6, 1) == 8
    assert stateprep.get_num_uccsd_parameters(10, 20, 0) == 875

    pool = get_operator_pool("uccsd", num_qubits=4, num_electrons=2)
    assert len(pool) == 3
    assert _dense_terms(pool[0], 4) == {
        "YZXI": 0.5 + 0.0j,
        "XZYI": -0.5 + 0.0j,
    }
    assert _dense_terms(pool[1], 4) == {
        "IYZX": 0.5 + 0.0j,
        "IXZY": -0.5 + 0.0j,
    }
    assert _dense_terms(pool[2], 4) == {
        "XXXY": 0.125 + 0.0j,
        "XXYX": 0.125 + 0.0j,
        "XYYY": 0.125 + 0.0j,
        "YXYY": 0.125 + 0.0j,
        "XYXX": -0.125 + 0.0j,
        "YXXX": -0.125 + 0.0j,
        "YYXY": -0.125 + 0.0j,
        "YYYX": -0.125 + 0.0j,
    }
    assert len(get_operator_pool("uccsd", n_qubits=6, n_electrons=3, spin=1)) == 8
    assert get_operator_pool("uccsd", num_qubits=2, num_electrons=2) == []


def test_operator_pool_registry_validates_factories(monkeypatch):
    monkeypatch.setattr(
        operator_pools_module,
        "_POOL_FACTORIES",
        dict(operator_pools_module._POOL_FACTORIES),
    )

    @operator_pool("one-body-test")
    def one_body_test(**config):
        return [PauliOperator(f"X{config['qubit']}")]

    assert "one-body-test" in get_available_operator_pools()
    generated = get_operator_pool("one-body-test", qubit=2)
    assert generated[0].pauli_string == "X2"

    @operator_pool("empty-test")
    def empty_test(**config):
        return []

    assert get_operator_pool("empty-test") == []


def test_uccsd_stateprep_preserves_particle_number_and_validates_parameters():
    theta = 0.4

    @rocq.kernel
    def single_reference():
        qubits = rocq.qvec(3)
        rocq.x(qubits[0])
        stateprep.single_excitation(qubits, theta, 0, 2)

    single_state = rocq.get_state(single_reference, backend="qpp-cpu")
    single_expected = np.zeros(8, dtype=np.complex128)
    single_expected[1] = np.cos(theta / 2.0)
    single_expected[4] = -np.sin(theta / 2.0)
    np.testing.assert_allclose(single_state, single_expected, atol=1.0e-12)

    @rocq.kernel
    def double_reference():
        qubits = rocq.qvec(4)
        rocq.x(qubits[0])
        rocq.x(qubits[1])
        stateprep.double_excitation(qubits, theta, 0, 1, 3, 2)

    double_state = rocq.get_state(double_reference, backend="qpp-cpu")
    double_expected = np.zeros(16, dtype=np.complex128)
    double_expected[3] = np.cos(theta / 2.0)
    double_expected[12] = -np.sin(theta / 2.0)
    np.testing.assert_allclose(double_state, double_expected, atol=1.0e-12)

    @rocq.kernel
    def ansatz(parameters):
        qubits = rocq.qvec(4)
        rocq.x(qubits[0])
        rocq.x(qubits[1])
        stateprep.uccsd(qubits, parameters, 2, 0)

    zero_state = rocq.get_state(
        ansatz,
        np.zeros(3),
        backend="qpp-cpu",
    )
    expected = np.zeros(16, dtype=np.complex128)
    expected[3] = 1.0
    np.testing.assert_allclose(zero_state, expected, atol=1.0e-12)

    state = rocq.get_state(
        ansatz,
        np.asarray([0.17, -0.23, 0.31]),
        backend="qpp-cpu",
    )
    outside_probability = sum(
        abs(amplitude) ** 2
        for basis_state, amplitude in enumerate(state)
        if bin(basis_state).count("1") != 2
    )
    assert outside_probability == pytest.approx(0.0, abs=1.0e-12)
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=1.0e-12)

    with pytest.raises(ValueError, match="exactly 3"):
        rocq.get_state(ansatz, np.zeros(2), backend="qpp-cpu")
    with pytest.raises(ValueError, match="same parity"):
        stateprep.get_uccsd_excitations(3, 6, 0)
    with pytest.raises(ValueError, match="even"):
        stateprep.get_uccsd_excitations(2, 3, 0)


def test_uccsd_operator_pool_connects_to_adapt_vqe_cpu_vertical():
    hpq = np.zeros((4, 4), dtype=np.complex128)
    hpq[0, 2] = 1.0
    hpq[2, 0] = 1.0
    molecule = MolecularHamiltonian.from_integrals(
        hpq,
        n_electrons=1,
        n_orbitals=2,
    )
    pool = get_operator_pool(
        "uccsd",
        num_qubits=4,
        num_electrons=1,
        spin=1,
    )
    assert len(pool) == 1

    @rocq.kernel
    def hartree_fock(qubits):
        rocq.x(qubits[0])

    energy, parameters, selected = adapt_vqe(
        hartree_fock,
        molecule.hamiltonian,
        pool,
        optimizer=OneParameterGridOptimizer(),
        max_iter=1,
        num_qubits=4,
        finite_difference_step=1.0e-5,
        backend="qpp-cpu",
    )

    assert energy < -0.999
    assert parameters.size == 1
    assert selected == pool
