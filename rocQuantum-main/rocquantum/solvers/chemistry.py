"""Host chemistry primitives compatible with the CUDA-QX solver surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from numbers import Integral, Number, Real
from pathlib import Path

import numpy as np

try:
    from rocq.operator import (
        PauliOperator,
        QuantumOperator,
        SumOperator,
        iter_pauli_terms,
    )
except ImportError:  # pragma: no cover - import-only environments.
    PauliOperator = None  # type: ignore
    QuantumOperator = None  # type: ignore
    SumOperator = None  # type: ignore
    iter_pauli_terms = None  # type: ignore


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_tolerance(value) -> float:
    tolerance = _finite_real(value, "tolerance")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    return tolerance


def _integer(value, name: str, *, minimum=None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


def _strict_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


def _nonempty_string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _complex_array(values, name: str, rank: int) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric rank-{rank} tensor.") from exc
    if raw.ndim != rank:
        raise ValueError(f"{name} must be a rank-{rank} tensor.")
    normalized = []
    for value in raw.reshape(-1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise ValueError(f"{name} must contain finite numeric values.")
        scalar = complex(value)
        if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
            raise ValueError(f"{name} must contain finite values.")
        normalized.append(scalar)
    return np.asarray(normalized, dtype=np.complex128).reshape(raw.shape)


def _normalize_atom(entry, index: int):
    if isinstance(entry, (str, bytes, Mapping)):
        raise ValueError(
            "geometry entries must be ('ELEMENT', (X, Y, Z)) pairs."
        )
    try:
        atom = list(entry)
    except TypeError as exc:
        raise ValueError(
            "geometry entries must be ('ELEMENT', (X, Y, Z)) pairs."
        ) from exc
    if len(atom) != 2:
        raise ValueError(
            "geometry entries must be ('ELEMENT', (X, Y, Z)) pairs."
        )
    symbol = _nonempty_string(atom[0], f"geometry[{index}] element")
    if isinstance(atom[1], (str, bytes, Mapping)):
        raise ValueError("geometry coordinates must contain exactly three values.")
    try:
        raw_coordinates = list(atom[1])
    except TypeError as exc:
        raise ValueError(
            "geometry coordinates must contain exactly three values."
        ) from exc
    if len(raw_coordinates) != 3:
        raise ValueError("geometry coordinates must contain exactly three values.")
    coordinates = tuple(
        _finite_real(value, f"geometry[{index}] coordinate")
        for value in raw_coordinates
    )
    return symbol, coordinates


def _read_xyz_geometry(path_value) -> tuple:
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"XYZ geometry file does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"XYZ geometry file could not be read: {path}") from exc
    if len(lines) < 2:
        raise ValueError("XYZ geometry must contain an atom count and comment line.")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("XYZ geometry first line must be a positive atom count.") from exc
    if atom_count <= 0:
        raise ValueError("XYZ geometry first line must be a positive atom count.")
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError("XYZ geometry contains fewer atom rows than declared.")

    atoms = []
    for index, line in enumerate(atom_lines):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                "XYZ atom rows must contain an element and three coordinates."
            )
        try:
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(
                "XYZ atom coordinates must be finite real numbers."
            ) from exc
        atoms.append(_normalize_atom((fields[0], coordinates), index))
    return tuple(atoms)


def _normalize_geometry(geometry) -> tuple:
    if isinstance(geometry, (str, Path)):
        return _read_xyz_geometry(geometry)
    if isinstance(geometry, (bytes, Mapping)):
        raise ValueError(
            "geometry must be an XYZ path or a non-empty sequence of atom pairs."
        )
    try:
        entries = list(geometry)
    except TypeError as exc:
        raise ValueError(
            "geometry must be an XYZ path or a non-empty sequence of atom pairs."
        ) from exc
    if not entries:
        raise ValueError("geometry must contain at least one atom.")
    return tuple(_normalize_atom(entry, index) for index, entry in enumerate(entries))


def _normalize_integrals(hpq, hpqrs=None):
    first = np.asarray(hpq, dtype=object)
    if hpqrs is None and first.ndim == 4:
        two_body = _complex_array(hpq, "hpqrs", 4)
        if len(set(two_body.shape)) != 1:
            raise ValueError("hpqrs dimensions must all have the same size.")
        dimension = int(two_body.shape[0])
        one_body = np.zeros((dimension, dimension), dtype=np.complex128)
        return one_body, two_body

    one_body = _complex_array(hpq, "hpq", 2)
    if one_body.shape[0] != one_body.shape[1] or one_body.shape[0] <= 0:
        raise ValueError("hpq must be a non-empty square tensor.")
    dimension = int(one_body.shape[0])
    if hpqrs is None:
        two_body = np.zeros(
            (dimension, dimension, dimension, dimension),
            dtype=np.complex128,
        )
    else:
        two_body = _complex_array(hpqrs, "hpqrs", 4)
        if two_body.shape != (dimension, dimension, dimension, dimension):
            raise ValueError("hpqrs shape must be (n, n, n, n) matching hpq.")
    return one_body, two_body


def _load_pyscf():
    """Load the optional chemistry dependency only at the adapter boundary."""

    try:
        return tuple(
            import_module(module_name)
            for module_name in (
                "pyscf.ao2mo",
                "pyscf.fci",
                "pyscf.gto",
                "pyscf.scf",
            )
        )
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", "") or ""
        if missing == "pyscf" or missing.startswith("pyscf."):
            raise ImportError(
                "create_molecule() requires the optional PySCF dependency. "
                "Install it with 'pip install rocquantum[chemistry]' or "
                "'pip install pyscf>=2.3', then retry."
            ) from exc
        raise


def _restricted_spin_integrals(one_body, two_body):
    """Expand restricted spatial-orbital integrals to interleaved spin orbitals.

    The layout follows CUDA-QX's PySCF adapter: alpha/beta spin orbitals are
    interleaved, and the two-body tensor already includes the one-half
    coefficient expected by :func:`jordan_wigner`.
    """

    spatial_one_body = _complex_array(one_body, "spatial one-body integrals", 2)
    spatial_two_body = _complex_array(two_body, "spatial two-body integrals", 4)
    orbitals = int(spatial_one_body.shape[0])
    if spatial_one_body.shape != (orbitals, orbitals):
        raise ValueError("spatial one-body integrals must be square.")
    if spatial_two_body.shape != (orbitals, orbitals, orbitals, orbitals):
        raise ValueError(
            "spatial two-body integral shape must match the orbital count."
        )

    modes = 2 * orbitals
    hpq = np.zeros((modes, modes), dtype=np.complex128)
    hpqrs = np.zeros((modes, modes, modes, modes), dtype=np.complex128)
    for p in range(orbitals):
        for q in range(orbitals):
            hpq[2 * p, 2 * q] = spatial_one_body[p, q]
            hpq[2 * p + 1, 2 * q + 1] = spatial_one_body[p, q]
            for r in range(orbitals):
                for s in range(orbitals):
                    coefficient = 0.5 * spatial_two_body[p, q, r, s]
                    hpqrs[2 * p, 2 * q, 2 * r, 2 * s] = coefficient
                    hpqrs[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = (
                        coefficient
                    )
                    hpqrs[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = coefficient
                    hpqrs[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = coefficient
    return hpq, hpqrs


def _jw_ladder(index: int, *, creation: bool):
    z_product = PauliOperator("I")
    for qubit in range(index):
        z_product = z_product * PauliOperator(f"Z{qubit}")
    sign = -1.0j if creation else 1.0j
    return 0.5 * z_product * (
        PauliOperator(f"X{index}") + sign * PauliOperator(f"Y{index}")
    )


def _canonical_pauli_sum(operator, tolerance: float):
    combined = {}
    for coefficient, paulis in iter_pauli_terms(operator):
        key = tuple(sorted((str(pauli).upper(), int(qubit)) for pauli, qubit in paulis))
        combined[key] = combined.get(key, 0.0j) + complex(coefficient)

    terms = []
    for paulis, coefficient in sorted(combined.items()):
        if abs(coefficient.real) < tolerance and abs(coefficient.imag) < tolerance:
            continue
        if abs(coefficient.imag) < tolerance:
            coefficient = complex(coefficient.real, 0.0)
        word = "I" if not paulis else " ".join(
            f"{pauli}{qubit}" for pauli, qubit in paulis
        )
        terms.append(PauliOperator(word, coefficient=coefficient))

    if not terms:
        return PauliOperator("I", coefficient=0.0)
    if len(terms) == 1:
        return terms[0]
    return SumOperator(terms)


def jordan_wigner(
    hpq,
    hpqrs=None,
    core_energy: float = 0.0,
    *,
    tolerance: float = 1.0e-15,
    tol=None,
):
    """Transform precomputed one-/two-body integrals to a Pauli Hamiltonian.

    This matches CUDA-QX's integral convention directly:
    ``sum hpq[p,q] a†_p a_q + sum hpqrs[p,q,r,s] a†_p a†_q a_r a_s``.
    Any conventional one-half factor must already be included in ``hpqrs``.
    """

    if PauliOperator is None:
        raise RuntimeError("Canonical 'rocq' package is required for jordan_wigner().")
    if isinstance(hpqrs, Number):
        if core_energy != 0.0:
            raise ValueError("core_energy was provided more than once.")
        core_energy = hpqrs
        hpqrs = None
    one_body, two_body = _normalize_integrals(hpq, hpqrs)
    core_energy = _finite_real(core_energy, "core_energy")
    normalized_tolerance = _positive_tolerance(tolerance)
    if tol is not None:
        alias_tolerance = _positive_tolerance(tol)
        if tolerance != 1.0e-15 and alias_tolerance != normalized_tolerance:
            raise ValueError("tolerance and tol must agree when both are provided.")
        normalized_tolerance = alias_tolerance
    tolerance = normalized_tolerance
    dimension = int(one_body.shape[0])

    creation = [_jw_ladder(index, creation=True) for index in range(dimension)]
    annihilation = [_jw_ladder(index, creation=False) for index in range(dimension)]
    hamiltonian = PauliOperator("I", coefficient=core_energy)

    for p in range(dimension):
        for q in range(dimension):
            coefficient = one_body[p, q]
            if abs(coefficient.real) < tolerance and abs(coefficient.imag) < tolerance:
                continue
            hamiltonian = hamiltonian + coefficient * creation[p] * annihilation[q]

    for p in range(dimension):
        for q in range(dimension):
            for r in range(dimension):
                for s in range(dimension):
                    coefficient = two_body[p, q, r, s]
                    if (
                        abs(coefficient.real) < tolerance
                        and abs(coefficient.imag) < tolerance
                    ):
                        continue
                    hamiltonian = (
                        hamiltonian
                        + coefficient
                        * creation[p]
                        * creation[q]
                        * annihilation[r]
                        * annihilation[s]
                    )
    return _canonical_pauli_sum(hamiltonian, tolerance)


@dataclass(frozen=True)
class MolecularHamiltonian:
    """Container for a Pauli Hamiltonian and its precomputed integrals."""

    hamiltonian: object
    hpq: np.ndarray
    hpqrs: np.ndarray
    n_electrons: int
    n_orbitals: int
    energies: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if QuantumOperator is not None and not isinstance(
            self.hamiltonian, QuantumOperator
        ):
            raise ValueError("hamiltonian must be a rocq.operator.QuantumOperator.")
        hpq, hpqrs = _normalize_integrals(self.hpq, self.hpqrs)
        for name, value in (
            ("n_electrons", self.n_electrons),
            ("n_orbitals", self.n_orbitals),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a non-negative integer.")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not isinstance(self.energies, Mapping):
            raise ValueError("energies must be a string-to-finite-real mapping.")
        energies = {}
        for key, value in self.energies.items():
            if not isinstance(key, str) or not key:
                raise ValueError("energies keys must be non-empty strings.")
            energies[key] = _finite_real(value, f"energies['{key}']")
        hpq.setflags(write=False)
        hpqrs.setflags(write=False)
        object.__setattr__(self, "hpq", hpq)
        object.__setattr__(self, "hpqrs", hpqrs)
        object.__setattr__(self, "n_electrons", int(self.n_electrons))
        object.__setattr__(self, "n_orbitals", int(self.n_orbitals))
        object.__setattr__(self, "energies", energies)

    @classmethod
    def from_integrals(
        cls,
        hpq,
        hpqrs=None,
        core_energy: float = 0.0,
        *,
        n_electrons: int = 0,
        n_orbitals=None,
        energies=None,
        tolerance: float = 1.0e-15,
    ):
        one_body, two_body = _normalize_integrals(hpq, hpqrs)
        if n_orbitals is None:
            n_orbitals = int(one_body.shape[0])
        operator = jordan_wigner(
            one_body,
            two_body,
            core_energy,
            tolerance=tolerance,
        )
        return cls(
            operator,
            one_body,
            two_body,
            n_electrons,
            n_orbitals,
            {} if energies is None else energies,
        )


def create_molecule(
    geometry,
    basis,
    spin,
    charge,
    *,
    driver: str = "pyscf",
    fermion_to_spin: str = "jordan_wigner",
    type: str = "gas_phase",
    symmetry: bool = False,
    memory: float = 4000.0,
    cycles: int = 100,
    initguess: str = "minao",
    UR: bool = False,
    nele_cas=None,
    norb_cas=None,
    MP2: bool = False,
    natorb: bool = False,
    casci: bool = False,
    ccsd: bool = False,
    casscf: bool = False,
    integrals_natorb: bool = False,
    integrals_casscf: bool = False,
    verbose: bool = False,
    tolerance: float = 1.0e-15,
):
    """Create a restricted molecular Hamiltonian through optional PySCF.

    This is a correctness-oriented CUDA-QX-compatible subset.  It accepts an
    atom list or XYZ path, runs RHF/ROHF, transforms full-space molecular
    integrals to interleaved spin orbitals, and optionally records an exact FCI
    energy when ``casci=True``.  Active-space, unrestricted, correlated-orbital,
    and non-Jordan-Wigner workflows fail explicitly rather than being ignored.
    """

    atoms = _normalize_geometry(geometry)
    basis = _nonempty_string(basis, "basis")
    spin = _integer(spin, "spin", minimum=0)
    charge = _integer(charge, "charge")
    driver = _nonempty_string(driver, "driver").lower()
    if driver != "pyscf":
        raise NotImplementedError(
            "Only the direct optional PySCF driver is implemented; use driver='pyscf'."
        )
    molecule_type = _nonempty_string(type, "type").lower()
    if molecule_type != "gas_phase":
        raise NotImplementedError(
            "Only type='gas_phase' molecular construction is implemented."
        )
    mapping = _nonempty_string(fermion_to_spin, "fermion_to_spin").lower()
    if mapping not in {"jordan_wigner", "jordan-wigner", "jw"}:
        raise NotImplementedError(
            "create_molecule() currently supports only Jordan-Wigner mapping."
        )

    symmetry = _strict_bool(symmetry, "symmetry")
    memory = _finite_real(memory, "memory")
    if memory <= 0.0:
        raise ValueError("memory must be positive.")
    cycles = _integer(cycles, "cycles", minimum=1)
    initguess = _nonempty_string(initguess, "initguess")
    ur = _strict_bool(UR, "UR")
    casci = _strict_bool(casci, "casci")
    verbose = _strict_bool(verbose, "verbose")
    advanced_flags = {
        "MP2": _strict_bool(MP2, "MP2"),
        "natorb": _strict_bool(natorb, "natorb"),
        "ccsd": _strict_bool(ccsd, "ccsd"),
        "casscf": _strict_bool(casscf, "casscf"),
        "integrals_natorb": _strict_bool(
            integrals_natorb, "integrals_natorb"
        ),
        "integrals_casscf": _strict_bool(
            integrals_casscf, "integrals_casscf"
        ),
    }
    tolerance = _positive_tolerance(tolerance)

    if (nele_cas is None) != (norb_cas is None):
        raise ValueError("nele_cas and norb_cas must be provided together.")
    if nele_cas is not None:
        raise NotImplementedError(
            "Active-space molecule construction is not implemented in this "
            "reference subset; omit nele_cas and norb_cas."
        )
    if ur:
        raise NotImplementedError(
            "Unrestricted PySCF integrals are not implemented in this reference subset."
        )
    requested_advanced = [name for name, enabled in advanced_flags.items() if enabled]
    if requested_advanced:
        raise NotImplementedError(
            "The following PySCF workflow options are not implemented in this "
            f"reference subset: {', '.join(requested_advanced)}."
        )

    ao2mo, fci, gto, scf = _load_pyscf()
    mol = gto.M(
        atom=list(atoms),
        basis=basis,
        spin=spin,
        charge=charge,
        unit="Angstrom",
        symmetry=symmetry,
        max_memory=memory,
        verbose=4 if verbose else 0,
    )
    mean_field = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mean_field.max_cycle = cycles
    mean_field.init_guess = initguess
    hf_energy = _finite_real(mean_field.kernel(), "PySCF HF energy")
    if not bool(getattr(mean_field, "converged", False)):
        raise RuntimeError(
            "PySCF Hartree-Fock calculation did not converge; adjust the "
            "geometry, basis, cycles, or initguess and retry."
        )

    mo_coeff_complex = _complex_array(
        mean_field.mo_coeff, "PySCF molecular-orbital coefficients", 2
    )
    if np.any(np.abs(mo_coeff_complex.imag) > tolerance):
        raise NotImplementedError(
            "Complex PySCF molecular orbitals are not implemented in this "
            "restricted reference subset."
        )
    # Passing a complex dtype to PySCF selects its GHF/spinor AO2MO path even
    # when every imaginary component is zero.  Preserve the real RHF/ROHF
    # contract at this adapter boundary.
    mo_coeff = np.asarray(mo_coeff_complex.real, dtype=np.float64)
    hcore = _complex_array(mean_field.get_hcore(), "PySCF core Hamiltonian", 2)
    if hcore.shape[0] != hcore.shape[1] or hcore.shape[0] != mo_coeff.shape[0]:
        raise RuntimeError(
            "PySCF returned incompatible core-Hamiltonian and MO-coefficient shapes."
        )
    one_body_spatial = mo_coeff.conj().T @ hcore @ mo_coeff
    num_orbitals = int(mo_coeff.shape[1])
    raw_eri = np.asarray(
        ao2mo.kernel(mol, mo_coeff, compact=False), dtype=np.complex128
    )
    expected_eri_size = num_orbitals**4
    if raw_eri.size != expected_eri_size:
        raise RuntimeError(
            "PySCF AO-to-MO transformation returned an unexpected integral shape."
        )
    # Match CUDA-QX's restricted adapter convention:
    # (pr|qs)_chemist -> h[p,q,r,s] for a†_p a†_q a_r a_s.
    two_body_spatial = raw_eri.reshape(
        (num_orbitals, num_orbitals, num_orbitals, num_orbitals)
    ).transpose(0, 2, 3, 1)
    hpq, hpqrs = _restricted_spin_integrals(
        one_body_spatial, two_body_spatial
    )

    core_energy = _finite_real(mol.energy_nuc(), "PySCF nuclear energy")
    energies = {
        "nuclear_energy": core_energy,
        "hf_energy": hf_energy,
    }
    if casci:
        fci_result = fci.FCI(mean_field).kernel()
        fci_energy = fci_result[0] if isinstance(fci_result, tuple) else fci_result
        energies["fci_energy"] = _finite_real(fci_energy, "PySCF FCI energy")

    hamiltonian = jordan_wigner(
        hpq,
        hpqrs,
        core_energy,
        tolerance=tolerance,
    )
    return MolecularHamiltonian(
        hamiltonian=hamiltonian,
        hpq=hpq,
        hpqrs=hpqrs,
        n_electrons=_integer(mol.nelectron, "PySCF electron count", minimum=0),
        n_orbitals=num_orbitals,
        energies=energies,
    )
