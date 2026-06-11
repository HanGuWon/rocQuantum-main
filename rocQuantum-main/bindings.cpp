#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "rocqCompiler/MLIRCompiler.h"
#include "rocqCompiler/QuantumBackend.h"
#include "rocquantum/QuantumSimulator.h"

namespace py = pybind11;

namespace {

std::vector<std::size_t> copy_nonnegative_indices(
    py::array_t<long long, py::array::c_style | py::array::forcecast> values,
    const char* label) {
    std::vector<std::size_t> out(static_cast<std::size_t>(values.size()));
    for (std::size_t idx = 0; idx < out.size(); ++idx) {
        if (values.data()[idx] < 0) {
            throw std::invalid_argument(std::string(label) + " must be non-negative.");
        }
        out[idx] = static_cast<std::size_t>(values.data()[idx]);
    }
    return out;
}

void apply_sparse_matrix_from_python(
    rocquantum::QuantumSimulator& self,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data,
    py::array_t<long long, py::array::c_style | py::array::forcecast> indices,
    py::array_t<long long, py::array::c_style | py::array::forcecast> indptr,
    const std::pair<std::size_t, std::size_t>& shape,
    const std::vector<unsigned>& targets) {
    std::vector<std::complex<double>> host_data(static_cast<std::size_t>(data.size()));
    if (!host_data.empty()) {
        std::memcpy(host_data.data(), data.data(), host_data.size() * sizeof(std::complex<double>));
    }
    auto host_indices = copy_nonnegative_indices(indices, "Sparse operation CSR indices");
    auto host_indptr = copy_nonnegative_indices(indptr, "Sparse operation CSR indptr");
    self.apply_sparse_matrix(host_data, host_indices, host_indptr, shape.first, shape.second, targets);
}

} // namespace

PYBIND11_MODULE(rocquantum_bind, m) {
    m.doc() = "pybind11 plugin for rocQuantum-1";

    py::class_<rocq::MLIRCompiler>(m, "MLIRCompiler")
        .def(py::init([](unsigned num_qubits, const std::string& backend_name) {
            auto backend = rocq::create_backend(backend_name);
            return std::make_unique<rocq::MLIRCompiler>(num_qubits, std::move(backend));
        }))
        .def("compile_and_execute",
             [](rocq::MLIRCompiler &self, const std::string &mlir, py::dict _args) {
                 return self.compile_and_execute(mlir, {});
             },
             py::arg("mlir"),
             py::arg("args") = py::dict(),
             "Executes the supported MLIR subset (qalloc, H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, CRX/CRY/CRZ/CP) "
             "through the selected backend and returns the final state vector. "
             "Unsupported ops raise actionable diagnostics.")
        .def("emit_qir", &rocq::MLIRCompiler::emit_qir,
             "Compiles the MLIR string down to QIR (LLVM IR).");

    auto simulator = py::class_<rocquantum::QuantumSimulator>(m, "QuantumSimulator");

    simulator
        .def(py::init<unsigned, std::size_t>(), py::arg("num_qubits"), py::arg("batch_size") = 1)
        .def("reset", &rocquantum::QuantumSimulator::reset)
        .def("measure_qubit",
             &rocquantum::QuantumSimulator::measure_qubit,
             py::arg("target"),
             "Measure one qubit, collapse the state, and return the sampled bit.")
        .def("reset_qubit",
             &rocquantum::QuantumSimulator::reset_qubit,
             py::arg("target"),
             "Reset one qubit by native measurement collapse followed by conditional X.")
        .def("apply_gate",
             [](rocquantum::QuantumSimulator& self,
                const std::string& gate_name,
                const std::vector<unsigned>& targets,
                const std::vector<double>& params) {
                 self.apply_gate(gate_name, targets, params);
             },
             py::arg("gate_name"),
             py::arg("targets"),
             py::arg("params") = std::vector<double>{})
        .def("apply_gate_batch",
             [](rocquantum::QuantumSimulator& self,
                const std::string& gate_name,
                const std::vector<unsigned>& targets,
                const std::vector<double>& params_by_batch) {
                 self.apply_gate_batch(gate_name, targets, params_by_batch);
             },
             py::arg("gate_name"),
             py::arg("targets"),
             py::arg("params_by_batch"),
             "Apply one RX/RY/RZ angle per simulator batch state.")
        .def("apply_matrix",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("apply_matrix requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("apply_matrix expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for apply_matrix.");
                 }

                 const std::size_t expected_dim = std::size_t{1} << targets.size();
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("apply_matrix dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 self.apply_matrix(host, targets);
             },
             py::arg("matrix"),
             py::arg("targets"))
        .def("apply_sparse_matrix",
             &apply_sparse_matrix_from_python,
             py::arg("data"),
             py::arg("indices"),
             py::arg("indptr"),
             py::arg("shape"),
             py::arg("targets"),
             "Apply a CSR sparse matrix to target qubits via the simulator sparse-operation hook.")
        .def("apply_controlled_matrix",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& controls,
                const std::vector<unsigned>& targets) {
                 if (controls.empty()) {
                     throw std::invalid_argument("apply_controlled_matrix requires at least one control qubit.");
                 }
                 if (targets.empty()) {
                     throw std::invalid_argument("apply_controlled_matrix requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("apply_controlled_matrix expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for apply_controlled_matrix.");
                 }

                 const std::size_t expected_dim = std::size_t{1} << targets.size();
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("apply_controlled_matrix dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 self.apply_controlled_matrix(host, controls, targets);
             },
             py::arg("matrix"),
             py::arg("controls"),
             py::arg("targets"),
             "Apply a target matrix under one or more all-one control qubits.")
        .def("set_statevector",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> state) {
                 std::vector<std::complex<double>> host(static_cast<std::size_t>(state.size()));
                 std::memcpy(host.data(), state.data(), host.size() * sizeof(std::complex<double>));
                 self.set_statevector(host);
             },
             py::arg("state"),
             "Upload a full host statevector into the simulator state.")
        .def("set_statevectors",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> states) {
                 std::vector<std::complex<double>> host(static_cast<std::size_t>(states.size()));
                 std::memcpy(host.data(), states.data(), host.size() * sizeof(std::complex<double>));
                 self.set_statevectors(host);
             },
             py::arg("states"),
             "Upload all host statevectors for a batched simulator state.")
        .def("get_statevector",
             [](const rocquantum::QuantumSimulator& self, std::size_t batch_index) {
                 auto state = self.get_statevector(batch_index);
                 py::array_t<std::complex<double>> result(state.size());
                 std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(std::complex<double>));
                 return result;
             },
             py::arg("batch_index") = 0)
        .def("get_statevectors",
             [](const rocquantum::QuantumSimulator& self) {
                 auto state = self.get_statevectors();
                 const auto batch = static_cast<py::ssize_t>(self.batch_size());
                 const auto width = static_cast<py::ssize_t>(state.size() / self.batch_size());
                 py::array_t<std::complex<double>> result({
                     batch,
                     width,
                 });
                 std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(std::complex<double>));
                 return result;
             })
        .def("probabilities",
             [](const rocquantum::QuantumSimulator& self, const std::vector<unsigned>& qubits) {
                 auto probabilities = self.probabilities(qubits);
                 py::array_t<double> result(probabilities.size());
                 std::memcpy(result.mutable_data(), probabilities.data(), probabilities.size() * sizeof(double));
                 return result;
             },
             py::arg("qubits"),
             "Return normalized computational-basis probabilities for selected qubits.")
        .def("probabilities_batch",
             [](const rocquantum::QuantumSimulator& self, const std::vector<unsigned>& qubits) {
                 auto probabilities = self.probabilities_batch(qubits);
                 const auto batch = static_cast<py::ssize_t>(self.batch_size());
                 const auto width = static_cast<py::ssize_t>(probabilities.size() / self.batch_size());
                 py::array_t<double> result({batch, width});
                 std::memcpy(result.mutable_data(), probabilities.data(), probabilities.size() * sizeof(double));
                 return result;
             },
             py::arg("qubits"),
             "Return batch-major normalized computational-basis probabilities for selected qubits.")
        .def("measure", &rocquantum::QuantumSimulator::measure, py::arg("qubits"), py::arg("shots"))
        .def("measure_batch",
             [](rocquantum::QuantumSimulator& self, const std::vector<unsigned>& qubits, int shots) {
                 auto samples = self.measure_batch(qubits, shots);
                 const auto batch = static_cast<py::ssize_t>(self.batch_size());
                 const auto width = static_cast<py::ssize_t>(shots);
                 py::array_t<long long> result({batch, width});
                 std::memcpy(result.mutable_data(), samples.data(), samples.size() * sizeof(long long));
                 return result;
             },
             py::arg("qubits"),
             py::arg("shots"),
             "Return batch-major sampled computational-basis outcomes for selected qubits.")
        .def("expectation_value",
             &rocquantum::QuantumSimulator::expectation_value,
             py::arg("pauli"),
             py::arg("target"),
             "Return <P_target> for a single Pauli observable.")
        .def("expectation_pauli_string",
             &rocquantum::QuantumSimulator::expectation_pauli_string,
             py::arg("pauli_string"),
             py::arg("targets"),
             "Return a non-destructive Pauli-string expectation value.")
        .def("expectation_pauli_string_batch",
             [](const rocquantum::QuantumSimulator& self,
                const std::string& pauli_string,
                const std::vector<unsigned>& targets) {
                 auto expectations = self.expectation_pauli_string_batch(pauli_string, targets);
                 py::array_t<double> result(expectations.size());
                 std::memcpy(result.mutable_data(), expectations.data(), expectations.size() * sizeof(double));
                 return result;
             },
             py::arg("pauli_string"),
             py::arg("targets"),
             "Return batch-major Pauli-string expectation values.")
        .def("expectation_matrix",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("expectation_matrix requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("expectation_matrix expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for expectation_matrix.");
                 }

                 const std::size_t expected_dim = std::size_t{1} << targets.size();
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("expectation_matrix dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 return self.expectation_matrix(host, targets);
             },
             py::arg("matrix"),
             py::arg("targets"),
             "Return <M_targets> for a dense matrix observable.")
        .def("expectation_matrix_batch",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("expectation_matrix_batch requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("expectation_matrix_batch expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for expectation_matrix_batch.");
                 }

                 const std::size_t expected_dim = std::size_t{1} << targets.size();
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("expectation_matrix_batch dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 auto expectations = self.expectation_matrix_batch(host, targets);
                 py::array_t<std::complex<double>> result(expectations.size());
                 std::memcpy(result.mutable_data(),
                             expectations.data(),
                             expectations.size() * sizeof(std::complex<double>));
                 return result;
             },
             py::arg("matrix"),
             py::arg("targets"),
             "Return batch-major <M_targets> values for a dense matrix observable.")
        .def("sparse_hamiltonian_moments",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data,
                py::array_t<long long, py::array::c_style | py::array::forcecast> indices,
                py::array_t<long long, py::array::c_style | py::array::forcecast> indptr,
                const std::pair<std::size_t, std::size_t>& shape) {
                 std::vector<std::complex<double>> host_data(static_cast<std::size_t>(data.size()));
                 std::vector<std::size_t> host_indices(static_cast<std::size_t>(indices.size()));
                 std::vector<std::size_t> host_indptr(static_cast<std::size_t>(indptr.size()));
                 std::memcpy(host_data.data(), data.data(), host_data.size() * sizeof(std::complex<double>));
                 for (std::size_t idx = 0; idx < host_indices.size(); ++idx) {
                     if (indices.data()[idx] < 0) {
                         throw std::invalid_argument("Sparse Hamiltonian CSR indices must be non-negative.");
                     }
                     host_indices[idx] = static_cast<std::size_t>(indices.data()[idx]);
                 }
                 for (std::size_t idx = 0; idx < host_indptr.size(); ++idx) {
                     if (indptr.data()[idx] < 0) {
                         throw std::invalid_argument("Sparse Hamiltonian CSR indptr must be non-negative.");
                     }
                     host_indptr[idx] = static_cast<std::size_t>(indptr.data()[idx]);
                 }
                 return self.sparse_hamiltonian_moments(
                     host_data,
                     host_indices,
                     host_indptr,
                     shape.first,
                     shape.second);
             },
             py::arg("data"),
             py::arg("indices"),
             py::arg("indptr"),
             py::arg("shape"),
             "Return (<H>, <H^2>) from a CSR sparse Hamiltonian without densifying it.")
        .def("sparse_hamiltonian_moments_batch",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data,
                py::array_t<long long, py::array::c_style | py::array::forcecast> indices,
                py::array_t<long long, py::array::c_style | py::array::forcecast> indptr,
                const std::pair<std::size_t, std::size_t>& shape) {
                 std::vector<std::complex<double>> host_data(static_cast<std::size_t>(data.size()));
                 std::vector<std::size_t> host_indices(static_cast<std::size_t>(indices.size()));
                 std::vector<std::size_t> host_indptr(static_cast<std::size_t>(indptr.size()));
                 std::memcpy(host_data.data(), data.data(), host_data.size() * sizeof(std::complex<double>));
                 for (std::size_t idx = 0; idx < host_indices.size(); ++idx) {
                     if (indices.data()[idx] < 0) {
                         throw std::invalid_argument("Sparse Hamiltonian CSR indices must be non-negative.");
                     }
                     host_indices[idx] = static_cast<std::size_t>(indices.data()[idx]);
                 }
                 for (std::size_t idx = 0; idx < host_indptr.size(); ++idx) {
                     if (indptr.data()[idx] < 0) {
                         throw std::invalid_argument("Sparse Hamiltonian CSR indptr must be non-negative.");
                     }
                     host_indptr[idx] = static_cast<std::size_t>(indptr.data()[idx]);
                 }
                 auto moments = self.sparse_hamiltonian_moments_batch(
                     host_data,
                     host_indices,
                     host_indptr,
                     shape.first,
                     shape.second);
                 py::array_t<std::complex<double>> means(moments.first.size());
                 py::array_t<std::complex<double>> second_moments(moments.second.size());
                 std::memcpy(means.mutable_data(),
                             moments.first.data(),
                             moments.first.size() * sizeof(std::complex<double>));
                 std::memcpy(second_moments.mutable_data(),
                             moments.second.data(),
                             moments.second.size() * sizeof(std::complex<double>));
                 return py::make_tuple(means, second_moments);
             },
             py::arg("data"),
             py::arg("indices"),
             py::arg("indptr"),
             py::arg("shape"),
             "Return batch-major (<H>, <H^2>) arrays from a CSR sparse Hamiltonian without densifying it.")
        .def("num_qubits", &rocquantum::QuantumSimulator::num_qubits)
        .def("batch_size", &rocquantum::QuantumSimulator::batch_size)
        // Legacy-style bindings
        .def("ApplyGate",
             [](rocquantum::QuantumSimulator& self, const std::string& gate_name, int target_qubit) {
                 self.ApplyGate(gate_name, target_qubit);
             },
             py::arg("gate_name"),
             py::arg("target_qubit"))
        .def("ApplyGate",
             [](rocquantum::QuantumSimulator& self, const std::string& gate_name, int control_qubit, int target_qubit) {
                 self.ApplyGate(gate_name, control_qubit, target_qubit);
             },
             py::arg("gate_name"),
             py::arg("control_qubit"),
             py::arg("target_qubit"))
        .def("ApplyGate",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                int target_qubit) {
                 if (matrix.ndim() != 2 || matrix.shape(0) != 2 || matrix.shape(1) != 2) {
                     throw std::invalid_argument("ApplyGate expects a 2x2 complex matrix.");
                 }
                 std::vector<std::complex<double>> host(4);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 self.ApplyGate(host, target_qubit);
             },
             py::arg("gate_matrix"),
             py::arg("target_qubit"))
        .def("ApplyGateBatch",
             [](rocquantum::QuantumSimulator& self,
                const std::string& gate_name,
                const std::vector<unsigned>& targets,
                const std::vector<double>& params_by_batch) {
                 self.ApplyGateBatch(gate_name, targets, params_by_batch);
             },
             py::arg("gate_name"),
             py::arg("targets"),
             py::arg("params_by_batch"))
        .def("ApplyControlledGate",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                int control_qubit,
                int target_qubit) {
                 if (matrix.ndim() != 2 || matrix.shape(0) != 2 || matrix.shape(1) != 2) {
                     throw std::invalid_argument("ApplyControlledGate expects a 2x2 complex matrix.");
                 }
                 std::vector<std::complex<double>> host(4);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 self.ApplyControlledGate(host, control_qubit, target_qubit);
             },
             py::arg("gate_matrix"),
             py::arg("control_qubit"),
             py::arg("target_qubit"))
        .def("Execute", &rocquantum::QuantumSimulator::Execute)
        .def("MeasureQubit", &rocquantum::QuantumSimulator::MeasureQubit, py::arg("target_qubit"))
        .def("ResetQubit", &rocquantum::QuantumSimulator::ResetQubit, py::arg("target_qubit"))
        .def("GetStateVector",
             [](const rocquantum::QuantumSimulator& self) {
                 auto state = self.GetStateVector();
                 py::array_t<std::complex<double>> result(state.size());
                 std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(std::complex<double>));
                 return result;
             })
        .def("GetStateVectors",
             [](const rocquantum::QuantumSimulator& self) {
                 auto state = self.GetStateVectors();
                 const auto batch = static_cast<py::ssize_t>(self.batch_size());
                 const auto width = static_cast<py::ssize_t>(state.size() / self.batch_size());
                 py::array_t<std::complex<double>> result({
                     batch,
                     width,
                 });
                 std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(std::complex<double>));
                 return result;
             })
        .def("Probabilities",
             [](const rocquantum::QuantumSimulator& self, const std::vector<unsigned>& qubits) {
                 auto probabilities = self.Probabilities(qubits);
                 py::array_t<double> result(probabilities.size());
                 std::memcpy(result.mutable_data(), probabilities.data(), probabilities.size() * sizeof(double));
                 return result;
             },
             py::arg("qubits"))
        .def("ProbabilitiesBatch",
             [](const rocquantum::QuantumSimulator& self, const std::vector<unsigned>& qubits) {
                 auto probabilities = self.ProbabilitiesBatch(qubits);
                 const auto batch = static_cast<py::ssize_t>(self.batch_size());
                 const auto width = static_cast<py::ssize_t>(probabilities.size() / self.batch_size());
                 py::array_t<double> result({batch, width});
                 std::memcpy(result.mutable_data(), probabilities.data(), probabilities.size() * sizeof(double));
                 return result;
             },
             py::arg("qubits"))
        .def("MeasureBatch",
             [](rocquantum::QuantumSimulator& self, const std::vector<unsigned>& qubits, int shots) {
                 auto samples = self.MeasureBatch(qubits, shots);
                 const auto batch = static_cast<py::ssize_t>(self.batch_size());
                 const auto width = static_cast<py::ssize_t>(shots);
                 py::array_t<long long> result({batch, width});
                 std::memcpy(result.mutable_data(), samples.data(), samples.size() * sizeof(long long));
                 return result;
             },
             py::arg("qubits"),
             py::arg("shots"))
        .def("GetExpectationValue",
             &rocquantum::QuantumSimulator::GetExpectationValue,
             py::arg("pauli"),
             py::arg("target_qubit"))
        .def("GetExpectationPauliString",
             &rocquantum::QuantumSimulator::GetExpectationPauliString,
             py::arg("pauli_string"),
             py::arg("targets"))
        .def("GetExpectationPauliStringBatch",
             [](const rocquantum::QuantumSimulator& self,
                const std::string& pauli_string,
                const std::vector<unsigned>& targets) {
                 auto expectations = self.GetExpectationPauliStringBatch(pauli_string, targets);
                 py::array_t<double> result(expectations.size());
                 std::memcpy(result.mutable_data(), expectations.data(), expectations.size() * sizeof(double));
                 return result;
             },
             py::arg("pauli_string"),
             py::arg("targets"))
        .def("ExpectationMatrix",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("ExpectationMatrix requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("ExpectationMatrix expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for ExpectationMatrix.");
                 }

                 const std::size_t expected_dim = std::size_t{1} << targets.size();
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("ExpectationMatrix dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 return self.ExpectationMatrix(host, targets);
             },
             py::arg("matrix"),
             py::arg("targets"))
        .def("ExpectationMatrixBatch",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("ExpectationMatrixBatch requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("ExpectationMatrixBatch expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for ExpectationMatrixBatch.");
                 }

                 const std::size_t expected_dim = std::size_t{1} << targets.size();
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("ExpectationMatrixBatch dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 auto expectations = self.ExpectationMatrixBatch(host, targets);
                 py::array_t<std::complex<double>> result(expectations.size());
                 std::memcpy(result.mutable_data(),
                             expectations.data(),
                             expectations.size() * sizeof(std::complex<double>));
                 return result;
             },
             py::arg("matrix"),
             py::arg("targets"))
        .def("SparseHamiltonianMoments",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data,
                py::array_t<long long, py::array::c_style | py::array::forcecast> indices,
                py::array_t<long long, py::array::c_style | py::array::forcecast> indptr,
                const std::pair<std::size_t, std::size_t>& shape) {
                 std::vector<std::complex<double>> host_data(static_cast<std::size_t>(data.size()));
                 std::vector<std::size_t> host_indices(static_cast<std::size_t>(indices.size()));
                 std::vector<std::size_t> host_indptr(static_cast<std::size_t>(indptr.size()));
                 std::memcpy(host_data.data(), data.data(), host_data.size() * sizeof(std::complex<double>));
                 for (std::size_t idx = 0; idx < host_indices.size(); ++idx) {
                     if (indices.data()[idx] < 0) {
                         throw std::invalid_argument("Sparse Hamiltonian CSR indices must be non-negative.");
                     }
                     host_indices[idx] = static_cast<std::size_t>(indices.data()[idx]);
                 }
                 for (std::size_t idx = 0; idx < host_indptr.size(); ++idx) {
                     if (indptr.data()[idx] < 0) {
                         throw std::invalid_argument("Sparse Hamiltonian CSR indptr must be non-negative.");
                     }
                     host_indptr[idx] = static_cast<std::size_t>(indptr.data()[idx]);
                 }
                 return self.SparseHamiltonianMoments(
                     host_data,
                     host_indices,
                     host_indptr,
                     shape.first,
                     shape.second);
             },
             py::arg("data"),
             py::arg("indices"),
             py::arg("indptr"),
             py::arg("shape"))
        .def("ApplySparseMatrix",
             &apply_sparse_matrix_from_python,
             py::arg("data"),
             py::arg("indices"),
             py::arg("indptr"),
             py::arg("shape"),
             py::arg("targets"))
        .def("SparseHamiltonianMomentsBatch",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data,
                py::array_t<long long, py::array::c_style | py::array::forcecast> indices,
                py::array_t<long long, py::array::c_style | py::array::forcecast> indptr,
                const std::pair<std::size_t, std::size_t>& shape) {
                 std::vector<std::complex<double>> host_data(static_cast<std::size_t>(data.size()));
                 std::vector<std::size_t> host_indices(static_cast<std::size_t>(indices.size()));
                 std::vector<std::size_t> host_indptr(static_cast<std::size_t>(indptr.size()));
                 std::memcpy(host_data.data(), data.data(), host_data.size() * sizeof(std::complex<double>));
                 for (std::size_t idx = 0; idx < host_indices.size(); ++idx) {
                     if (indices.data()[idx] < 0) {
                         throw std::invalid_argument("Sparse Hamiltonian CSR indices must be non-negative.");
                     }
                     host_indices[idx] = static_cast<std::size_t>(indices.data()[idx]);
                 }
                 for (std::size_t idx = 0; idx < host_indptr.size(); ++idx) {
                     if (indptr.data()[idx] < 0) {
                         throw std::invalid_argument("Sparse Hamiltonian CSR indptr must be non-negative.");
                     }
                     host_indptr[idx] = static_cast<std::size_t>(indptr.data()[idx]);
                 }
                 auto moments = self.SparseHamiltonianMomentsBatch(
                     host_data,
                     host_indices,
                     host_indptr,
                     shape.first,
                     shape.second);
                 py::array_t<std::complex<double>> means(moments.first.size());
                 py::array_t<std::complex<double>> second_moments(moments.second.size());
                 std::memcpy(means.mutable_data(),
                             moments.first.data(),
                             moments.first.size() * sizeof(std::complex<double>));
                 std::memcpy(second_moments.mutable_data(),
                             moments.second.data(),
                             moments.second.size() * sizeof(std::complex<double>));
                 return py::make_tuple(means, second_moments);
             },
             py::arg("data"),
             py::arg("indices"),
             py::arg("indptr"),
             py::arg("shape"));

    // Historical alias for integrations/tests still referencing QSim
    m.attr("QSim") = m.attr("QuantumSimulator");
}
