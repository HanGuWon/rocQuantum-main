#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include "rocqCompiler/MLIRCompiler.h"
#include "rocqCompiler/QuantumBackend.h"
#include "rocquantum/QuantumSimulator.h"

namespace py = pybind11;

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
             "Executes the supported MLIR subset (qalloc, H/X/Y/Z, CNOT, RX/RY/RZ) "
             "through the selected backend and returns the final state vector. "
             "Unsupported ops raise actionable diagnostics.")
        .def("emit_qir", &rocq::MLIRCompiler::emit_qir,
             "Compiles the MLIR string down to QIR (LLVM IR).");

    auto simulator = py::class_<rocquantum::QuantumSimulator>(m, "QuantumSimulator");

    simulator
        .def(py::init<unsigned>(), py::arg("num_qubits"))
        .def("reset", &rocquantum::QuantumSimulator::reset)
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
        .def("set_statevector",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> state) {
                 std::vector<std::complex<double>> host(static_cast<std::size_t>(state.size()));
                 std::memcpy(host.data(), state.data(), host.size() * sizeof(std::complex<double>));
                 self.set_statevector(host);
             },
             py::arg("state"),
             "Upload a full host statevector into the simulator state.")
        .def("get_statevector",
             [](const rocquantum::QuantumSimulator& self) {
                 auto state = self.get_statevector();
                 py::array_t<std::complex<double>> result(state.size());
                 std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(std::complex<double>));
                 return result;
             })
        .def("measure", &rocquantum::QuantumSimulator::measure, py::arg("qubits"), py::arg("shots"))
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
        .def("num_qubits", &rocquantum::QuantumSimulator::num_qubits)
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
        .def("Execute", &rocquantum::QuantumSimulator::Execute)
        .def("ResetQubit", &rocquantum::QuantumSimulator::ResetQubit, py::arg("target_qubit"))
        .def("GetStateVector",
             [](const rocquantum::QuantumSimulator& self) {
                 auto state = self.GetStateVector();
                 py::array_t<std::complex<double>> result(state.size());
                 std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(std::complex<double>));
                 return result;
             })
        .def("GetExpectationValue",
             &rocquantum::QuantumSimulator::GetExpectationValue,
             py::arg("pauli"),
             py::arg("target_qubit"))
        .def("GetExpectationPauliString",
             &rocquantum::QuantumSimulator::GetExpectationPauliString,
             py::arg("pauli_string"),
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
             py::arg("shape"));

    // Historical alias for integrations/tests still referencing QSim
    m.attr("QSim") = m.attr("QuantumSimulator");
}
