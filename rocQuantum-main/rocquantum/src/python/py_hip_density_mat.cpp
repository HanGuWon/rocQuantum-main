#include <pybind11/pybind11>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <complex>
#include <cstdint>
#include <vector>
#include <stdexcept>

#include "hipDensityMat.hpp"
#include "../hipDensityMat/hipDensityMat_internal.hpp"

namespace py = pybind11;

// Helper to check HIP API calls for errors.
#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        throw std::runtime_error("HIP error (" + std::to_string(err) + "): " + hipGetErrorString(err)); \
    } \
} while (0)

// Helper to check hipDensityMat API calls for errors.
#define HIPDENSITYMAT_CHECK(cmd) do { \
    hipDensityMatStatus_t status = cmd; \
    if (status != HIPDENSITYMAT_STATUS_SUCCESS) { \
        throw std::runtime_error("hipDensityMat error: " + std::to_string(status)); \
    } \
} while (0)


PYBIND11_MODULE(rocq_hip, m) {
    m.doc() = "Python bindings for the rocQuantum hipDensityMat library";

    py::class_<hipDensityMatState>(m, "DensityMatrixState")
        .def(py::init([](int num_qubits) {
            hipDensityMatState_t state_handle;
            HIPDENSITYMAT_CHECK(hipDensityMatCreateState(&state_handle, num_qubits));
            // We cast the opaque pointer to the internal struct pointer for pybind11
            return static_cast<hipDensityMatState*>(state_handle);
        }), py::arg("num_qubits"), "Create a new density matrix state.")
        .def("__del__", [](hipDensityMatState* state) {
            // The handle is the same as the pointer to the internal struct
            hipDensityMatDestroyState(state);
        })
        .def("apply_gate", [](hipDensityMatState* state, py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> gate_matrix, int target_qubit, bool adjoint) {
            if (gate_matrix.ndim() != 2 || gate_matrix.shape(0) != 2 || gate_matrix.shape(1) != 2) {
                throw std::invalid_argument("Gate matrix must be a 2x2 NumPy array.");
            }
            
            auto mat_unchecked = gate_matrix.unchecked<2>();
            hipComplex gate_host[4];
            gate_host[0] = reinterpret_cast<hipComplex(&)>(mat_unchecked(0, 0));
            gate_host[1] = reinterpret_cast<hipComplex(&)>(mat_unchecked(0, 1));
            gate_host[2] = reinterpret_cast<hipComplex(&)>(mat_unchecked(1, 0));
            gate_host[3] = reinterpret_cast<hipComplex(&)>(mat_unchecked(1, 1));

            if (adjoint) {
                // Conjugate transpose
                gate_host[1] = hipConjf(gate_host[1]);
                gate_host[2] = hipConjf(gate_host[2]);
                std::swap(gate_host[1], gate_host[2]);
            }

            HIPDENSITYMAT_CHECK(hipDensityMatApplyGate(state, target_qubit, gate_host));
        }, py::arg("gate_matrix"), py::arg("target_qubit"), py::arg("adjoint") = false, "Apply a single-qubit gate. If adjoint is true, applies the conjugate transpose.")
        .def("apply_cnot", [](hipDensityMatState* state, int control_qubit, int target_qubit) {
            HIPDENSITYMAT_CHECK(hipDensityMatApplyCNOT(state, control_qubit, target_qubit));
        }, py::arg("control_qubit"), py::arg("target_qubit"), "Apply a CNOT gate.")
        .def("apply_controlled_gate", [](hipDensityMatState* state, py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> gate_matrix, int control_qubit, int target_qubit) {
            if (gate_matrix.ndim() != 2 || gate_matrix.shape(0) != 2 || gate_matrix.shape(1) != 2) {
                throw std::invalid_argument("Gate matrix must be a 2x2 NumPy array.");
            }
            
            hipComplex* gate_matrix_device = nullptr;
            HIP_CHECK(hipMalloc(&gate_matrix_device, 4 * sizeof(hipComplex)));
            HIP_CHECK(hipMemcpy(gate_matrix_device, gate_matrix.data(), 4 * sizeof(hipComplex), hipMemcpyHostToDevice));
            
            hipDensityMatStatus_t status = hipDensityMatApplyControlledGate(state, control_qubit, target_qubit, gate_matrix_device);
            
            HIP_CHECK(hipFree(gate_matrix_device));
            HIPDENSITYMAT_CHECK(status);
        }, py::arg("gate_matrix"), py::arg("control_qubit"), py::arg("target_qubit"), "Apply a controlled single-qubit gate.")
        .def("compute_expectation", [](hipDensityMatState* state, hipDensityMatPauli_t pauli_op, int target_qubit) {
            double result;
            HIPDENSITYMAT_CHECK(hipDensityMatComputeExpectation(state, target_qubit, pauli_op, &result));
            return result;
        }, py::arg("pauli_op"), py::arg("target_qubit"), "Compute the expectation value of a single-qubit Pauli observable.")
        .def("_compute_z_product_expectation", [](hipDensityMatState* state, std::vector<int> z_qubit_indices) {
            double result;
            HIPDENSITYMAT_CHECK(hipDensityMatComputePauliZProductExpectation(state, z_qubit_indices.size(), z_qubit_indices.data(), &result));
            return result;
        }, py::arg("z_qubit_indices"), "Compute expectation of a Pauli Z product.")
        .def("apply_bit_flip_channel", [](hipDensityMatState* state, int target_qubit, double probability) {
            HIPDENSITYMAT_CHECK(hipDensityMatApplyBitFlipChannel(state, target_qubit, probability));
        }, py::arg("target_qubit"), py::arg("probability"), "Apply a bit-flip noise channel.")
        .def("apply_phase_flip_channel", [](hipDensityMatState* state, int target_qubit, double probability) {
            HIPDENSITYMAT_CHECK(hipDensityMatApplyPhaseFlipChannel(state, target_qubit, probability));
        }, py::arg("target_qubit"), py::arg("probability"), "Apply a phase-flip noise channel.")
        .def("apply_depolarizing_channel", [](hipDensityMatState* state, int target_qubit, double probability) {
            HIPDENSITYMAT_CHECK(hipDensityMatApplyDepolarizingChannel(state, target_qubit, probability));
        }, py::arg("target_qubit"), py::arg("probability"), "Apply a depolarizing noise channel.")
        .def("apply_amplitude_damping_channel", [](hipDensityMatState* state, int target_qubit, double gamma) {
            HIPDENSITYMAT_CHECK(hipDensityMatApplyAmplitudeDampingChannel(state, target_qubit, gamma));
        }, py::arg("target_qubit"), py::arg("gamma"), "Apply an amplitude damping noise channel.")
        .def("apply_channel", [](hipDensityMatState* state, py::object target_qubit, py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> kraus_matrices) {
            std::vector<int> targets;
            if (py::isinstance<py::int_>(target_qubit)) {
                targets.push_back(target_qubit.cast<int>());
            } else {
                targets = target_qubit.cast<std::vector<int>>();
            }
            if (targets.empty()) {
                throw std::invalid_argument("Kraus channel must target at least one qubit.");
            }
            if (targets.size() > 20) {
                throw std::invalid_argument("Kraus channel target count is too large.");
            }

            const int matrix_dim = 1 << static_cast<int>(targets.size());
            if (kraus_matrices.ndim() != 3 ||
                kraus_matrices.shape(1) != matrix_dim ||
                kraus_matrices.shape(2) != matrix_dim) {
                throw std::invalid_argument(
                    "Kraus matrices must have shape (num_kraus, 2**len(target_qubits), 2**len(target_qubits)).");
            }
            const int num_kraus = static_cast<int>(kraus_matrices.shape(0));
            if (num_kraus <= 0) {
                throw std::invalid_argument("Kraus channel must include at least one matrix.");
            }

            auto matrices = kraus_matrices.unchecked<3>();
            const size_t matrix_elements = static_cast<size_t>(matrix_dim) * static_cast<size_t>(matrix_dim);
            std::vector<hipComplex> kraus_host(static_cast<size_t>(num_kraus) * matrix_elements);
            for (int k = 0; k < num_kraus; ++k) {
                for (int row = 0; row < matrix_dim; ++row) {
                    for (int col = 0; col < matrix_dim; ++col) {
                        const std::complex<float> value = matrices(k, row, col);
                        kraus_host[static_cast<size_t>(k) * matrix_elements + row * matrix_dim + col] =
                            make_hipFloatComplex(value.real(), value.imag());
                    }
                }
            }

            hipDensityMatChannel_t channel{
                num_kraus,
                kraus_host.data(),
                static_cast<int>(targets.size()),
                targets.data()};
            HIPDENSITYMAT_CHECK(hipDensityMatApplyChannel(state, targets[0], &channel));
        }, py::arg("target_qubit"), py::arg("kraus_matrices"), "Apply a generic Kraus channel to one or more qubits.")
        .def("sample", [](hipDensityMatState* state, std::vector<int> measured_qubits, int num_shots) {
            if (num_shots < 0) {
                throw std::invalid_argument("num_shots must be non-negative.");
            }
            py::array_t<uint64_t> results(static_cast<py::ssize_t>(num_shots));
            HIPDENSITYMAT_CHECK(hipDensityMatSample(
                state,
                measured_qubits.data(),
                static_cast<int>(measured_qubits.size()),
                num_shots,
                results.mutable_data()));
            return results;
        }, py::arg("measured_qubits"), py::arg("num_shots"), "Sample computational-basis outcomes.")
        .def("get_density_matrix", [](hipDensityMatState* state) {
            if (state == nullptr) {
                throw std::invalid_argument("DensityMatrixState is null.");
            }

            auto* internal_state = static_cast<hipDensityMatState*>(state);
            const int64_t dim = 1LL << internal_state->num_qubits_;
            py::array_t<std::complex<float>> host_density({dim, dim});
            HIP_CHECK(hipMemcpy(
                host_density.mutable_data(),
                internal_state->device_data_,
                static_cast<size_t>(internal_state->num_elements_) * sizeof(hipComplex),
                hipMemcpyDeviceToHost
            ));
            return host_density;
        }, "Copy the full density matrix back to host memory.");

    py::enum_<hipDensityMatPauli_t>(m, "Pauli")
        .value("X", HIPDENSITYMAT_PAULI_X)
        .value("Y", HIPDENSITYMAT_PAULI_Y)
        .value("Z", HIPDENSITYMAT_PAULI_Z)
        .export_values();
}
