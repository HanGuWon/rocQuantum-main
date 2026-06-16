#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cctype>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef ROCQUANTUM_ENABLE_MLIR_COMPILER
#include "rocqCompiler/MLIRCompiler.h"
#include "rocqCompiler/QuantumBackend.h"
#endif
#include "rocquantum/QuantumSimulator.h"

namespace py = pybind11;

namespace {

constexpr const char* kDisabledMlirCompilerMessage =
    "MLIR compiler support is disabled in this rocquantum_bind build. "
    "The default ROCm runtime binding exposes QuantumSimulator and framework adapters "
    "without linking the experimental rocqCompiler MLIR stack.";

#ifndef ROCQUANTUM_ENABLE_MLIR_COMPILER
class DisabledRuntimeMLIRCompiler {
public:
    DisabledRuntimeMLIRCompiler(unsigned num_qubits, std::string backend_name)
        : num_qubits_(num_qubits), backend_name_(std::move(backend_name)) {}

    std::vector<std::complex<double>> compile_and_execute(const std::string& mlir) const {
        (void)mlir;
        throw std::runtime_error(kDisabledMlirCompilerMessage);
    }

    std::string emit_qir(const std::string& mlir) const {
        (void)mlir;
        throw std::runtime_error(kDisabledMlirCompilerMessage);
    }

    unsigned num_qubits() const {
        return num_qubits_;
    }

    const std::string& backend_name() const {
        return backend_name_;
    }

private:
    unsigned num_qubits_;
    std::string backend_name_;
};
#endif

struct AdjointOperationPayload {
    std::string name;
    std::string rocq_name;
    std::vector<unsigned> wires;
    std::vector<double> params;
    std::vector<int> param_indices;
    std::vector<double> param_derivative_scales;
    std::vector<int> trainable_param_indices;
    std::vector<int> trainable_param_positions;
    bool has_matrix = false;
    std::vector<std::complex<double>> matrix;
    std::vector<unsigned> controls;
    std::vector<bool> control_values;
    bool has_sparse_matrix = false;
    std::vector<std::complex<double>> sparse_data;
    std::vector<std::size_t> sparse_indices;
    std::vector<std::size_t> sparse_indptr;
    std::size_t sparse_rows = 0;
    std::size_t sparse_cols = 0;
};

struct PauliTermPayload {
    std::complex<double> coefficient;
    std::string pauli_string;
    std::vector<unsigned> targets;
};

struct DenseMatrixPayload {
    std::vector<std::complex<double>> matrix;
    std::vector<unsigned> targets;
};

struct SparseMatrixPayload {
    std::vector<std::complex<double>> data;
    std::vector<std::size_t> indices;
    std::vector<std::size_t> indptr;
    std::vector<unsigned> targets;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

struct ObservablePayload {
    std::vector<PauliTermPayload> pauli_terms;
    std::vector<DenseMatrixPayload> dense_terms;
    std::vector<SparseMatrixPayload> sparse_terms;
};

std::size_t checked_power_of_two(std::size_t exponent, const std::string& label) {
    if (exponent >= static_cast<std::size_t>(std::numeric_limits<std::size_t>::digits)) {
        throw std::invalid_argument(label + " exceeds the supported host bit width.");
    }
    return std::size_t{1} << exponent;
}

std::size_t checked_bit_mask(unsigned bit, const std::string& label) {
    return checked_power_of_two(static_cast<std::size_t>(bit), label);
}

std::size_t checked_square_size(std::size_t dimension, const std::string& label) {
    if (dimension != 0 && dimension > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::invalid_argument(label + " is too large for host-side binding validation.");
    }
    return dimension * dimension;
}

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

std::string uppercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return value;
}

py::object required_dict_item(py::dict dict, const char* key, const char* context) {
    const py::str py_key(key);
    if (!dict.contains(py_key)) {
        throw std::invalid_argument(std::string(context) + " payload is missing '" + key + "'.");
    }
    return py::reinterpret_borrow<py::object>(dict[py_key]);
}

bool dict_contains(py::dict dict, const char* key) {
    return dict.contains(py::str(key));
}

[[noreturn]] void throw_adjoint_not_implemented(const std::string& message) {
    const std::string full_message = "rocQuantum status 5: " + message;
    PyErr_SetString(PyExc_NotImplementedError, full_message.c_str());
    throw py::error_already_set();
}

std::vector<int> trainable_intersection(
    const std::vector<int>& param_indices,
    const std::unordered_map<int, std::size_t>& trainable_columns) {
    std::vector<int> out;
    for (int param_index : param_indices) {
        if (trainable_columns.find(param_index) != trainable_columns.end()) {
            out.push_back(param_index);
        }
    }
    return out;
}

std::vector<int> trainable_positions(
    const std::vector<int>& param_indices,
    const std::unordered_map<int, std::size_t>& trainable_columns) {
    std::vector<int> out;
    for (std::size_t idx = 0; idx < param_indices.size(); ++idx) {
        if (trainable_columns.find(param_indices[idx]) != trainable_columns.end()) {
            out.push_back(static_cast<int>(idx));
        }
    }
    return out;
}

std::vector<std::complex<double>> parse_complex_matrix_payload(py::handle values);
std::vector<std::complex<double>> parse_complex_vector_payload(py::handle values);
std::vector<std::size_t> parse_size_vector(py::handle values, const char* label);

std::vector<AdjointOperationPayload> parse_adjoint_operations(
    py::sequence operations,
    const std::unordered_map<int, std::size_t>& trainable_columns) {
    std::vector<AdjointOperationPayload> out;
    for (py::handle item : operations) {
        py::dict op = py::reinterpret_borrow<py::dict>(item);
        AdjointOperationPayload payload;
        payload.name = required_dict_item(op, "name", "adjoint operation").cast<std::string>();
        payload.rocq_name = required_dict_item(op, "rocq_name", "adjoint operation").cast<std::string>();
        payload.wires = required_dict_item(op, "wires", "adjoint operation").cast<std::vector<unsigned>>();
        payload.params = required_dict_item(op, "params", "adjoint operation").cast<std::vector<double>>();
        payload.param_indices =
            required_dict_item(op, "param_indices", "adjoint operation").cast<std::vector<int>>();
        if (dict_contains(op, "param_derivative_scales")) {
            payload.param_derivative_scales =
                py::reinterpret_borrow<py::object>(op[py::str("param_derivative_scales")])
                    .cast<std::vector<double>>();
        } else {
            payload.param_derivative_scales.assign(payload.param_indices.size(), 1.0);
        }
        if (payload.param_derivative_scales.size() != payload.param_indices.size()) {
            throw std::invalid_argument(
                "adjoint operation param_derivative_scales and param_indices lengths differ.");
        }

        if (dict_contains(op, "trainable_param_indices")) {
            payload.trainable_param_indices =
                py::reinterpret_borrow<py::object>(op[py::str("trainable_param_indices")])
                    .cast<std::vector<int>>();
        } else {
            payload.trainable_param_indices = trainable_intersection(payload.param_indices, trainable_columns);
        }

        if (dict_contains(op, "trainable_param_positions")) {
            payload.trainable_param_positions =
                py::reinterpret_borrow<py::object>(op[py::str("trainable_param_positions")])
                    .cast<std::vector<int>>();
        } else {
            payload.trainable_param_positions = trainable_positions(payload.param_indices, trainable_columns);
        }

        if (payload.trainable_param_indices.size() != payload.trainable_param_positions.size()) {
            throw std::invalid_argument(
                "adjoint operation trainable_param_indices and trainable_param_positions lengths differ.");
        }
        if (dict_contains(op, "matrix")) {
            payload.has_matrix = true;
            payload.matrix =
                parse_complex_matrix_payload(py::reinterpret_borrow<py::object>(op[py::str("matrix")]));
            if (dict_contains(op, "controls")) {
                payload.controls =
                    py::reinterpret_borrow<py::object>(op[py::str("controls")]).cast<std::vector<unsigned>>();
                if (dict_contains(op, "control_values")) {
                    payload.control_values =
                        py::reinterpret_borrow<py::object>(op[py::str("control_values")]).cast<std::vector<bool>>();
                } else {
                    payload.control_values.assign(payload.controls.size(), true);
                }
                if (payload.control_values.size() != payload.controls.size()) {
                    throw std::invalid_argument(
                        "Matrix adjoint operation control_values length must match controls length.");
                }
            }
            if (!payload.trainable_param_positions.empty()) {
                throw_adjoint_not_implemented(
                    "binding adjoint_jacobian does not differentiate trainable matrix operation payloads.");
            }
        }
        if (dict_contains(op, "sparse_data")) {
            payload.has_sparse_matrix = true;
            payload.sparse_data =
                parse_complex_vector_payload(required_dict_item(op, "sparse_data", "adjoint operation"));
            payload.sparse_indices =
                parse_size_vector(required_dict_item(op, "sparse_indices", "adjoint operation"),
                                  "adjoint sparse operation CSR indices");
            payload.sparse_indptr =
                parse_size_vector(required_dict_item(op, "sparse_indptr", "adjoint operation"),
                                  "adjoint sparse operation CSR indptr");
            auto shape = required_dict_item(op, "sparse_shape", "adjoint operation")
                             .cast<std::pair<std::size_t, std::size_t>>();
            payload.sparse_rows = shape.first;
            payload.sparse_cols = shape.second;
            if (!payload.trainable_param_positions.empty()) {
                throw_adjoint_not_implemented(
                    "binding adjoint_jacobian does not differentiate trainable sparse matrix operation payloads.");
            }
        }
        if (payload.has_matrix && payload.has_sparse_matrix) {
            throw std::invalid_argument("adjoint operation cannot contain both dense and sparse matrix payloads.");
        }
        out.push_back(std::move(payload));
    }
    return out;
}

std::complex<double> parse_complex_pair(py::handle value) {
    const auto pair = py::reinterpret_borrow<py::object>(value).cast<std::pair<double, double>>();
    return {pair.first, pair.second};
}

std::vector<std::complex<double>> parse_complex_vector_payload(py::handle values) {
    std::vector<std::complex<double>> out;
    for (py::handle item : py::reinterpret_borrow<py::sequence>(values)) {
        out.push_back(parse_complex_pair(item));
    }
    return out;
}

std::vector<std::complex<double>> parse_complex_matrix_payload(py::handle values) {
    std::vector<std::complex<double>> out;
    for (py::handle row_item : py::reinterpret_borrow<py::sequence>(values)) {
        for (py::handle item : py::reinterpret_borrow<py::sequence>(row_item)) {
            out.push_back(parse_complex_pair(item));
        }
    }
    return out;
}

std::vector<std::size_t> parse_size_vector(py::handle values, const char* label) {
    std::vector<std::size_t> out;
    for (py::handle item : py::reinterpret_borrow<py::sequence>(values)) {
        const long long value = py::reinterpret_borrow<py::object>(item).cast<long long>();
        if (value < 0) {
            throw std::invalid_argument(std::string(label) + " must be non-negative.");
        }
        out.push_back(static_cast<std::size_t>(value));
    }
    return out;
}

std::vector<ObservablePayload> parse_adjoint_observables(py::sequence observables) {
    std::vector<ObservablePayload> out;
    for (py::handle observable_item : observables) {
        py::sequence observable_terms = py::reinterpret_borrow<py::sequence>(observable_item);
        ObservablePayload observable;
        for (py::handle term_item : observable_terms) {
            py::dict term = py::reinterpret_borrow<py::dict>(term_item);
            if (dict_contains(term, "kind")) {
                const std::string kind =
                    required_dict_item(term, "kind", "adjoint observable").cast<std::string>();
                if (kind == "matrix") {
                    DenseMatrixPayload payload;
                    payload.matrix =
                        parse_complex_matrix_payload(required_dict_item(term, "matrix", "adjoint observable"));
                    payload.targets =
                        required_dict_item(term, "targets", "adjoint observable").cast<std::vector<unsigned>>();
                    observable.dense_terms.push_back(std::move(payload));
                    continue;
                }
                if (kind == "sparse") {
                    SparseMatrixPayload payload;
                    payload.data =
                        parse_complex_vector_payload(required_dict_item(term, "data", "adjoint observable"));
                    payload.indices = parse_size_vector(
                        required_dict_item(term, "indices", "adjoint observable"), "adjoint sparse CSR indices");
                    payload.indptr = parse_size_vector(
                        required_dict_item(term, "indptr", "adjoint observable"), "adjoint sparse CSR indptr");
                    auto shape = required_dict_item(term, "shape", "adjoint observable")
                                     .cast<std::pair<std::size_t, std::size_t>>();
                    payload.rows = shape.first;
                    payload.cols = shape.second;
                    if (dict_contains(term, "targets")) {
                        payload.targets =
                            py::reinterpret_borrow<py::object>(term[py::str("targets")]).cast<std::vector<unsigned>>();
                    }
                    observable.sparse_terms.push_back(std::move(payload));
                    continue;
                }
                throw_adjoint_not_implemented("unsupported adjoint observable kind: " + kind);
            }

            PauliTermPayload payload;
            payload.coefficient = parse_complex_pair(required_dict_item(term, "coefficient", "adjoint observable"));
            payload.pauli_string =
                required_dict_item(term, "pauli_string", "adjoint observable").cast<std::string>();
            payload.targets = required_dict_item(term, "targets", "adjoint observable").cast<std::vector<unsigned>>();
            if (payload.pauli_string.size() != payload.targets.size()) {
                throw std::invalid_argument("Pauli-string length must match the observable target count.");
            }
            observable.pauli_terms.push_back(std::move(payload));
        }
        if (observable.pauli_terms.empty() && observable.dense_terms.empty() && observable.sparse_terms.empty()) {
            throw std::invalid_argument("adjoint_jacobian observable payloads must not be empty.");
        }
        out.push_back(std::move(observable));
    }
    return out;
}

std::vector<std::complex<double>> apply_pauli_to_state(
    const std::vector<std::complex<double>>& state,
    const std::string& pauli_string,
    const std::vector<unsigned>& targets,
    unsigned num_qubits) {
    const std::size_t expected_size = checked_power_of_two(num_qubits, "simulator qubit count");
    if (state.size() != expected_size) {
        throw std::invalid_argument("Statevector size does not match the simulator qubit count.");
    }
    if (pauli_string.size() != targets.size()) {
        throw std::invalid_argument("Pauli-string length must match target count.");
    }

    std::vector<std::complex<double>> out(state.size(), std::complex<double>{0.0, 0.0});
    const std::complex<double> imag{0.0, 1.0};
    for (std::size_t basis = 0; basis < state.size(); ++basis) {
        std::size_t mapped_basis = basis;
        std::complex<double> phase{1.0, 0.0};
        for (std::size_t idx = 0; idx < targets.size(); ++idx) {
            const unsigned target = targets[idx];
            if (target >= num_qubits) {
                throw std::invalid_argument("Pauli observable target exceeds simulator qubit count.");
            }
            const char pauli = static_cast<char>(std::toupper(static_cast<unsigned char>(pauli_string[idx])));
            const std::size_t mask = checked_bit_mask(target, "Pauli observable target");
            const bool bit_is_one = (basis & mask) != 0;
            if (pauli == 'I') {
                continue;
            }
            if (pauli == 'X') {
                mapped_basis ^= mask;
                continue;
            }
            if (pauli == 'Y') {
                phase *= bit_is_one ? -imag : imag;
                mapped_basis ^= mask;
                continue;
            }
            if (pauli == 'Z') {
                if (bit_is_one) {
                    phase *= -1.0;
                }
                continue;
            }
            throw std::invalid_argument("Unsupported Pauli observable character.");
        }
        out[mapped_basis] += phase * state[basis];
    }
    return out;
}

std::vector<std::complex<double>> apply_controlled_pauli_to_state(
    const std::vector<std::complex<double>>& state,
    char pauli,
    unsigned control,
    unsigned target,
    unsigned num_qubits) {
    if (control >= num_qubits || target >= num_qubits) {
        throw std::invalid_argument("Controlled Pauli generator qubit exceeds simulator qubit count.");
    }
    if (control == target) {
        throw std::invalid_argument("Controlled Pauli generator control and target must differ.");
    }
    const std::size_t expected_size = checked_power_of_two(num_qubits, "simulator qubit count");
    if (state.size() != expected_size) {
        throw std::invalid_argument("Statevector size does not match the simulator qubit count.");
    }

    std::vector<std::complex<double>> out(state.size(), std::complex<double>{0.0, 0.0});
    const std::complex<double> imag{0.0, 1.0};
    const std::size_t control_mask = checked_bit_mask(control, "Controlled Pauli generator control");
    const std::size_t target_mask = checked_bit_mask(target, "Controlled Pauli generator target");
    pauli = static_cast<char>(std::toupper(static_cast<unsigned char>(pauli)));

    for (std::size_t basis = 0; basis < state.size(); ++basis) {
        if ((basis & control_mask) == 0) {
            continue;
        }
        const bool target_bit_is_one = (basis & target_mask) != 0;
        std::size_t mapped_basis = basis;
        std::complex<double> phase{1.0, 0.0};
        if (pauli == 'X') {
            mapped_basis ^= target_mask;
        } else if (pauli == 'Y') {
            phase *= target_bit_is_one ? -imag : imag;
            mapped_basis ^= target_mask;
        } else if (pauli == 'Z') {
            if (target_bit_is_one) {
                phase *= -1.0;
            }
        } else {
            throw std::invalid_argument("Unsupported controlled Pauli generator character.");
        }
        out[mapped_basis] += phase * state[basis];
    }
    return out;
}

std::size_t extract_local_basis_index(std::size_t basis, const std::vector<unsigned>& targets, unsigned num_qubits) {
    std::size_t local_index = 0;
    for (std::size_t bit = 0; bit < targets.size(); ++bit) {
        const unsigned target = targets[bit];
        if (target >= num_qubits) {
            throw std::invalid_argument("Dense observable target exceeds simulator qubit count.");
        }
        if (((basis >> target) & std::size_t{1}) != 0) {
            local_index |= checked_power_of_two(bit, "Dense observable local target bit");
        }
    }
    return local_index;
}

std::size_t replace_local_basis_index(
    std::size_t basis,
    const std::vector<unsigned>& targets,
    std::size_t local_index,
    unsigned num_qubits) {
    std::size_t out = basis;
    for (std::size_t bit = 0; bit < targets.size(); ++bit) {
        const unsigned target = targets[bit];
        if (target >= num_qubits) {
            throw std::invalid_argument("Dense observable target exceeds simulator qubit count.");
        }
        const std::size_t mask = checked_bit_mask(target, "Dense observable target");
        if (((local_index >> bit) & std::size_t{1}) != 0) {
            out |= mask;
        } else {
            out &= ~mask;
        }
    }
    return out;
}

std::vector<std::complex<double>> apply_dense_matrix_to_state(
    const std::vector<std::complex<double>>& state,
    const DenseMatrixPayload& payload,
    unsigned num_qubits) {
    if (payload.targets.empty()) {
        throw std::invalid_argument("Dense adjoint observable requires at least one target qubit.");
    }
    if (payload.targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
        throw std::invalid_argument("Too many dense observable target qubits.");
    }
    const std::size_t dim = checked_power_of_two(payload.targets.size(), "dense observable target count");
    if (payload.matrix.size() != checked_square_size(dim, "dense observable matrix")) {
        throw std::invalid_argument("Dense adjoint observable matrix dimension does not match target count.");
    }

    std::vector<std::complex<double>> out(state.size(), std::complex<double>{0.0, 0.0});
    for (std::size_t basis = 0; basis < state.size(); ++basis) {
        const std::size_t input_local = extract_local_basis_index(basis, payload.targets, num_qubits);
        for (std::size_t output_local = 0; output_local < dim; ++output_local) {
            const std::size_t output_basis =
                replace_local_basis_index(basis, payload.targets, output_local, num_qubits);
            out[output_basis] += payload.matrix[output_local * dim + input_local] * state[basis];
        }
    }
    return out;
}

std::vector<std::complex<double>> apply_sparse_matrix_to_state(
    const std::vector<std::complex<double>>& state,
    const SparseMatrixPayload& payload,
    unsigned num_qubits) {
    if (payload.indptr.size() != payload.rows + 1) {
        throw std::invalid_argument("Sparse adjoint observable indptr length must equal rows + 1.");
    }
    if (payload.data.size() != payload.indices.size()) {
        throw std::invalid_argument("Sparse adjoint observable data and indices lengths differ.");
    }
    for (std::size_t row = 0; row < payload.rows; ++row) {
        if (payload.indptr[row] > payload.indptr[row + 1] || payload.indptr[row + 1] > payload.data.size()) {
            throw std::invalid_argument("Sparse adjoint observable CSR indptr is invalid.");
        }
        for (std::size_t offset = payload.indptr[row]; offset < payload.indptr[row + 1]; ++offset) {
            if (payload.indices[offset] >= payload.cols) {
                throw std::invalid_argument("Sparse adjoint observable column index is out of bounds.");
            }
        }
    }

    std::vector<std::complex<double>> out(state.size(), std::complex<double>{0.0, 0.0});
    if (!payload.targets.empty()) {
        if (payload.targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
            throw std::invalid_argument("Too many sparse observable target qubits.");
        }
        const std::size_t expected_state_size = checked_power_of_two(num_qubits, "simulator qubit count");
        if (state.size() != expected_state_size) {
            throw std::invalid_argument("Statevector size does not match the simulator qubit count.");
        }
        const std::size_t dim = checked_power_of_two(payload.targets.size(), "sparse observable target count");
        if (payload.rows != dim || payload.cols != dim) {
            throw std::invalid_argument("Sparse adjoint observable matrix dimension does not match target count.");
        }

        for (std::size_t basis = 0; basis < state.size(); ++basis) {
            if (extract_local_basis_index(basis, payload.targets, num_qubits) != 0) {
                continue;
            }
            for (std::size_t row = 0; row < payload.rows; ++row) {
                const std::size_t output_basis = replace_local_basis_index(basis, payload.targets, row, num_qubits);
                for (std::size_t offset = payload.indptr[row]; offset < payload.indptr[row + 1]; ++offset) {
                    const std::size_t input_basis =
                        replace_local_basis_index(basis, payload.targets, payload.indices[offset], num_qubits);
                    out[output_basis] += payload.data[offset] * state[input_basis];
                }
            }
        }
        return out;
    }

    if (payload.rows != state.size() || payload.cols != state.size()) {
        throw_adjoint_not_implemented(
            "binding adjoint_jacobian full-state CSR sparse observables must match the state dimension.");
    }

    for (std::size_t row = 0; row < payload.rows; ++row) {
        for (std::size_t offset = payload.indptr[row]; offset < payload.indptr[row + 1]; ++offset) {
            const std::size_t col = payload.indices[offset];
            out[row] += payload.data[offset] * state[col];
        }
    }
    return out;
}

std::vector<std::complex<double>> apply_observable_to_state(
    const std::vector<std::complex<double>>& state,
    const ObservablePayload& observable,
    unsigned num_qubits) {
    std::vector<std::complex<double>> out(state.size(), std::complex<double>{0.0, 0.0});
    for (const auto& term : observable.pauli_terms) {
        auto term_state = apply_pauli_to_state(state, term.pauli_string, term.targets, num_qubits);
        for (std::size_t idx = 0; idx < out.size(); ++idx) {
            out[idx] += term.coefficient * term_state[idx];
        }
    }
    for (const auto& term : observable.dense_terms) {
        auto term_state = apply_dense_matrix_to_state(state, term, num_qubits);
        for (std::size_t idx = 0; idx < out.size(); ++idx) {
            out[idx] += term_state[idx];
        }
    }
    for (const auto& term : observable.sparse_terms) {
        auto term_state = apply_sparse_matrix_to_state(state, term, num_qubits);
        for (std::size_t idx = 0; idx < out.size(); ++idx) {
            out[idx] += term_state[idx];
        }
    }
    return out;
}

std::complex<double> inner_product(
    const std::vector<std::complex<double>>& left,
    const std::vector<std::complex<double>>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("Statevector inner product operands must have the same size.");
    }
    std::complex<double> out{0.0, 0.0};
    for (std::size_t idx = 0; idx < left.size(); ++idx) {
        out += std::conj(left[idx]) * right[idx];
    }
    return out;
}

std::vector<std::complex<double>> conjugate_transpose_matrix(
    const std::vector<std::complex<double>>& matrix,
    std::size_t dim) {
    std::vector<std::complex<double>> out(matrix.size());
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col < dim; ++col) {
            out[row * dim + col] = std::conj(matrix[col * dim + row]);
        }
    }
    return out;
}

SparseMatrixPayload conjugate_transpose_sparse_matrix(const SparseMatrixPayload& payload) {
    SparseMatrixPayload out;
    out.rows = payload.cols;
    out.cols = payload.rows;
    out.targets = payload.targets;
    out.data.resize(payload.data.size());
    out.indices.resize(payload.indices.size());
    out.indptr.assign(out.rows + 1, 0);

    for (std::size_t col : payload.indices) {
        if (col >= payload.cols) {
            throw std::invalid_argument("Sparse adjoint operation column index is out of bounds.");
        }
        ++out.indptr[col + 1];
    }
    for (std::size_t row = 0; row < out.rows; ++row) {
        out.indptr[row + 1] += out.indptr[row];
    }

    auto next = out.indptr;
    for (std::size_t row = 0; row < payload.rows; ++row) {
        for (std::size_t offset = payload.indptr[row]; offset < payload.indptr[row + 1]; ++offset) {
            const std::size_t col = payload.indices[offset];
            const std::size_t dest = next[col]++;
            out.indices[dest] = row;
            out.data[dest] = std::conj(payload.data[offset]);
        }
    }
    return out;
}

std::string inverse_gate_name(const std::string& gate_name, std::vector<double>& params) {
    const std::string normalized = uppercase_ascii(gate_name);
    if (normalized == "I" || normalized == "IDENTITY" || normalized == "H" || normalized == "HADAMARD" ||
        normalized == "X" || normalized == "PAULIX" || normalized == "Y" || normalized == "PAULIY" ||
        normalized == "Z" || normalized == "PAULIZ" || normalized == "CNOT" || normalized == "CX" ||
        normalized == "CZ" || normalized == "SWAP" || normalized == "MCX" || normalized == "CCX" ||
        normalized == "TOFFOLI" || normalized == "CSWAP" || normalized == "FREDKIN") {
        return gate_name;
    }
    if (normalized == "S") {
        return "SDG";
    }
    if (normalized == "SDG" || normalized == "SDAG") {
        return "S";
    }
    if (normalized == "T") {
        return "TDG";
    }
    if (normalized == "TDG" || normalized == "TDAG") {
        return "T";
    }
    if (normalized == "RX" || normalized == "RY" || normalized == "RZ" || normalized == "P" ||
        normalized == "PHASE" || normalized == "CRX" || normalized == "CRY" || normalized == "CRZ" ||
        normalized == "CP" || normalized == "CPHASE" || normalized == "CONTROLLEDPHASE") {
        if (params.empty()) {
            throw std::invalid_argument("Parametric adjoint operation payload is missing its angle.");
        }
        params[0] = -params[0];
        return gate_name;
    }

    throw_adjoint_not_implemented("unsupported operation in binding adjoint_jacobian: " + gate_name);
}

void apply_adjoint_operation(
    rocquantum::QuantumSimulator& simulator,
    const AdjointOperationPayload& operation,
    bool inverse) {
    if (operation.has_matrix) {
        if (operation.wires.empty()) {
            throw std::invalid_argument("Matrix adjoint operation requires at least one target wire.");
        }
        if (operation.wires.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
            throw std::invalid_argument("Too many matrix operation target qubits.");
        }
        const std::size_t dim = checked_power_of_two(operation.wires.size(), "matrix operation target count");
        if (operation.matrix.size() != checked_square_size(dim, "matrix operation")) {
            throw std::invalid_argument("Matrix adjoint operation dimension does not match target count.");
        }
        std::vector<std::complex<double>> matrix = operation.matrix;
        if (inverse) {
            matrix = conjugate_transpose_matrix(operation.matrix, dim);
        }
        if (operation.controls.empty()) {
            simulator.apply_matrix(matrix, operation.wires);
        } else {
            std::vector<unsigned> flipped_controls;
            try {
                for (std::size_t idx = 0; idx < operation.controls.size(); ++idx) {
                    if (!operation.control_values[idx]) {
                        simulator.apply_gate("X", std::vector<unsigned>{operation.controls[idx]}, std::vector<double>{});
                        flipped_controls.push_back(operation.controls[idx]);
                    }
                }
                simulator.apply_controlled_matrix(matrix, operation.controls, operation.wires);
                for (auto it = flipped_controls.rbegin(); it != flipped_controls.rend(); ++it) {
                    simulator.apply_gate("X", std::vector<unsigned>{*it}, std::vector<double>{});
                }
                flipped_controls.clear();
            } catch (...) {
                for (auto it = flipped_controls.rbegin(); it != flipped_controls.rend(); ++it) {
                    simulator.apply_gate("X", std::vector<unsigned>{*it}, std::vector<double>{});
                }
                throw;
            }
        }
        return;
    }

    if (operation.has_sparse_matrix) {
        if (operation.wires.empty()) {
            throw std::invalid_argument("Sparse matrix adjoint operation requires at least one target wire.");
        }
        if (operation.wires.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
            throw std::invalid_argument("Too many sparse matrix operation target qubits.");
        }
        const std::size_t dim = checked_power_of_two(operation.wires.size(), "sparse matrix operation target count");
        if (operation.sparse_rows != dim || operation.sparse_cols != dim) {
            throw std::invalid_argument("Sparse matrix adjoint operation dimension does not match target count.");
        }
        if (operation.sparse_indptr.size() != operation.sparse_rows + 1) {
            throw std::invalid_argument("Sparse matrix adjoint operation indptr length must equal rows + 1.");
        }
        if (operation.sparse_data.size() != operation.sparse_indices.size()) {
            throw std::invalid_argument("Sparse matrix adjoint operation data and indices lengths differ.");
        }
        SparseMatrixPayload payload;
        payload.data = operation.sparse_data;
        payload.indices = operation.sparse_indices;
        payload.indptr = operation.sparse_indptr;
        payload.targets = operation.wires;
        payload.rows = operation.sparse_rows;
        payload.cols = operation.sparse_cols;
        for (std::size_t row = 0; row < payload.rows; ++row) {
            if (payload.indptr[row] > payload.indptr[row + 1] ||
                payload.indptr[row + 1] > payload.data.size()) {
                throw std::invalid_argument("Sparse matrix adjoint operation CSR indptr is invalid.");
            }
        }
        if (inverse) {
            payload = conjugate_transpose_sparse_matrix(payload);
        }
        simulator.apply_sparse_matrix(payload.data, payload.indices, payload.indptr, payload.rows, payload.cols,
                                      payload.targets);
        return;
    }

    std::string gate_name = operation.rocq_name.empty() ? operation.name : operation.rocq_name;
    std::vector<double> params = operation.params;
    if (inverse) {
        gate_name = inverse_gate_name(gate_name, params);
    } else {
        const std::string normalized = uppercase_ascii(gate_name);
        const bool supported =
            normalized == "I" || normalized == "IDENTITY" || normalized == "H" || normalized == "HADAMARD" ||
            normalized == "X" || normalized == "PAULIX" || normalized == "Y" || normalized == "PAULIY" ||
            normalized == "Z" || normalized == "PAULIZ" || normalized == "S" || normalized == "SDG" ||
            normalized == "SDAG" || normalized == "T" || normalized == "TDG" || normalized == "TDAG" ||
            normalized == "CNOT" || normalized == "CX" || normalized == "CZ" || normalized == "SWAP" ||
            normalized == "RX" || normalized == "RY" || normalized == "RZ" || normalized == "P" ||
            normalized == "PHASE" || normalized == "CRX" || normalized == "CRY" || normalized == "CRZ" ||
            normalized == "CP" || normalized == "CPHASE" || normalized == "CONTROLLEDPHASE" ||
            normalized == "MCX" || normalized == "CCX" || normalized == "TOFFOLI" || normalized == "CSWAP" ||
            normalized == "FREDKIN";
        if (!supported) {
            throw_adjoint_not_implemented("unsupported operation in binding adjoint_jacobian: " + gate_name);
        }
    }
    simulator.apply_gate(gate_name, operation.wires, params);
}

char trainable_rotation_generator(const AdjointOperationPayload& operation) {
    const std::string normalized = uppercase_ascii(operation.rocq_name.empty() ? operation.name : operation.rocq_name);
    if (operation.trainable_param_positions.empty()) {
        return '\0';
    }
    if (operation.trainable_param_positions.size() != 1 || operation.trainable_param_positions[0] != 0 ||
        operation.params.size() != 1) {
        throw_adjoint_not_implemented(
            "binding adjoint_jacobian currently supports one trainable scalar rotation parameter per operation.");
    }
    if ((normalized == "RX" || normalized == "RY" || normalized == "RZ" || normalized == "P" ||
         normalized == "PHASE") &&
        operation.wires.size() != 1) {
        throw_adjoint_not_implemented("single-qubit adjoint rotations require one target wire.");
    }
    if ((normalized == "CRX" || normalized == "CRY" || normalized == "CRZ" || normalized == "CP" ||
         normalized == "CPHASE" || normalized == "CONTROLLEDPHASE") &&
        operation.wires.size() != 2) {
        throw_adjoint_not_implemented("controlled adjoint rotations require control and target wires.");
    }
    if (normalized == "RX" || normalized == "CRX") {
        return 'X';
    }
    if (normalized == "RY" || normalized == "CRY") {
        return 'Y';
    }
    if (normalized == "RZ" || normalized == "P" || normalized == "PHASE" || normalized == "CRZ" ||
        normalized == "CP" || normalized == "CPHASE" || normalized == "CONTROLLEDPHASE") {
        return 'Z';
    }
    throw_adjoint_not_implemented(
        "binding adjoint_jacobian currently differentiates trainable RX/RY/RZ/P/CRX/CRY/CRZ/CP operations only.");
}

std::vector<std::complex<double>> apply_rotation_generator_to_state(
    const AdjointOperationPayload& operation,
    const std::vector<std::complex<double>>& state,
    unsigned num_qubits) {
    const char generator = trainable_rotation_generator(operation);
    if (generator == '\0') {
        return {};
    }
    const std::string normalized = uppercase_ascii(operation.rocq_name.empty() ? operation.name : operation.rocq_name);
    if (normalized == "CRX" || normalized == "CRY" || normalized == "CRZ" || normalized == "CP" ||
        normalized == "CPHASE" || normalized == "CONTROLLEDPHASE") {
        return apply_controlled_pauli_to_state(state, generator, operation.wires[0], operation.wires[1], num_qubits);
    }
    return apply_pauli_to_state(state, std::string(1, generator), operation.wires, num_qubits);
}

py::array_t<double> compute_binding_adjoint_jacobian(
    const rocquantum::QuantumSimulator& self,
    py::sequence operations,
    py::sequence observables,
    const std::vector<int>& trainable_params) {
    if (self.batch_size() != 1) {
        throw_adjoint_not_implemented("adjoint_jacobian requires a single-state simulator.");
    }

    std::unordered_map<int, std::size_t> trainable_columns;
    for (std::size_t idx = 0; idx < trainable_params.size(); ++idx) {
        trainable_columns[trainable_params[idx]] = idx;
    }

    auto parsed_operations = parse_adjoint_operations(operations, trainable_columns);
    auto parsed_observables = parse_adjoint_observables(observables);

    rocquantum::QuantumSimulator forward(self.num_qubits());
    std::vector<std::vector<std::complex<double>>> forward_states;
    forward_states.push_back(forward.get_statevector());
    for (const auto& operation : parsed_operations) {
        apply_adjoint_operation(forward, operation, false);
        forward_states.push_back(forward.get_statevector());
    }

    py::array_t<double> jacobian({
        static_cast<py::ssize_t>(parsed_observables.size()),
        static_cast<py::ssize_t>(trainable_params.size()),
    });
    auto out = jacobian.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < out.shape(0); ++row) {
        for (py::ssize_t col = 0; col < out.shape(1); ++col) {
            out(row, col) = 0.0;
        }
    }

    const auto final_state = forward_states.back();
    for (std::size_t observable_idx = 0; observable_idx < parsed_observables.size(); ++observable_idx) {
        std::vector<std::complex<double>> lambda =
            apply_observable_to_state(final_state, parsed_observables[observable_idx], self.num_qubits());

        for (std::size_t reverse_idx = parsed_operations.size(); reverse_idx > 0; --reverse_idx) {
            const std::size_t operation_idx = reverse_idx - 1;
            const auto& operation = parsed_operations[operation_idx];
            auto generator_state =
                apply_rotation_generator_to_state(operation, forward_states[operation_idx + 1], self.num_qubits());
            if (!generator_state.empty()) {
                for (auto& amplitude : generator_state) {
                    amplitude *= std::complex<double>{0.0, -0.5};
                }
                const double contribution = 2.0 * std::real(inner_product(lambda, generator_state));
                for (std::size_t trainable_idx = 0; trainable_idx < operation.trainable_param_indices.size();
                     ++trainable_idx) {
                    const int param_index = operation.trainable_param_indices[trainable_idx];
                    auto column_it = trainable_columns.find(param_index);
                    if (column_it != trainable_columns.end()) {
                        const int param_position = operation.trainable_param_positions[trainable_idx];
                        if (param_position < 0 ||
                            static_cast<std::size_t>(param_position) >= operation.param_derivative_scales.size()) {
                            throw std::invalid_argument(
                                "adjoint operation trainable_param_positions entry is out of range.");
                        }
                        const double derivative_scale =
                            operation.param_derivative_scales[static_cast<std::size_t>(param_position)];
                        out(static_cast<py::ssize_t>(observable_idx),
                            static_cast<py::ssize_t>(column_it->second)) += derivative_scale * contribution;
                    }
                }
            }

            rocquantum::QuantumSimulator reverse_state(self.num_qubits());
            reverse_state.set_statevector(lambda);
            apply_adjoint_operation(reverse_state, operation, true);
            lambda = reverse_state.get_statevector();
        }
    }

    return jacobian;
}

} // namespace

PYBIND11_MODULE(rocquantum_bind, m) {
    m.doc() = "pybind11 plugin for rocQuantum-1";

#ifdef ROCQUANTUM_ENABLE_MLIR_COMPILER
    m.attr("MLIR_COMPILER_ENABLED") = py::bool_(true);
    m.attr("MLIR_COMPILER_RUNTIME_KIND") = "linked_runtime";
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
#else
    m.attr("MLIR_COMPILER_ENABLED") = py::bool_(false);
    m.attr("MLIR_COMPILER_RUNTIME_KIND") = "disabled_runtime_guard";
    py::class_<DisabledRuntimeMLIRCompiler>(m, "MLIRCompiler")
        .def(py::init<unsigned, std::string>(),
             py::arg("num_qubits"),
             py::arg("backend_name"),
             "Records a requested MLIR compiler configuration. The default build does not link "
             "the experimental rocqCompiler MLIR runtime.")
        .def_property_readonly("num_qubits", &DisabledRuntimeMLIRCompiler::num_qubits)
        .def_property_readonly("backend_name", &DisabledRuntimeMLIRCompiler::backend_name)
        .def("compile_and_execute",
             [](DisabledRuntimeMLIRCompiler& self, const std::string& mlir, py::dict _args) {
                 (void)_args;
                 return self.compile_and_execute(mlir);
             },
             py::arg("mlir"),
             py::arg("args") = py::dict(),
             "Fails fast when the default binding is built without the experimental "
             "rocqCompiler MLIR runtime. Supported source subset: qalloc, H/X/Y/Z/S/Sdg/T/Tdg, "
             "CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, CRX/CRY/CRZ/CP.")
        .def("emit_qir", &DisabledRuntimeMLIRCompiler::emit_qir,
             "Fails fast when the default binding is built without the experimental "
             "rocqCompiler MLIR runtime.");
#endif

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

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "matrix target count");
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

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "controlled matrix target count");
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
        .def("adjoint_jacobian",
             &compute_binding_adjoint_jacobian,
             py::arg("operations"),
             py::arg("observables"),
             py::arg("trainable_params"),
             "Return an exact adjoint Jacobian for supported RX/RY/RZ/CRX/CRY/CRZ observable payloads.")
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

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "expectation matrix target count");
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
        .def("expectation_matrix_moments",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("expectation_matrix_moments requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("expectation_matrix_moments expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for expectation_matrix_moments.");
                 }

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "expectation matrix target count");
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("expectation_matrix_moments dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 return self.expectation_matrix_moments(host, targets);
             },
             py::arg("matrix"),
             py::arg("targets"),
             "Return (<M>, <M^2>) for a dense matrix observable.")
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

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "expectation matrix target count");
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
        .def("expectation_matrix_moments_batch",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("expectation_matrix_moments_batch requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("expectation_matrix_moments_batch expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for expectation_matrix_moments_batch.");
                 }

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "expectation matrix target count");
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument(
                         "expectation_matrix_moments_batch dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 auto moments = self.expectation_matrix_moments_batch(host, targets);
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
             py::arg("matrix"),
             py::arg("targets"),
             "Return batch-major (<M>, <M^2>) arrays for a dense matrix observable.")
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
        .def("AdjointJacobian",
             &compute_binding_adjoint_jacobian,
             py::arg("operations"),
             py::arg("observables"),
             py::arg("trainable_params"))
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

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "batch matrix target count");
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
        .def("ExpectationMatrixMoments",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("ExpectationMatrixMoments requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("ExpectationMatrixMoments expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for ExpectationMatrixMoments.");
                 }

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "batch matrix target count");
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("ExpectationMatrixMoments dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 return self.ExpectationMatrixMoments(host, targets);
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

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "batch expectation matrix target count");
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
        .def("ExpectationMatrixMomentsBatch",
             [](const rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (targets.empty()) {
                     throw std::invalid_argument("ExpectationMatrixMomentsBatch requires at least one target qubit.");
                 }
                 if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
                     throw std::invalid_argument("ExpectationMatrixMomentsBatch expects a square complex matrix.");
                 }
                 if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
                     throw std::invalid_argument("Too many target qubits for ExpectationMatrixMomentsBatch.");
                 }

                 const std::size_t expected_dim = checked_power_of_two(targets.size(), "batch expectation matrix target count");
                 const std::size_t actual_dim = static_cast<std::size_t>(matrix.shape(0));
                 if (actual_dim != expected_dim) {
                     throw std::invalid_argument("ExpectationMatrixMomentsBatch dimension must be 2^len(targets).");
                 }

                 std::vector<std::complex<double>> host(actual_dim * actual_dim);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 auto moments = self.ExpectationMatrixMomentsBatch(host, targets);
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
