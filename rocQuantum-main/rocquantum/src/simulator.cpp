#include "rocquantum/QuantumSimulator.h"

#include <algorithm>
#include <cctype>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <hip/hip_runtime.h>

namespace {

std::string normalize_gate_name(const std::string& gate_name) {
    std::string upper;
    upper.reserve(gate_name.size());
    std::transform(gate_name.begin(),
                   gate_name.end(),
                   std::back_inserter(upper),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return upper;
}

void check_status(rocqStatus_t status, const char* context) {
    if (status != ROCQ_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("hipStateVec error during ") + context +
                                 " (status " + std::to_string(status) + ")");
    }
}

void check_hip(hipError_t status, const char* context) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("HIP error during ") + context + ": " +
                                 hipGetErrorString(status));
    }
}

rocComplex to_roc_complex(const std::complex<double>& value) {
#ifdef ROCQ_PRECISION_DOUBLE
    return rocComplex{value.real(), value.imag()};
#else
    return rocComplex{static_cast<float>(value.real()), static_cast<float>(value.imag())};
#endif
}

std::vector<rocComplex> row_major_to_column_major(const std::vector<std::complex<double>>& matrix,
                                                  std::size_t matrix_dim) {
    std::vector<rocComplex> out(matrix.size());
    for (std::size_t row = 0; row < matrix_dim; ++row) {
        for (std::size_t col = 0; col < matrix_dim; ++col) {
            out[row + col * matrix_dim] = to_roc_complex(matrix[row * matrix_dim + col]);
        }
    }
    return out;
}

} // namespace

namespace rocquantum {

QuantumSimulator::QuantumSimulator(unsigned num_qubits)
    : num_qubits_(num_qubits),
      state_vec_size_(0),
      sim_handle_(nullptr),
      device_state_vector_(nullptr) {
    if (num_qubits_ == 0) {
        throw std::invalid_argument("QuantumSimulator requires at least one qubit.");
    }
    if (num_qubits_ >= static_cast<unsigned>(sizeof(std::size_t) * 8)) {
        throw std::invalid_argument("num_qubits is too large for this build.");
    }

    state_vec_size_ = std::size_t{1} << num_qubits_;

    check_status(rocsvCreate(&sim_handle_), "handle creation");
    try {
        check_status(rocsvAllocateState(sim_handle_, num_qubits_, &device_state_vector_, 1), "state allocation");
        check_status(rocsvInitializeState(sim_handle_, device_state_vector_, num_qubits_), "state initialization");
    } catch (...) {
        if (sim_handle_) {
            rocsvDestroy(sim_handle_);
            sim_handle_ = nullptr;
            device_state_vector_ = nullptr;
        }
        throw;
    }
}

QuantumSimulator::~QuantumSimulator() {
    if (sim_handle_) {
        rocsvDestroy(sim_handle_);
        sim_handle_ = nullptr;
        device_state_vector_ = nullptr;
    }
}

void QuantumSimulator::reset() {
    check_status(rocsvInitializeState(sim_handle_, device_state_vector_, num_qubits_), "reset");
}

void QuantumSimulator::apply_gate(const std::string& gate_name,
                                  const std::vector<unsigned>& targets,
                                  const std::vector<double>& params) {
    if (targets.empty()) {
        throw std::invalid_argument("apply_gate requires at least one target qubit.");
    }

    const std::string normalized = normalize_gate_name(gate_name);
    for (unsigned target : targets) {
        ensure_valid_qubit(target);
    }

    if (normalized == "I" || normalized == "IDENTITY") {
        return;
    }

    if (normalized == "CNOT" || normalized == "CX") {
        if (targets.size() != 2) {
            throw std::invalid_argument("CNOT requires exactly 2 target qubits.");
        }
        check_status(rocsvApplyCNOT(sim_handle_, device_state_vector_, num_qubits_, targets[0], targets[1]),
                     "apply CNOT");
        return;
    }
    if (normalized == "CZ") {
        if (targets.size() != 2) {
            throw std::invalid_argument("CZ requires exactly 2 target qubits.");
        }
        check_status(rocsvApplyCZ(sim_handle_, device_state_vector_, num_qubits_, targets[0], targets[1]),
                     "apply CZ");
        return;
    }
    if (normalized == "SWAP") {
        if (targets.size() != 2) {
            throw std::invalid_argument("SWAP requires exactly 2 target qubits.");
        }
        check_status(rocsvApplySWAP(sim_handle_, device_state_vector_, num_qubits_, targets[0], targets[1]),
                     "apply SWAP");
        return;
    }
    if (normalized == "H" || normalized == "HADAMARD") {
        if (targets.size() != 1) {
            throw std::invalid_argument("H requires exactly 1 target qubit.");
        }
        check_status(rocsvApplyH(sim_handle_, device_state_vector_, num_qubits_, targets[0]), "apply H");
        return;
    }
    if (normalized == "X" || normalized == "PAULIX") {
        if (targets.size() != 1) {
            throw std::invalid_argument("X requires exactly 1 target qubit.");
        }
        check_status(rocsvApplyX(sim_handle_, device_state_vector_, num_qubits_, targets[0]), "apply X");
        return;
    }
    if (normalized == "Y" || normalized == "PAULIY") {
        if (targets.size() != 1) {
            throw std::invalid_argument("Y requires exactly 1 target qubit.");
        }
        check_status(rocsvApplyY(sim_handle_, device_state_vector_, num_qubits_, targets[0]), "apply Y");
        return;
    }
    if (normalized == "Z" || normalized == "PAULIZ") {
        if (targets.size() != 1) {
            throw std::invalid_argument("Z requires exactly 1 target qubit.");
        }
        check_status(rocsvApplyZ(sim_handle_, device_state_vector_, num_qubits_, targets[0]), "apply Z");
        return;
    }
    if (normalized == "S") {
        if (targets.size() != 1) {
            throw std::invalid_argument("S requires exactly 1 target qubit.");
        }
        check_status(rocsvApplyS(sim_handle_, device_state_vector_, num_qubits_, targets[0]), "apply S");
        return;
    }
    if (normalized == "SDG" || normalized == "SDAG") {
        if (targets.size() != 1) {
            throw std::invalid_argument("SDG requires exactly 1 target qubit.");
        }
        check_status(rocsvApplySdg(sim_handle_, device_state_vector_, num_qubits_, targets[0]), "apply SDG");
        return;
    }
    if (normalized == "T") {
        if (targets.size() != 1) {
            throw std::invalid_argument("T requires exactly 1 target qubit.");
        }
        check_status(rocsvApplyT(sim_handle_, device_state_vector_, num_qubits_, targets[0]), "apply T");
        return;
    }

    if (normalized == "RX") {
        if (targets.size() != 1) {
            throw std::invalid_argument("RX requires exactly 1 target qubit.");
        }
        if (params.empty()) {
            throw std::invalid_argument("RX requires one angle parameter.");
        }
        check_status(rocsvApplyRx(sim_handle_, device_state_vector_, num_qubits_, targets[0], params[0]),
                     "apply RX");
        return;
    }
    if (normalized == "RY") {
        if (targets.size() != 1) {
            throw std::invalid_argument("RY requires exactly 1 target qubit.");
        }
        if (params.empty()) {
            throw std::invalid_argument("RY requires one angle parameter.");
        }
        check_status(rocsvApplyRy(sim_handle_, device_state_vector_, num_qubits_, targets[0], params[0]),
                     "apply RY");
        return;
    }
    if (normalized == "RZ") {
        if (targets.size() != 1) {
            throw std::invalid_argument("RZ requires exactly 1 target qubit.");
        }
        if (params.empty()) {
            throw std::invalid_argument("RZ requires one angle parameter.");
        }
        check_status(rocsvApplyRz(sim_handle_, device_state_vector_, num_qubits_, targets[0], params[0]),
                     "apply RZ");
        return;
    }
    if (normalized == "CRX") {
        if (targets.size() != 2) {
            throw std::invalid_argument("CRX requires control and target qubits.");
        }
        if (params.empty()) {
            throw std::invalid_argument("CRX requires one angle parameter.");
        }
        check_status(rocsvApplyCRX(sim_handle_, device_state_vector_, num_qubits_, targets[0], targets[1], params[0]),
                     "apply CRX");
        return;
    }
    if (normalized == "CRY") {
        if (targets.size() != 2) {
            throw std::invalid_argument("CRY requires control and target qubits.");
        }
        if (params.empty()) {
            throw std::invalid_argument("CRY requires one angle parameter.");
        }
        check_status(rocsvApplyCRY(sim_handle_, device_state_vector_, num_qubits_, targets[0], targets[1], params[0]),
                     "apply CRY");
        return;
    }
    if (normalized == "CRZ") {
        if (targets.size() != 2) {
            throw std::invalid_argument("CRZ requires control and target qubits.");
        }
        if (params.empty()) {
            throw std::invalid_argument("CRZ requires one angle parameter.");
        }
        check_status(rocsvApplyCRZ(sim_handle_, device_state_vector_, num_qubits_, targets[0], targets[1], params[0]),
                     "apply CRZ");
        return;
    }

    throw std::runtime_error("Gate '" + gate_name + "' is not supported.");
}

void QuantumSimulator::apply_matrix(const std::vector<std::complex<double>>& matrix,
                                    const std::vector<unsigned>& targets) {
    if (targets.empty()) {
        throw std::invalid_argument("apply_matrix requires at least one target qubit.");
    }
    for (unsigned target : targets) {
        ensure_valid_qubit(target);
    }

    if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
        throw std::invalid_argument("Too many target qubits for matrix application.");
    }
    const std::size_t matrix_dim = std::size_t{1} << targets.size();
    const std::size_t expected_elements = matrix_dim * matrix_dim;
    if (matrix.size() != expected_elements) {
        throw std::invalid_argument("Matrix element count does not match target qubit count.");
    }
    if (matrix_dim > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
        throw std::invalid_argument("Matrix dimension exceeds API limit.");
    }

    const std::vector<rocComplex> matrix_col_major = row_major_to_column_major(matrix, matrix_dim);

    rocComplex* d_matrix = nullptr;
    check_hip(hipMalloc(&d_matrix, matrix_col_major.size() * sizeof(rocComplex)), "matrix allocation");
    try {
        check_hip(hipMemcpy(d_matrix,
                            matrix_col_major.data(),
                            matrix_col_major.size() * sizeof(rocComplex),
                            hipMemcpyHostToDevice),
                  "matrix upload");
        check_status(rocsvApplyMatrix(sim_handle_,
                                      device_state_vector_,
                                      num_qubits_,
                                      targets.data(),
                                      static_cast<unsigned>(targets.size()),
                                      d_matrix,
                                      static_cast<unsigned>(matrix_dim)),
                     "apply matrix");
    } catch (...) {
        hipFree(d_matrix);
        throw;
    }
    check_hip(hipFree(d_matrix), "matrix free");
}

std::vector<std::complex<double>> QuantumSimulator::get_statevector() const {
    std::vector<rocComplex> raw(state_vec_size_);
    check_status(rocsvGetStateVectorFull(sim_handle_, device_state_vector_, raw.data()), "state readback");

    std::vector<std::complex<double>> out(state_vec_size_);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out[i] = std::complex<double>(static_cast<double>(raw[i].x),
                                      static_cast<double>(raw[i].y));
    }
    return out;
}

std::vector<long long> QuantumSimulator::measure(const std::vector<unsigned>& qubits, int shots) {
    if (shots < 0) {
        throw std::invalid_argument("shots must be non-negative.");
    }
    if (shots == 0) {
        return {};
    }
    if (qubits.empty()) {
        throw std::invalid_argument("measure requires at least one target qubit.");
    }

    std::unordered_set<unsigned> unique_qubits;
    for (unsigned q : qubits) {
        ensure_valid_qubit(q);
        if (!unique_qubits.insert(q).second) {
            throw std::invalid_argument("measure qubits must be unique.");
        }
    }

    std::vector<uint64_t> sampled(static_cast<std::size_t>(shots), 0);
    check_status(rocsvSample(sim_handle_,
                             device_state_vector_,
                             num_qubits_,
                             qubits.data(),
                             static_cast<unsigned>(qubits.size()),
                             static_cast<unsigned>(shots),
                             sampled.data()),
                 "sampling");

    std::vector<long long> out(static_cast<std::size_t>(shots), 0);
    for (std::size_t i = 0; i < sampled.size(); ++i) {
        out[i] = static_cast<long long>(sampled[i]);
    }
    return out;
}

unsigned QuantumSimulator::num_qubits() const noexcept {
    return num_qubits_;
}

void QuantumSimulator::ApplyGate(const std::string& gate_name, int target_qubit) {
    apply_gate(gate_name, {static_cast<unsigned>(target_qubit)}, {});
}

void QuantumSimulator::ApplyGate(const std::string& gate_name, int control_qubit, int target_qubit) {
    apply_gate(gate_name,
               {static_cast<unsigned>(control_qubit), static_cast<unsigned>(target_qubit)},
               {});
}

void QuantumSimulator::ApplyGate(const std::vector<std::complex<double>>& gate_matrix, int target_qubit) {
    apply_matrix(gate_matrix, {static_cast<unsigned>(target_qubit)});
}

void QuantumSimulator::Execute() {
    synchronize();
}

std::vector<std::complex<double>> QuantumSimulator::GetStateVector() const {
    return get_statevector();
}

void QuantumSimulator::ensure_valid_qubit(unsigned qubit) const {
    if (qubit >= num_qubits_) {
        throw std::out_of_range("Qubit index out of bounds for simulator instance.");
    }
}

void QuantumSimulator::synchronize() const {
    hipStream_t stream = nullptr;
    check_status(rocsvGetStream(sim_handle_, &stream), "stream query");
    if (stream) {
        check_hip(hipStreamSynchronize(stream), "stream synchronization");
    } else {
        check_hip(hipDeviceSynchronize(), "device synchronization");
    }
}

} // namespace rocquantum
