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

QuantumSimulator::QuantumSimulator(unsigned num_qubits, std::size_t batch_size)
    : num_qubits_(num_qubits),
      batch_size_(batch_size),
      state_vec_size_(0),
      sim_handle_(nullptr),
      device_state_vector_(nullptr) {
    if (num_qubits_ == 0) {
        throw std::invalid_argument("QuantumSimulator requires at least one qubit.");
    }
    if (batch_size_ == 0) {
        throw std::invalid_argument("QuantumSimulator batch_size must be at least 1.");
    }
    if (batch_size_ > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
        throw std::invalid_argument("QuantumSimulator batch_size exceeds API limit.");
    }
    if (num_qubits_ >= static_cast<unsigned>(sizeof(std::size_t) * 8)) {
        throw std::invalid_argument("num_qubits is too large for this build.");
    }

    state_vec_size_ = std::size_t{1} << num_qubits_;

    check_status(rocsvCreate(&sim_handle_), "handle creation");
    try {
        check_status(rocsvAllocateState(sim_handle_, num_qubits_, &device_state_vector_, batch_size_),
                     "state allocation");
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

int QuantumSimulator::measure_qubit(unsigned target) {
    if (batch_size_ != 1) {
        throw std::invalid_argument("measure_qubit is only valid when batch_size is 1.");
    }
    ensure_valid_qubit(target);
    int outcome = 0;
    double probability = 0.0;
    check_status(rocsvMeasure(sim_handle_,
                              device_state_vector_,
                              num_qubits_,
                              target,
                              &outcome,
                              &probability),
                 "measure qubit");
    return outcome;
}

void QuantumSimulator::reset_qubit(unsigned target) {
    const int outcome = measure_qubit(target);
    if (outcome == 1) {
        check_status(rocsvApplyX(sim_handle_, device_state_vector_, num_qubits_, target),
                     "reset qubit conditional X");
    }
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
    if (normalized == "TDG" || normalized == "TDAG") {
        if (targets.size() != 1) {
            throw std::invalid_argument("TDG requires exactly 1 target qubit.");
        }
        check_status(rocsvApplyTdg(sim_handle_, device_state_vector_, num_qubits_, targets[0]), "apply TDG");
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
    if (normalized == "P" || normalized == "PHASE") {
        if (targets.size() != 1) {
            throw std::invalid_argument("P requires exactly 1 target qubit.");
        }
        if (params.empty()) {
            throw std::invalid_argument("P requires one angle parameter.");
        }
        check_status(rocsvApplyP(sim_handle_, device_state_vector_, num_qubits_, targets[0], params[0]),
                     "apply P");
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
    if (normalized == "CP" || normalized == "CPHASE" || normalized == "CONTROLLEDPHASE") {
        if (targets.size() != 2) {
            throw std::invalid_argument("CP requires control and target qubits.");
        }
        if (params.empty()) {
            throw std::invalid_argument("CP requires one angle parameter.");
        }
        check_status(rocsvApplyCP(sim_handle_, device_state_vector_, num_qubits_, targets[0], targets[1], params[0]),
                     "apply CP");
        return;
    }
    if (normalized == "MCX" || normalized == "CCX" || normalized == "TOFFOLI") {
        if (targets.size() < 2) {
            throw std::invalid_argument("MCX requires at least one control qubit and one target qubit.");
        }
        if ((normalized == "CCX" || normalized == "TOFFOLI") && targets.size() != 3) {
            throw std::invalid_argument("CCX/Toffoli requires exactly two controls and one target qubit.");
        }

        const unsigned target = targets.back();
        std::vector<unsigned> controls(targets.begin(), targets.end() - 1);
        std::unordered_set<unsigned> unique_controls;
        for (unsigned control : controls) {
            if (control == target) {
                throw std::invalid_argument("MCX control qubits must differ from the target qubit.");
            }
            if (!unique_controls.insert(control).second) {
                throw std::invalid_argument("MCX control qubits must be unique.");
            }
        }

        check_status(
            rocsvApplyMultiControlledX(sim_handle_,
                                       device_state_vector_,
                                       num_qubits_,
                                       controls.data(),
                                       static_cast<unsigned>(controls.size()),
                                       target),
            "apply MCX");
        return;
    }
    if (normalized == "CSWAP" || normalized == "FREDKIN") {
        if (targets.size() != 3) {
            throw std::invalid_argument("CSWAP requires control and two target qubits.");
        }
        check_status(rocsvApplyCSWAP(sim_handle_,
                                     device_state_vector_,
                                     num_qubits_,
                                     targets[0],
                                     targets[1],
                                     targets[2]),
                     "apply CSWAP");
        return;
    }

    throw std::runtime_error("Gate '" + gate_name + "' is not supported.");
}

void QuantumSimulator::apply_gate_batch(const std::string& gate_name,
                                        const std::vector<unsigned>& targets,
                                        const std::vector<double>& params_by_batch) {
    const std::string normalized = normalize_gate_name(gate_name);
    if (params_by_batch.size() != batch_size_) {
        throw std::invalid_argument("Batched gate parameter count must equal batch_size.");
    }

    if (normalized == "RX" || normalized == "RY" || normalized == "RZ") {
        if (targets.size() != 1) {
            throw std::invalid_argument("Batched RX/RY/RZ gates require exactly 1 target qubit.");
        }
        ensure_valid_qubit(targets[0]);
    } else if (normalized == "CRX" || normalized == "CRY" || normalized == "CRZ") {
        if (targets.size() != 2) {
            throw std::invalid_argument("Batched CRX/CRY/CRZ gates require control and target qubits.");
        }
        ensure_valid_qubit(targets[0]);
        ensure_valid_qubit(targets[1]);
        if (targets[0] == targets[1]) {
            throw std::invalid_argument("Batched controlled rotation control and target must differ.");
        }
    }

    if (normalized == "RX") {
        check_status(rocsvApplyRxBatch(sim_handle_,
                                       device_state_vector_,
                                       num_qubits_,
                                       targets[0],
                                       params_by_batch.data(),
                                       params_by_batch.size()),
                     "apply batched RX");
        return;
    }
    if (normalized == "RY") {
        check_status(rocsvApplyRyBatch(sim_handle_,
                                       device_state_vector_,
                                       num_qubits_,
                                       targets[0],
                                       params_by_batch.data(),
                                       params_by_batch.size()),
                     "apply batched RY");
        return;
    }
    if (normalized == "RZ") {
        check_status(rocsvApplyRzBatch(sim_handle_,
                                       device_state_vector_,
                                       num_qubits_,
                                       targets[0],
                                       params_by_batch.data(),
                                       params_by_batch.size()),
                     "apply batched RZ");
        return;
    }
    if (normalized == "CRX") {
        check_status(rocsvApplyCRXBatch(sim_handle_,
                                        device_state_vector_,
                                        num_qubits_,
                                        targets[0],
                                        targets[1],
                                        params_by_batch.data(),
                                        params_by_batch.size()),
                     "apply batched CRX");
        return;
    }
    if (normalized == "CRY") {
        check_status(rocsvApplyCRYBatch(sim_handle_,
                                        device_state_vector_,
                                        num_qubits_,
                                        targets[0],
                                        targets[1],
                                        params_by_batch.data(),
                                        params_by_batch.size()),
                     "apply batched CRY");
        return;
    }
    if (normalized == "CRZ") {
        check_status(rocsvApplyCRZBatch(sim_handle_,
                                        device_state_vector_,
                                        num_qubits_,
                                        targets[0],
                                        targets[1],
                                        params_by_batch.data(),
                                        params_by_batch.size()),
                     "apply batched CRZ");
        return;
    }

    throw std::runtime_error("Gate '" + gate_name + "' is not supported for batched parameters.");
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

void QuantumSimulator::apply_controlled_matrix(const std::vector<std::complex<double>>& matrix,
                                               const std::vector<unsigned>& controls,
                                               const std::vector<unsigned>& targets) {
    if (controls.empty()) {
        throw std::invalid_argument("apply_controlled_matrix requires at least one control qubit.");
    }
    if (targets.empty()) {
        throw std::invalid_argument("apply_controlled_matrix requires at least one target qubit.");
    }

    std::unordered_set<unsigned> seen;
    for (unsigned control : controls) {
        ensure_valid_qubit(control);
        if (!seen.insert(control).second) {
            throw std::invalid_argument("apply_controlled_matrix control qubits must be unique.");
        }
    }
    for (unsigned target : targets) {
        ensure_valid_qubit(target);
        if (!seen.insert(target).second) {
            throw std::invalid_argument("apply_controlled_matrix controls and targets must be unique and disjoint.");
        }
    }

    if (targets.size() >= static_cast<std::size_t>(sizeof(std::size_t) * 8)) {
        throw std::invalid_argument("Too many target qubits for controlled matrix application.");
    }
    const std::size_t matrix_dim = std::size_t{1} << targets.size();
    const std::size_t expected_elements = matrix_dim * matrix_dim;
    if (matrix.size() != expected_elements) {
        throw std::invalid_argument("Controlled matrix element count does not match target qubit count.");
    }

    const std::vector<rocComplex> matrix_col_major = row_major_to_column_major(matrix, matrix_dim);

    rocComplex* d_matrix = nullptr;
    check_hip(hipMalloc(&d_matrix, matrix_col_major.size() * sizeof(rocComplex)), "controlled matrix allocation");
    try {
        check_hip(hipMemcpy(d_matrix,
                            matrix_col_major.data(),
                            matrix_col_major.size() * sizeof(rocComplex),
                            hipMemcpyHostToDevice),
                  "controlled matrix upload");
        check_status(rocsvApplyControlledMatrix(sim_handle_,
                                                device_state_vector_,
                                                num_qubits_,
                                                controls.data(),
                                                static_cast<unsigned>(controls.size()),
                                                targets.data(),
                                                static_cast<unsigned>(targets.size()),
                                                d_matrix),
                     "apply controlled matrix");
    } catch (...) {
        hipFree(d_matrix);
        throw;
    }
    check_hip(hipFree(d_matrix), "controlled matrix free");
}

void QuantumSimulator::set_statevector(const std::vector<std::complex<double>>& state) {
    if (batch_size_ != 1) {
        throw std::invalid_argument("set_statevector is only valid when batch_size is 1; use set_statevectors.");
    }
    if (state.size() != state_vec_size_) {
        throw std::invalid_argument("set_statevector input size must equal 2^num_qubits.");
    }

    set_statevectors(state);
}

void QuantumSimulator::set_statevectors(const std::vector<std::complex<double>>& states) {
    if (states.size() != batch_size_ * state_vec_size_) {
        throw std::invalid_argument("set_statevectors input size must equal batch_size * 2^num_qubits.");
    }

    std::vector<rocComplex> raw(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) {
        raw[i] = to_roc_complex(states[i]);
    }

    check_hip(hipMemcpy(device_state_vector_,
                        raw.data(),
                        raw.size() * sizeof(rocComplex),
                        hipMemcpyHostToDevice),
              "statevectors upload");
}

std::vector<std::complex<double>> QuantumSimulator::get_statevector(std::size_t batch_index) const {
    if (batch_index >= batch_size_) {
        throw std::out_of_range("batch_index is out of range for simulator batch_size.");
    }

    std::vector<rocComplex> raw(state_vec_size_);
    check_status(
        rocsvGetStateVectorSlice(
            sim_handle_,
            device_state_vector_,
            raw.data(),
            static_cast<unsigned>(batch_index)),
        "state slice readback");

    std::vector<std::complex<double>> out(state_vec_size_);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out[i] = std::complex<double>(static_cast<double>(raw[i].x),
                                      static_cast<double>(raw[i].y));
    }
    return out;
}

std::vector<std::complex<double>> QuantumSimulator::get_statevectors() const {
    std::vector<rocComplex> raw(batch_size_ * state_vec_size_);
    check_status(rocsvGetStateVectorFull(sim_handle_, device_state_vector_, raw.data()), "state batch readback");

    std::vector<std::complex<double>> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out[i] = std::complex<double>(static_cast<double>(raw[i].x),
                                      static_cast<double>(raw[i].y));
    }
    return out;
}

std::vector<double> QuantumSimulator::probabilities(const std::vector<unsigned>& qubits) const {
    if (batch_size_ != 1) {
        throw std::invalid_argument("probabilities is only valid when batch_size is 1; use probabilities_batch.");
    }
    std::vector<double> out = probabilities_batch(qubits);
    return out;
}

std::vector<double> QuantumSimulator::probabilities_batch(const std::vector<unsigned>& qubits) const {
    std::vector<unsigned> targets;
    if (qubits.empty()) {
        targets.reserve(num_qubits_);
        for (unsigned q = 0; q < num_qubits_; ++q) {
            targets.push_back(q);
        }
    } else {
        targets = qubits;
    }

    if (targets.size() > 20) {
        throw std::runtime_error("probabilities currently supports at most 20 target qubits.");
    }

    std::unordered_set<unsigned> unique_qubits;
    for (unsigned q : targets) {
        ensure_valid_qubit(q);
        if (!unique_qubits.insert(q).second) {
            throw std::invalid_argument("probability qubits must be unique.");
        }
    }

    const std::size_t num_outcomes = std::size_t{1} << targets.size();
    std::vector<double> out(batch_size_ * num_outcomes, 0.0);
    check_status(rocsvProbabilitiesBatch(sim_handle_,
                                         device_state_vector_,
                                         num_qubits_,
                                         targets.data(),
                                         static_cast<unsigned>(targets.size()),
                                         out.data()),
                 "batch probabilities");
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

std::vector<long long> QuantumSimulator::measure_batch(const std::vector<unsigned>& qubits, int shots) {
    if (shots < 0) {
        throw std::invalid_argument("shots must be non-negative.");
    }
    if (qubits.empty()) {
        throw std::invalid_argument("measure_batch requires at least one target qubit.");
    }

    std::unordered_set<unsigned> unique_qubits;
    for (unsigned q : qubits) {
        ensure_valid_qubit(q);
        if (!unique_qubits.insert(q).second) {
            throw std::invalid_argument("measure_batch qubits must be unique.");
        }
    }

    const std::size_t shot_count = static_cast<std::size_t>(shots);
    std::vector<long long> out(batch_size_ * shot_count, 0);
    if (shots == 0) {
        return out;
    }

    std::vector<uint64_t> sampled(shot_count, 0);
    for (std::size_t batch_index = 0; batch_index < batch_size_; ++batch_index) {
        rocComplex* batch_state = device_state_vector_ + batch_index * state_vec_size_;
        check_status(rocsvSample(sim_handle_,
                                 batch_state,
                                 num_qubits_,
                                 qubits.data(),
                                 static_cast<unsigned>(qubits.size()),
                                 static_cast<unsigned>(shots),
                                 sampled.data()),
                     "batch sampling");
        const std::size_t offset = batch_index * shot_count;
        for (std::size_t shot = 0; shot < shot_count; ++shot) {
            out[offset + shot] = static_cast<long long>(sampled[shot]);
        }
    }
    return out;
}

double QuantumSimulator::expectation_value(const std::string& pauli, unsigned target) {
    return expectation_pauli_string(pauli, {target});
}

double QuantumSimulator::expectation_pauli_string(const std::string& pauli_string,
                                                  const std::vector<unsigned>& targets) {
    if (batch_size_ != 1) {
        throw std::invalid_argument(
            "expectation_pauli_string is only valid when batch_size is 1; use expectation_pauli_string_batch.");
    }
    std::vector<double> out = expectation_pauli_string_batch(pauli_string, targets);
    return out.front();
}

std::vector<double> QuantumSimulator::expectation_pauli_string_batch(const std::string& pauli_string,
                                                                     const std::vector<unsigned>& targets) {
    if (pauli_string.size() != targets.size()) {
        throw std::invalid_argument("Pauli string length must match target qubit count.");
    }
    if (targets.empty()) {
        return std::vector<double>(batch_size_, 1.0);
    }

    std::string normalized;
    normalized.reserve(pauli_string.size());
    std::unordered_set<unsigned> unique_qubits;
    for (std::size_t idx = 0; idx < targets.size(); ++idx) {
        ensure_valid_qubit(targets[idx]);
        if (!unique_qubits.insert(targets[idx]).second) {
            throw std::invalid_argument("Pauli expectation targets must be unique.");
        }

        const char pauli = static_cast<char>(std::toupper(static_cast<unsigned char>(pauli_string[idx])));
        if (pauli != 'I' && pauli != 'X' && pauli != 'Y' && pauli != 'Z') {
            throw std::invalid_argument("Pauli string may only contain I, X, Y, or Z.");
        }
        normalized.push_back(pauli);
    }

    std::vector<double> out(batch_size_, 0.0);
    check_status(rocsvGetExpectationPauliStringBatch(sim_handle_,
                                                     device_state_vector_,
                                                     num_qubits_,
                                                     normalized.c_str(),
                                                     targets.data(),
                                                     static_cast<unsigned>(targets.size()),
                                                     out.data()),
                 "batch Pauli-string expectation");
    return out;
}

std::complex<double> QuantumSimulator::expectation_matrix(
    const std::vector<std::complex<double>>& matrix,
    const std::vector<unsigned>& targets) const {
    if (targets.empty()) {
        throw std::invalid_argument("expectation_matrix requires at least one target qubit.");
    }
    if (targets.size() > 4) {
        throw std::runtime_error("expectation_matrix currently supports at most 4 target qubits.");
    }

    std::unordered_set<unsigned> unique_targets;
    for (unsigned target : targets) {
        ensure_valid_qubit(target);
        if (!unique_targets.insert(target).second) {
            throw std::invalid_argument("expectation_matrix target qubits must be unique.");
        }
    }

    const std::size_t matrix_dim = std::size_t{1} << targets.size();
    const std::size_t expected_elements = matrix_dim * matrix_dim;
    if (matrix.size() != expected_elements) {
        throw std::invalid_argument("Expectation matrix element count does not match target qubit count.");
    }

    const std::vector<rocComplex> matrix_col_major = row_major_to_column_major(matrix, matrix_dim);
    rocComplex* d_matrix = nullptr;
    check_hip(hipMalloc(&d_matrix, matrix_col_major.size() * sizeof(rocComplex)),
              "expectation matrix allocation");
    rocComplex raw_result = to_roc_complex(std::complex<double>{0.0, 0.0});
    try {
        check_hip(hipMemcpy(d_matrix,
                            matrix_col_major.data(),
                            matrix_col_major.size() * sizeof(rocComplex),
                            hipMemcpyHostToDevice),
                  "expectation matrix upload");
        check_status(rocsvGetExpectationMatrix(sim_handle_,
                                               device_state_vector_,
                                               num_qubits_,
                                               targets.data(),
                                               static_cast<unsigned>(targets.size()),
                                               d_matrix,
                                               matrix_dim,
                                               &raw_result),
                     "matrix expectation");
    } catch (...) {
        hipFree(d_matrix);
        throw;
    }
    check_hip(hipFree(d_matrix), "expectation matrix free");

    return {static_cast<double>(raw_result.x), static_cast<double>(raw_result.y)};
}

std::vector<std::complex<double>> QuantumSimulator::expectation_matrix_batch(
    const std::vector<std::complex<double>>& matrix,
    const std::vector<unsigned>& targets) const {
    if (targets.empty()) {
        throw std::invalid_argument("expectation_matrix_batch requires at least one target qubit.");
    }
    if (targets.size() > 4) {
        throw std::runtime_error("expectation_matrix_batch currently supports at most 4 target qubits.");
    }

    std::unordered_set<unsigned> unique_targets;
    for (unsigned target : targets) {
        ensure_valid_qubit(target);
        if (!unique_targets.insert(target).second) {
            throw std::invalid_argument("expectation_matrix_batch target qubits must be unique.");
        }
    }

    const std::size_t matrix_dim = std::size_t{1} << targets.size();
    const std::size_t expected_elements = matrix_dim * matrix_dim;
    if (matrix.size() != expected_elements) {
        throw std::invalid_argument("Expectation matrix element count does not match target qubit count.");
    }

    const std::vector<rocComplex> matrix_col_major = row_major_to_column_major(matrix, matrix_dim);
    rocComplex* d_matrix = nullptr;
    check_hip(hipMalloc(&d_matrix, matrix_col_major.size() * sizeof(rocComplex)),
              "batched expectation matrix allocation");
    std::vector<rocComplex> raw_results(batch_size_, to_roc_complex(std::complex<double>{0.0, 0.0}));
    try {
        check_hip(hipMemcpy(d_matrix,
                            matrix_col_major.data(),
                            matrix_col_major.size() * sizeof(rocComplex),
                            hipMemcpyHostToDevice),
                  "batched expectation matrix upload");
        check_status(rocsvGetExpectationMatrixBatch(sim_handle_,
                                                    device_state_vector_,
                                                    num_qubits_,
                                                    targets.data(),
                                                    static_cast<unsigned>(targets.size()),
                                                    d_matrix,
                                                    matrix_dim,
                                                    raw_results.data()),
                     "batch matrix expectation");
    } catch (...) {
        hipFree(d_matrix);
        throw;
    }
    check_hip(hipFree(d_matrix), "batched expectation matrix free");

    std::vector<std::complex<double>> out(raw_results.size());
    for (std::size_t idx = 0; idx < raw_results.size(); ++idx) {
        out[idx] = {static_cast<double>(raw_results[idx].x), static_cast<double>(raw_results[idx].y)};
    }
    return out;
}

std::pair<std::complex<double>, std::complex<double>> QuantumSimulator::sparse_hamiltonian_moments(
    const std::vector<std::complex<double>>& data,
    const std::vector<std::size_t>& indices,
    const std::vector<std::size_t>& indptr,
    std::size_t rows,
    std::size_t cols) const {
    if (rows != state_vec_size_ || cols != state_vec_size_) {
        throw std::invalid_argument("Sparse Hamiltonian shape must match the simulator state dimension.");
    }
    if (indptr.size() != rows + 1) {
        throw std::invalid_argument("Sparse Hamiltonian CSR indptr length must equal rows + 1.");
    }
    if (data.size() != indices.size()) {
        throw std::invalid_argument("Sparse Hamiltonian CSR data and indices lengths must match.");
    }
    if (indptr.empty() || indptr.front() != 0 || indptr.back() != data.size()) {
        throw std::invalid_argument("Sparse Hamiltonian CSR indptr must start at 0 and end at nnz.");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        if (indptr[row] > indptr[row + 1]) {
            throw std::invalid_argument("Sparse Hamiltonian CSR indptr must be monotonic.");
        }
    }
    for (const std::size_t col : indices) {
        if (col >= cols) {
            throw std::invalid_argument("Sparse Hamiltonian CSR column index is out of bounds.");
        }
    }

    std::vector<rocComplex> raw_data;
    raw_data.reserve(data.size());
    for (const std::complex<double>& value : data) {
        raw_data.push_back(to_roc_complex(value));
    }

    rocComplex* d_data = nullptr;
    std::size_t* d_indices = nullptr;
    std::size_t* d_indptr = nullptr;
    auto free_sparse_buffers = [&]() {
        if (d_data) {
            check_hip(hipFree(d_data), "sparse Hamiltonian data free");
        }
        if (d_indices) {
            check_hip(hipFree(d_indices), "sparse Hamiltonian indices free");
        }
        if (d_indptr) {
            check_hip(hipFree(d_indptr), "sparse Hamiltonian indptr free");
        }
    };
    auto cleanup_after_error = [&]() {
        if (d_data) {
            hipFree(d_data);
        }
        if (d_indices) {
            hipFree(d_indices);
        }
        if (d_indptr) {
            hipFree(d_indptr);
        }
    };

    rocComplex raw_mean = to_roc_complex(std::complex<double>{0.0, 0.0});
    rocComplex raw_second_moment = to_roc_complex(std::complex<double>{0.0, 0.0});
    try {
        if (!raw_data.empty()) {
            check_hip(hipMalloc(&d_data, raw_data.size() * sizeof(rocComplex)),
                      "sparse Hamiltonian data allocation");
            check_hip(hipMemcpy(d_data,
                                raw_data.data(),
                                raw_data.size() * sizeof(rocComplex),
                                hipMemcpyHostToDevice),
                      "sparse Hamiltonian data upload");
        }
        if (!indices.empty()) {
            check_hip(hipMalloc(&d_indices, indices.size() * sizeof(std::size_t)),
                      "sparse Hamiltonian indices allocation");
            check_hip(hipMemcpy(d_indices,
                                indices.data(),
                                indices.size() * sizeof(std::size_t),
                                hipMemcpyHostToDevice),
                      "sparse Hamiltonian indices upload");
        }
        check_hip(hipMalloc(&d_indptr, indptr.size() * sizeof(std::size_t)),
                  "sparse Hamiltonian indptr allocation");
        check_hip(hipMemcpy(d_indptr,
                            indptr.data(),
                            indptr.size() * sizeof(std::size_t),
                            hipMemcpyHostToDevice),
                  "sparse Hamiltonian indptr upload");

        check_status(rocsvGetSparseMatrixMoments(sim_handle_,
                                                 device_state_vector_,
                                                 num_qubits_,
                                                 d_data,
                                                 d_indices,
                                                 d_indptr,
                                                 rows,
                                                 cols,
                                                 raw_data.size(),
                                                 &raw_mean,
                                                 &raw_second_moment),
                     "sparse Hamiltonian moments");
    } catch (...) {
        cleanup_after_error();
        throw;
    }
    free_sparse_buffers();

    return {{static_cast<double>(raw_mean.x), static_cast<double>(raw_mean.y)},
            {static_cast<double>(raw_second_moment.x), static_cast<double>(raw_second_moment.y)}};
}

std::pair<std::vector<std::complex<double>>, std::vector<std::complex<double>>> QuantumSimulator::sparse_hamiltonian_moments_batch(
    const std::vector<std::complex<double>>& data,
    const std::vector<std::size_t>& indices,
    const std::vector<std::size_t>& indptr,
    std::size_t rows,
    std::size_t cols) const {
    if (rows != state_vec_size_ || cols != state_vec_size_) {
        throw std::invalid_argument("Sparse Hamiltonian shape must match the simulator state dimension.");
    }
    if (indptr.size() != rows + 1) {
        throw std::invalid_argument("Sparse Hamiltonian CSR indptr length must equal rows + 1.");
    }
    if (data.size() != indices.size()) {
        throw std::invalid_argument("Sparse Hamiltonian CSR data and indices lengths must match.");
    }
    if (indptr.empty() || indptr.front() != 0 || indptr.back() != data.size()) {
        throw std::invalid_argument("Sparse Hamiltonian CSR indptr must start at 0 and end at nnz.");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        if (indptr[row] > indptr[row + 1]) {
            throw std::invalid_argument("Sparse Hamiltonian CSR indptr must be monotonic.");
        }
    }
    for (const std::size_t col : indices) {
        if (col >= cols) {
            throw std::invalid_argument("Sparse Hamiltonian CSR column index is out of bounds.");
        }
    }

    std::vector<rocComplex> raw_data;
    raw_data.reserve(data.size());
    for (const std::complex<double>& value : data) {
        raw_data.push_back(to_roc_complex(value));
    }

    rocComplex* d_data = nullptr;
    std::size_t* d_indices = nullptr;
    std::size_t* d_indptr = nullptr;
    auto free_sparse_buffers = [&]() {
        if (d_data) {
            check_hip(hipFree(d_data), "batched sparse Hamiltonian data free");
        }
        if (d_indices) {
            check_hip(hipFree(d_indices), "batched sparse Hamiltonian indices free");
        }
        if (d_indptr) {
            check_hip(hipFree(d_indptr), "batched sparse Hamiltonian indptr free");
        }
    };
    auto cleanup_after_error = [&]() {
        if (d_data) {
            hipFree(d_data);
        }
        if (d_indices) {
            hipFree(d_indices);
        }
        if (d_indptr) {
            hipFree(d_indptr);
        }
    };

    std::vector<rocComplex> raw_means(batch_size_, to_roc_complex(std::complex<double>{0.0, 0.0}));
    std::vector<rocComplex> raw_second_moments(batch_size_, to_roc_complex(std::complex<double>{0.0, 0.0}));
    try {
        if (!raw_data.empty()) {
            check_hip(hipMalloc(&d_data, raw_data.size() * sizeof(rocComplex)),
                      "batched sparse Hamiltonian data allocation");
            check_hip(hipMemcpy(d_data,
                                raw_data.data(),
                                raw_data.size() * sizeof(rocComplex),
                                hipMemcpyHostToDevice),
                      "batched sparse Hamiltonian data upload");
        }
        if (!indices.empty()) {
            check_hip(hipMalloc(&d_indices, indices.size() * sizeof(std::size_t)),
                      "batched sparse Hamiltonian indices allocation");
            check_hip(hipMemcpy(d_indices,
                                indices.data(),
                                indices.size() * sizeof(std::size_t),
                                hipMemcpyHostToDevice),
                      "batched sparse Hamiltonian indices upload");
        }
        check_hip(hipMalloc(&d_indptr, indptr.size() * sizeof(std::size_t)),
                  "batched sparse Hamiltonian indptr allocation");
        check_hip(hipMemcpy(d_indptr,
                            indptr.data(),
                            indptr.size() * sizeof(std::size_t),
                            hipMemcpyHostToDevice),
                  "batched sparse Hamiltonian indptr upload");

        check_status(rocsvGetSparseMatrixMomentsBatch(sim_handle_,
                                                      device_state_vector_,
                                                      num_qubits_,
                                                      d_data,
                                                      d_indices,
                                                      d_indptr,
                                                      rows,
                                                      cols,
                                                      raw_data.size(),
                                                      raw_means.data(),
                                                      raw_second_moments.data()),
                     "batch sparse Hamiltonian moments");
    } catch (...) {
        cleanup_after_error();
        throw;
    }
    free_sparse_buffers();

    std::vector<std::complex<double>> means(raw_means.size());
    std::vector<std::complex<double>> second_moments(raw_second_moments.size());
    for (std::size_t idx = 0; idx < raw_means.size(); ++idx) {
        means[idx] = {static_cast<double>(raw_means[idx].x), static_cast<double>(raw_means[idx].y)};
        second_moments[idx] = {
            static_cast<double>(raw_second_moments[idx].x),
            static_cast<double>(raw_second_moments[idx].y)};
    }
    return {means, second_moments};
}

unsigned QuantumSimulator::num_qubits() const noexcept {
    return num_qubits_;
}

std::size_t QuantumSimulator::batch_size() const noexcept {
    return batch_size_;
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

void QuantumSimulator::ApplyGateBatch(const std::string& gate_name,
                                      const std::vector<unsigned>& targets,
                                      const std::vector<double>& params_by_batch) {
    apply_gate_batch(gate_name, targets, params_by_batch);
}

void QuantumSimulator::ApplyControlledGate(const std::vector<std::complex<double>>& gate_matrix,
                                           int control_qubit,
                                           int target_qubit) {
    if (control_qubit < 0 || target_qubit < 0) {
        throw std::out_of_range("Qubit index out of bounds for simulator instance.");
    }
    apply_controlled_matrix(gate_matrix,
                            {static_cast<unsigned>(control_qubit)},
                            {static_cast<unsigned>(target_qubit)});
}

void QuantumSimulator::Execute() {
    synchronize();
}

int QuantumSimulator::MeasureQubit(int target_qubit) {
    if (target_qubit < 0) {
        throw std::out_of_range("Qubit index out of bounds for simulator instance.");
    }
    return measure_qubit(static_cast<unsigned>(target_qubit));
}

void QuantumSimulator::ResetQubit(int target_qubit) {
    if (target_qubit < 0) {
        throw std::out_of_range("Qubit index out of bounds for simulator instance.");
    }
    reset_qubit(static_cast<unsigned>(target_qubit));
}

std::vector<std::complex<double>> QuantumSimulator::GetStateVector() const {
    return get_statevector();
}

std::vector<std::complex<double>> QuantumSimulator::GetStateVectors() const {
    return get_statevectors();
}

std::vector<double> QuantumSimulator::Probabilities(const std::vector<unsigned>& qubits) const {
    return probabilities(qubits);
}

std::vector<double> QuantumSimulator::ProbabilitiesBatch(const std::vector<unsigned>& qubits) const {
    return probabilities_batch(qubits);
}

std::vector<long long> QuantumSimulator::MeasureBatch(const std::vector<unsigned>& qubits, int shots) {
    return measure_batch(qubits, shots);
}

double QuantumSimulator::GetExpectationValue(const std::string& pauli, int target_qubit) {
    if (target_qubit < 0) {
        throw std::out_of_range("Qubit index out of bounds for simulator instance.");
    }
    return expectation_value(pauli, static_cast<unsigned>(target_qubit));
}

double QuantumSimulator::GetExpectationPauliString(const std::string& pauli_string,
                                                   const std::vector<unsigned>& targets) {
    return expectation_pauli_string(pauli_string, targets);
}

std::vector<double> QuantumSimulator::GetExpectationPauliStringBatch(
    const std::string& pauli_string,
    const std::vector<unsigned>& targets) {
    return expectation_pauli_string_batch(pauli_string, targets);
}

std::complex<double> QuantumSimulator::ExpectationMatrix(
    const std::vector<std::complex<double>>& matrix,
    const std::vector<unsigned>& targets) const {
    return expectation_matrix(matrix, targets);
}

std::vector<std::complex<double>> QuantumSimulator::ExpectationMatrixBatch(
    const std::vector<std::complex<double>>& matrix,
    const std::vector<unsigned>& targets) const {
    return expectation_matrix_batch(matrix, targets);
}

std::pair<std::complex<double>, std::complex<double>> QuantumSimulator::SparseHamiltonianMoments(
    const std::vector<std::complex<double>>& data,
    const std::vector<std::size_t>& indices,
    const std::vector<std::size_t>& indptr,
    std::size_t rows,
    std::size_t cols) const {
    return sparse_hamiltonian_moments(data, indices, indptr, rows, cols);
}

std::pair<std::vector<std::complex<double>>, std::vector<std::complex<double>>> QuantumSimulator::SparseHamiltonianMomentsBatch(
    const std::vector<std::complex<double>>& data,
    const std::vector<std::size_t>& indices,
    const std::vector<std::size_t>& indptr,
    std::size_t rows,
    std::size_t cols) const {
    return sparse_hamiltonian_moments_batch(data, indices, indptr, rows, cols);
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
