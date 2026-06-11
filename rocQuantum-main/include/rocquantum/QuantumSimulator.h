#pragma once

#include <complex>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "rocquantum/hipStateVec.h"

namespace rocquantum {

class QuantumSimulator {
public:
    explicit QuantumSimulator(unsigned num_qubits, std::size_t batch_size = 1);
    ~QuantumSimulator();

    void reset();
    void reset_qubit(unsigned target);
    void apply_gate(const std::string& gate_name,
                    const std::vector<unsigned>& targets,
                    const std::vector<double>& params = {});
    void apply_gate_batch(const std::string& gate_name,
                          const std::vector<unsigned>& targets,
                          const std::vector<double>& params_by_batch);
    void apply_matrix(const std::vector<std::complex<double>>& matrix,
                      const std::vector<unsigned>& targets);
    void apply_controlled_matrix(const std::vector<std::complex<double>>& matrix,
                                 const std::vector<unsigned>& controls,
                                 const std::vector<unsigned>& targets);
    void set_statevector(const std::vector<std::complex<double>>& state);
    void set_statevectors(const std::vector<std::complex<double>>& states);
    std::vector<std::complex<double>> get_statevector(std::size_t batch_index = 0) const;
    std::vector<std::complex<double>> get_statevectors() const;
    std::vector<double> probabilities(const std::vector<unsigned>& qubits) const;
    std::vector<double> probabilities_batch(const std::vector<unsigned>& qubits) const;
    std::vector<long long> measure(const std::vector<unsigned>& qubits, int shots);
    std::vector<long long> measure_batch(const std::vector<unsigned>& qubits, int shots);
    double expectation_value(const std::string& pauli, unsigned target);
    double expectation_pauli_string(const std::string& pauli_string,
                                    const std::vector<unsigned>& targets);
    std::vector<double> expectation_pauli_string_batch(const std::string& pauli_string,
                                                       const std::vector<unsigned>& targets);
    std::complex<double> expectation_matrix(const std::vector<std::complex<double>>& matrix,
                                            const std::vector<unsigned>& targets) const;
    std::vector<std::complex<double>> expectation_matrix_batch(
        const std::vector<std::complex<double>>& matrix,
        const std::vector<unsigned>& targets) const;
    std::pair<std::complex<double>, std::complex<double>> sparse_hamiltonian_moments(
        const std::vector<std::complex<double>>& data,
        const std::vector<std::size_t>& indices,
        const std::vector<std::size_t>& indptr,
        std::size_t rows,
        std::size_t cols) const;
    std::pair<std::vector<std::complex<double>>, std::vector<std::complex<double>>> sparse_hamiltonian_moments_batch(
        const std::vector<std::complex<double>>& data,
        const std::vector<std::size_t>& indices,
        const std::vector<std::size_t>& indptr,
        std::size_t rows,
        std::size_t cols) const;
    unsigned num_qubits() const noexcept;
    std::size_t batch_size() const noexcept;

    // Legacy API retained for compatibility with previous QSim bindings.
    void ApplyGate(const std::string& gate_name, int target_qubit);
    void ApplyGate(const std::string& gate_name, int control_qubit, int target_qubit);
    void ApplyGate(const std::vector<std::complex<double>>& gate_matrix, int target_qubit);
    void ApplyGateBatch(const std::string& gate_name,
                        const std::vector<unsigned>& targets,
                        const std::vector<double>& params_by_batch);
    void ApplyControlledGate(const std::vector<std::complex<double>>& gate_matrix,
                             int control_qubit,
                             int target_qubit);
    void Execute();
    void ResetQubit(int target_qubit);
    std::vector<std::complex<double>> GetStateVector() const;
    std::vector<std::complex<double>> GetStateVectors() const;
    std::vector<double> Probabilities(const std::vector<unsigned>& qubits) const;
    std::vector<double> ProbabilitiesBatch(const std::vector<unsigned>& qubits) const;
    std::vector<long long> MeasureBatch(const std::vector<unsigned>& qubits, int shots);
    double GetExpectationValue(const std::string& pauli, int target_qubit);
    double GetExpectationPauliString(const std::string& pauli_string,
                                     const std::vector<unsigned>& targets);
    std::vector<double> GetExpectationPauliStringBatch(const std::string& pauli_string,
                                                       const std::vector<unsigned>& targets);
    std::complex<double> ExpectationMatrix(const std::vector<std::complex<double>>& matrix,
                                           const std::vector<unsigned>& targets) const;
    std::vector<std::complex<double>> ExpectationMatrixBatch(
        const std::vector<std::complex<double>>& matrix,
        const std::vector<unsigned>& targets) const;
    std::pair<std::complex<double>, std::complex<double>> SparseHamiltonianMoments(
        const std::vector<std::complex<double>>& data,
        const std::vector<std::size_t>& indices,
        const std::vector<std::size_t>& indptr,
        std::size_t rows,
        std::size_t cols) const;
    std::pair<std::vector<std::complex<double>>, std::vector<std::complex<double>>> SparseHamiltonianMomentsBatch(
        const std::vector<std::complex<double>>& data,
        const std::vector<std::size_t>& indices,
        const std::vector<std::size_t>& indptr,
        std::size_t rows,
        std::size_t cols) const;

private:
    void ensure_valid_qubit(unsigned qubit) const;
    void synchronize() const;

    unsigned num_qubits_;
    std::size_t batch_size_;
    size_t state_vec_size_;
    rocsvHandle_t sim_handle_;
    rocComplex* device_state_vector_;
};

using QSim = QuantumSimulator;

} // namespace rocquantum
