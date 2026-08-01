#ifndef ROCQ_COMPILER_REFERENCE_STATEVEC_BACKEND_H
#define ROCQ_COMPILER_REFERENCE_STATEVEC_BACKEND_H

#include "QuantumBackend.h"

#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace rocq {

/// Small, dependency-free CPU state-vector backend.
///
/// This backend is intended as a deterministic compiler/runtime reference
/// target and numerical oracle.  It is deliberately not a high-performance
/// simulator.
class ReferenceStateVecBackend final : public QuantumBackend {
public:
    void initialize(unsigned num_qubits) override;
    void apply_gate(const std::string& gate_name,
                    const std::vector<unsigned>& targets) override;
    void apply_parametrized_gate(
        const std::string& gate_name,
        double parameter,
        const std::vector<unsigned>& targets) override;
    std::vector<std::complex<double>> get_state_vector() override;
    void destroy() override;

private:
    using Complex = std::complex<double>;

    void ensure_initialized() const;
    void validate_targets(const std::vector<unsigned>& targets,
                          std::size_t minimum,
                          std::size_t maximum,
                          const std::string& gate_name) const;
    void apply_single_qubit(
        unsigned target,
        Complex m00,
        Complex m01,
        Complex m10,
        Complex m11,
        std::size_t control_mask = 0);
    void apply_multi_controlled_x(const std::vector<unsigned>& controls,
                                  unsigned target);
    void apply_swap(unsigned first,
                    unsigned second,
                    std::size_t control_mask = 0);

    unsigned num_qubits_ = 0;
    std::vector<Complex> state_;
    bool initialized_ = false;
};

std::unique_ptr<QuantumBackend> create_reference_backend();

} // namespace rocq

#endif // ROCQ_COMPILER_REFERENCE_STATEVEC_BACKEND_H
