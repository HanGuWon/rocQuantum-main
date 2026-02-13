#ifndef GATEFUSION_H
#define GATEFUSION_H

#include "hipStateVec.h"
#include <cstddef>
#include <vector>
#include <string>
#include <variant>

namespace rocquantum {

// Represents a single gate operation for the fusion queue
struct GateOp {
    std::string name;
    std::vector<unsigned> targets;
    std::vector<unsigned> controls;
    std::vector<double> params;
};

class GateFusion {
public:
    GateFusion(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits);
    ~GateFusion();

    // Processes a queue of gates, applying fusion where possible
    rocqStatus_t processQueue(const std::vector<GateOp>& queue);

private:
    rocsvHandle_t handle_;
    rocComplex* d_state_;
    unsigned numQubits_;
    rocComplex* d_fused_matrix_buffer_ = nullptr;
    size_t fused_matrix_buffer_bytes_ = 0;

    // A simple fusion strategy for single-qubit gates
    rocqStatus_t fuseAndApplySingleQubitGates(const std::vector<GateOp>& gate_chunk);
    rocqStatus_t ensure_fused_matrix_buffer(size_t required_bytes);
};

} // namespace rocquantum

#endif // GATEFUSION_H
