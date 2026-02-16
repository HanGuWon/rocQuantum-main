#include "rocquantum/hipDensityMat.h"

#include "hipDensityMat.hpp"

namespace {

rocqStatus_t map_density_status(hipDensityMatStatus_t status) {
    switch (status) {
        case HIPDENSITYMAT_STATUS_SUCCESS:
            return ROCQ_STATUS_SUCCESS;
        case HIPDENSITYMAT_STATUS_ALLOC_FAILED:
            return ROCQ_STATUS_ALLOCATION_FAILED;
        case HIPDENSITYMAT_STATUS_INVALID_VALUE:
            return ROCQ_STATUS_INVALID_VALUE;
        case HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED:
            return ROCQ_STATUS_NOT_IMPLEMENTED;
        case HIPDENSITYMAT_STATUS_EXECUTION_FAILED:
            return ROCQ_STATUS_HIP_ERROR;
        default:
            return ROCQ_STATUS_FAILURE;
    }
}

hipDensityMatPauli_t to_density_pauli(rocdmPauli_t pauli) {
    switch (pauli) {
        case ROCDM_PAULI_X:
            return HIPDENSITYMAT_PAULI_X;
        case ROCDM_PAULI_Y:
            return HIPDENSITYMAT_PAULI_Y;
        case ROCDM_PAULI_Z:
            return HIPDENSITYMAT_PAULI_Z;
        default:
            return HIPDENSITYMAT_PAULI_Z;
    }
}

}  // namespace

extern "C" {

rocqStatus_t rocdmCreateState(rocdmHandle_t* state, int num_qubits) {
    return map_density_status(hipDensityMatCreateState(state, num_qubits));
}

rocqStatus_t rocdmDestroyState(rocdmHandle_t state) {
    return map_density_status(hipDensityMatDestroyState(state));
}

rocqStatus_t rocdmApplyBitFlipChannel(rocdmHandle_t state, int target_qubit, double probability) {
    return map_density_status(hipDensityMatApplyBitFlipChannel(state, target_qubit, probability));
}

rocqStatus_t rocdmApplyPhaseFlipChannel(rocdmHandle_t state, int target_qubit, double probability) {
    return map_density_status(hipDensityMatApplyPhaseFlipChannel(state, target_qubit, probability));
}

rocqStatus_t rocdmApplyDepolarizingChannel(rocdmHandle_t state, int target_qubit, double probability) {
    return map_density_status(hipDensityMatApplyDepolarizingChannel(state, target_qubit, probability));
}

rocqStatus_t rocdmApplyAmplitudeDampingChannel(rocdmHandle_t state, int target_qubit, double gamma) {
    return map_density_status(hipDensityMatApplyAmplitudeDampingChannel(state, target_qubit, gamma));
}

rocqStatus_t rocdmApplyGate(rocdmHandle_t state, int target_qubit, const hipComplex* gate_matrix_host) {
    return map_density_status(hipDensityMatApplyGate(state, target_qubit, gate_matrix_host));
}

rocqStatus_t rocdmApplyCNOT(rocdmHandle_t state, int control_qubit, int target_qubit) {
    return map_density_status(hipDensityMatApplyCNOT(state, control_qubit, target_qubit));
}

rocqStatus_t rocdmComputeExpectation(rocdmHandle_t state,
                                     int target_qubit,
                                     rocdmPauli_t pauli,
                                     double* result_host) {
    return map_density_status(
        hipDensityMatComputeExpectation(state, target_qubit, to_density_pauli(pauli), result_host));
}

rocqStatus_t rocdmComputePauliZProductExpectation(rocdmHandle_t state,
                                                  int num_z_qubits,
                                                  const int* z_qubit_indices,
                                                  double* result_host) {
    return map_density_status(
        hipDensityMatComputePauliZProductExpectation(state, num_z_qubits, z_qubit_indices, result_host));
}

rocqStatus_t rocdmApplyChannel(rocdmHandle_t state, int target_qubit, const void* channel_params) {
    return map_density_status(hipDensityMatApplyChannel(state, target_qubit, channel_params));
}

}  // extern "C"

