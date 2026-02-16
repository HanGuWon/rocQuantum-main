#ifndef ROCQUANTUM_HIP_DENSITY_MAT_H
#define ROCQUANTUM_HIP_DENSITY_MAT_H

#include <hip/hip_runtime.h>

#include "rocquantum/hipStateVec.h"

typedef struct hipDensityMatState* rocdmHandle_t;

typedef enum {
    ROCDM_PAULI_X = 0,
    ROCDM_PAULI_Y = 1,
    ROCDM_PAULI_Z = 2
} rocdmPauli_t;

#ifdef __cplusplus
extern "C" {
#endif

rocqStatus_t rocdmCreateState(rocdmHandle_t* state, int num_qubits);
rocqStatus_t rocdmDestroyState(rocdmHandle_t state);

rocqStatus_t rocdmApplyBitFlipChannel(rocdmHandle_t state, int target_qubit, double probability);
rocqStatus_t rocdmApplyPhaseFlipChannel(rocdmHandle_t state, int target_qubit, double probability);
rocqStatus_t rocdmApplyDepolarizingChannel(rocdmHandle_t state, int target_qubit, double probability);
rocqStatus_t rocdmApplyAmplitudeDampingChannel(rocdmHandle_t state, int target_qubit, double gamma);

rocqStatus_t rocdmApplyGate(rocdmHandle_t state, int target_qubit, const hipComplex* gate_matrix_host);
rocqStatus_t rocdmApplyCNOT(rocdmHandle_t state, int control_qubit, int target_qubit);

rocqStatus_t rocdmComputeExpectation(rocdmHandle_t state,
                                     int target_qubit,
                                     rocdmPauli_t pauli,
                                     double* result_host);

rocqStatus_t rocdmComputePauliZProductExpectation(rocdmHandle_t state,
                                                  int num_z_qubits,
                                                  const int* z_qubit_indices,
                                                  double* result_host);

rocqStatus_t rocdmApplyChannel(rocdmHandle_t state, int target_qubit, const void* channel_params);

#ifdef __cplusplus
}
#endif

#endif

