// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#ifndef HIPDENSITYMAT_HPP
#define HIPDENSITYMAT_HPP

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle representing the density matrix state on the GPU.
 */
typedef struct hipDensityMatState* hipDensityMatState_t;

/**
 * @brief Status codes returned by the hipDensityMat API functions.
 */
typedef enum {
    HIPDENSITYMAT_STATUS_SUCCESS = 0,
    HIPDENSITYMAT_STATUS_ALLOC_FAILED = 1,
    HIPDENSITYMAT_STATUS_INVALID_VALUE = 2,
    HIPDENSITYMAT_STATUS_EXECUTION_FAILED = 3,
    HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED = 4
} hipDensityMatStatus_t;

typedef struct {
    int num_kraus;
    const hipComplex* kraus_matrices_host;
    int num_targets;
    const int* target_qubits_host;
} hipDensityMatChannel_t;

/**
 * @brief Creates and initializes a density matrix state for a given number of qubits.
 *
 * The density matrix is initialized to the |0...0><0...0| pure state.
 *
 * @param[out] state Pointer to the handle to store the created state.
 * @param[in] num_qubits The number of qubits in the quantum system.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatCreateState(hipDensityMatState_t* state, int num_qubits);

/**
 * @brief Destroys a density matrix state and frees all associated resources.
 *
 * @param[in] state The handle to the state to be destroyed.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatDestroyState(hipDensityMatState_t state);

/**
 * @brief Applies a single-qubit Kraus operator to the density matrix.
 *
 * This is the fundamental operation ρ' = KρK† for a target qubit.
 *
 * @param[in] state The state handle.
 * @param[in] target_qubit The index of the qubit to apply the operator to.
 * @param[in] kraus_matrix_host Pointer to a 2x2 hipComplex matrix on the host.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyKrausOperator(
    hipDensityMatState_t state,
    int target_qubit,
    const hipComplex* kraus_matrix_host);

/**
 * @brief Applies a single-qubit Bit-Flip noise channel to the density matrix.
 *
 * This simulates a bit-flip error (X gate) with a given probability.
 *
 * @param[in] state The state handle.
 * @param[in] target_qubit The index of the qubit to apply the channel to.
 * @param[in] probability The probability (p) of a bit-flip error occurring.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyBitFlipChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability);

/**
 * @brief Applies a single-qubit Phase-Flip noise channel to the density matrix.
 *
 * This simulates a phase-flip error (Z gate) with a given probability.
 *
 * @param[in] state The state handle.
 * @param[in] target_qubit The index of the qubit to apply the channel to.
 * @param[in] probability The probability (p) of a phase-flip error occurring.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyPhaseFlipChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability);

/**
 * @brief Applies a single-qubit Depolarizing noise channel to the density matrix.
 *
 * This simulates the state decohering towards the maximally mixed state.
 *
 * @param[in] state The state handle.
 * @param[in] target_qubit The index of the qubit to apply the channel to.
 * @param[in] probability The probability (p) of depolarization occurring.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyDepolarizingChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability);

/**
 * @brief Enum identifying single-qubit Pauli operators.
 */
typedef enum {
    HIPDENSITYMAT_PAULI_X,
    HIPDENSITYMAT_PAULI_Y,
    HIPDENSITYMAT_PAULI_Z
} hipDensityMatPauli_t;

/**
 * @brief Computes the expectation value of a single-qubit Pauli observable.
 *
 * Calculates <O> = Tr(Oρ) for a given Pauli operator O on a target qubit.
 *
 * @param[in] state The state handle.
 * @param[in] target_qubit The index of the qubit to measure.
 * @param[in] pauli_op The Pauli operator (X, Y, or Z) to measure.
 * @param[out] result_host Pointer to a double on the host to store the result.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatComputeExpectation(
    hipDensityMatState_t state,
    int target_qubit,
    hipDensityMatPauli_t pauli_op,
    double* result_host);

/**
 * @brief Computes the expectation value of a multi-qubit Pauli Z product observable.
 *
 * Calculates <Z_i Z_j ...> = Tr((Z_i ⊗ Z_j ⊗ ...)ρ) for a given set of qubits.
 *
 * @param[in] state The state handle.
 * @param[in] num_z_qubits The number of qubits in the Pauli Z product.
 * @param[in] z_qubit_indices Array of qubit indices to apply the Z operator to.
 * @param[out] result_host Pointer to a double on the host to store the result.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatComputePauliZProductExpectation(
    hipDensityMatState_t state,
    int num_z_qubits,
    const int* z_qubit_indices,
    double* result_host);

/**
 * @brief Applies a single-qubit Amplitude Damping noise channel.
 *
 * This channel models energy dissipation, e.g., a |1> state decaying to |0>.
 *
 * @param[in] state The state handle.
 * @param[in] target_qubit The index of the qubit to apply the channel to.
 * @param[in] gamma The probability of the qubit relaxing from |1> to |0>.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyAmplitudeDampingChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double gamma);

/**
 * @brief Applies an ideal (noiseless) single-qubit gate to the density matrix.
 *
 * The operation is ρ' = UρU†, where U is the provided gate matrix.
 *
 * @param[in] state The state handle.
 * @param[in] target_qubit The index of the qubit to apply the gate to.
 * @param[in] gate_matrix_host Pointer to a 2x2 unitary hipComplex matrix on the host.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyGate(
    hipDensityMatState_t state,
    int target_qubit,
    const hipComplex* gate_matrix_host);



/**
 * @brief Applies an ideal (noiseless) CNOT gate to the density matrix.
 *
 * The operation is ρ' = UρU†, where U is the CNOT gate matrix.
 *
 * @param[in] state The state handle.
 * @param[in] control_qubit The index of the control qubit.
 * @param[in] target_qubit The index of the target qubit.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyCNOT(
    hipDensityMatState_t state,
    int control_qubit,
    int target_qubit);

/**
 * @brief Applies a generic controlled single-qubit gate to the density matrix.
 *
 * The operation is ρ' = UρU†, where U is the controlled-gate matrix.
 * The single-qubit gate is applied to the target qubit if the control qubit is in state |1>.
 *
 * @param[in] state The state handle.
 * @param[in] control_qubit The index of the control qubit.
 * @param[in] target_qubit The index of the target qubit.
 * @param[in] gate_matrix_device Pointer to a 2x2 unitary hipComplex matrix on the device.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyControlledGate(
    hipDensityMatState_t state,
    int control_qubit,
    int target_qubit,
    const hipComplex* gate_matrix_device);



/**
 * @brief Applies a generic Kraus channel to one or more target qubits.
 *
 * For legacy single-qubit callers, channel_params may point to a
 * hipDensityMatChannel_t with num_targets <= 0 or target_qubits_host == nullptr;
 * target_qubit is then used and kraus_matrices_host is interpreted as
 * num_kraus contiguous 2x2 row-major matrices.
 *
 * For multi-qubit channels, target_qubits_host supplies num_targets unique
 * target qubits and kraus_matrices_host is interpreted as num_kraus contiguous
 * (2^num_targets)x(2^num_targets) row-major matrices. The target-qubit order
 * defines the local matrix bit order.
 *
 * @param[in] state The state handle.
 * @param[in] target_qubit Legacy single-qubit target when channel_params does not provide target_qubits_host.
 * @param[in] channel_params Pointer to a hipDensityMatChannel_t.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatApplyChannel(hipDensityMatState_t state, int target_qubit, const void* channel_params);

/**
 * @brief Samples computational-basis outcomes from the density-matrix diagonal.
 *
 * This correctness path reduces measured-qubit marginal probabilities on the
 * GPU, copies only the outcome probability vector to host memory, then draws
 * samples on the host.
 *
 * @param[in] state The state handle.
 * @param[in] measured_qubits Array of qubits to sample.
 * @param[in] num_measured_qubits Number of measured qubits.
 * @param[in] num_shots Number of samples to draw.
 * @param[out] results_host Host array of num_shots uint64_t outcomes.
 * @return hipDensityMatStatus_t Status of the operation.
 */
hipDensityMatStatus_t hipDensityMatSample(
    hipDensityMatState_t state,
    const int* measured_qubits,
    int num_measured_qubits,
    int num_shots,
    uint64_t* results_host);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIPDENSITYMAT_HPP
