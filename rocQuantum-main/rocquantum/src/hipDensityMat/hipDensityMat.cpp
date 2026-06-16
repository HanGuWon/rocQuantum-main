// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

#include "hipDensityMat.hpp"
#include "hipDensityMat_internal.hpp"

#include <hip/hip_complex.h>
#include <algorithm>
#include <cstdint>
#include <cmath> // For sqrt
#include <limits>
#include <random>
#include <stdexcept>
#include <vector> // For host-side reduction

// Helper to check HIP API calls for errors.
inline hipDensityMatStatus_t check_hip_error(hipError_t err) {
    if (err != hipSuccess) {
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

/**
 * @brief GPU kernel to apply a single-qubit Kraus operator: ρ' = KρK†.
 */
__global__ void apply_single_qubit_kraus_kernel(
    hipComplex* rho_out,
    const hipComplex* rho_in,
    const hipComplex* K,
    int num_qubits,
    int target_qubit)
{
    const int64_t dim = 1LL << num_qubits;
    const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    const int64_t low_mask = (1LL << target_qubit) - 1;
    const int64_t high_mask = ~((1LL << (target_qubit + 1)) - 1);
    const int64_t row_low = row & low_mask, col_low = col & low_mask;
    const int64_t row_high = row & high_mask, col_high = col & high_mask;
    const int row_t = (row >> target_qubit) & 1, col_t = (col >> target_qubit) & 1;

    const int64_t k0 = row_high | (0 << target_qubit) | row_low;
    const int64_t k1 = row_high | (1 << target_qubit) | row_low;
    const int64_t l0 = col_high | (0 << target_qubit) | col_low;
    const int64_t l1 = col_high | (1 << target_qubit) | col_low;

    hipComplex result = make_hipFloatComplex(0.0f, 0.0f);
    for (int kt = 0; kt < 2; ++kt) {
        for (int lt = 0; lt < 2; ++lt) {
            const int64_t k = (kt == 0) ? k0 : k1;
            const int64_t l = (lt == 0) ? l0 : l1;
            hipComplex K_rowt_kt = K[row_t * 2 + kt];
            hipComplex rho_kl = rho_in[k * dim + l];
            hipComplex K_colt_lt_conj = hipConjf(K[col_t * 2 + lt]);
            hipComplex term = hipCmulf(K_rowt_kt, rho_kl);
            term = hipCmulf(term, K_colt_lt_conj);
            result = hipCaddf(result, term);
        }
    }
    rho_out[row * dim + col] = result;
}

__global__ void apply_multi_qubit_kraus_kernel(
    hipComplex* rho_out,
    const hipComplex* rho_in,
    const hipComplex* K,
    const int* target_qubits,
    int num_targets,
    int num_qubits)
{
    const int64_t dim = 1LL << num_qubits;
    const int target_dim = 1 << num_targets;
    const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    int64_t row_base = row;
    int64_t col_base = col;
    int row_target = 0;
    int col_target = 0;
    for (int bit = 0; bit < num_targets; ++bit) {
        const int qubit = target_qubits[bit];
        const int64_t mask = 1LL << qubit;
        if (row & mask) row_target |= (1 << bit);
        if (col & mask) col_target |= (1 << bit);
        row_base &= ~mask;
        col_base &= ~mask;
    }

    hipComplex result = make_hipFloatComplex(0.0f, 0.0f);
    for (int kt = 0; kt < target_dim; ++kt) {
        int64_t k = row_base;
        for (int bit = 0; bit < num_targets; ++bit) {
            if ((kt >> bit) & 1) {
                k |= (1LL << target_qubits[bit]);
            }
        }

        for (int lt = 0; lt < target_dim; ++lt) {
            int64_t l = col_base;
            for (int bit = 0; bit < num_targets; ++bit) {
                if ((lt >> bit) & 1) {
                    l |= (1LL << target_qubits[bit]);
                }
            }

            hipComplex k_row = K[row_target * target_dim + kt];
            hipComplex rho_kl = rho_in[k * dim + l];
            hipComplex k_col_conj = hipConjf(K[col_target * target_dim + lt]);
            hipComplex term = hipCmulf(k_row, rho_kl);
            term = hipCmulf(term, k_col_conj);
            result = hipCaddf(result, term);
        }
    }
    rho_out[row * dim + col] = result;
}

/**
 * @brief GPU kernel for element-wise addition: target += source.
 */
__global__ void accumulate_kernel(hipComplex* target, const hipComplex* source, int64_t num_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        target[idx] = hipCaddf(target[idx], source[idx]);
    }
}

__global__ void extract_density_diagonal_kernel(const hipComplex* rho, double* diagonal, int64_t dim)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        diagonal[idx] = static_cast<double>(rho[idx * dim + idx].x);
    }
}

__global__ void accumulate_density_marginal_probabilities_kernel(
    const hipComplex* rho,
    double* outcome_probs,
    int64_t dim,
    const int* measured_qubits,
    int num_measured_qubits)
{
    int64_t basis = blockIdx.x * blockDim.x + threadIdx.x;
    if (basis >= dim) {
        return;
    }

    size_t outcome = 0;
    for (int m = 0; m < num_measured_qubits; ++m) {
        if ((basis >> measured_qubits[m]) & 1LL) {
            outcome |= (size_t{1} << m);
        }
    }

    const size_t diagonal_index =
        static_cast<size_t>(basis) * static_cast<size_t>(dim) + static_cast<size_t>(basis);
    const double prob = static_cast<double>(rho[diagonal_index].x);
    if (prob > 0.0) {
        atomicAdd(&outcome_probs[outcome], prob);
    }
}

__global__ void density_matrix_expectation_matrix_kernel(
    const hipComplex* rho,
    const hipComplex* matrix,
    const int* target_qubits,
    int num_target_qubits,
    int64_t dim,
    int matrix_dim,
    hipComplex* block_sums)
{
    extern __shared__ double shared[];
    double* real_sums = shared;
    double* imag_sums = shared + blockDim.x;

    const unsigned int tid = threadIdx.x;
    const int64_t row_start = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    double local_re = 0.0;
    double local_im = 0.0;
    for (int64_t row_index = row_start; row_index < dim; row_index += stride) {
        int row_target = 0;
        int64_t base_index = row_index;
        for (int bit = 0; bit < num_target_qubits; ++bit) {
            const int qubit = target_qubits[bit];
            const int64_t mask = int64_t{1} << qubit;
            if ((row_index & mask) != 0) {
                row_target |= (1 << bit);
            }
            base_index &= ~mask;
        }

        for (int col_target = 0; col_target < matrix_dim; ++col_target) {
            int64_t col_index = base_index;
            for (int bit = 0; bit < num_target_qubits; ++bit) {
                if ((col_target >> bit) & 1) {
                    col_index |= (int64_t{1} << target_qubits[bit]);
                }
            }

            const hipComplex m = matrix[row_target * matrix_dim + col_target];
            const hipComplex rho_value =
                rho[static_cast<size_t>(col_index) * static_cast<size_t>(dim) + static_cast<size_t>(row_index)];
            local_re += static_cast<double>(m.x) * static_cast<double>(rho_value.x) -
                        static_cast<double>(m.y) * static_cast<double>(rho_value.y);
            local_im += static_cast<double>(m.x) * static_cast<double>(rho_value.y) +
                        static_cast<double>(m.y) * static_cast<double>(rho_value.x);
        }
    }

    real_sums[tid] = local_re;
    imag_sums[tid] = local_im;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            real_sums[tid] += real_sums[tid + s];
            imag_sums[tid] += imag_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] =
            make_hipFloatComplex(static_cast<float>(real_sums[0]), static_cast<float>(imag_sums[0]));
    }
}

namespace {

bool checked_element_count_bytes(int64_t element_count, size_t element_size, size_t* size_bytes)
{
    if (size_bytes == nullptr || element_count < 0) {
        return false;
    }
    const uint64_t max_size = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    const uint64_t count = static_cast<uint64_t>(element_count);
    if (element_size != 0 && count > max_size / element_size) {
        return false;
    }
    *size_bytes = static_cast<size_t>(count) * element_size;
    return true;
}

bool checked_density_dimension(int num_qubits, int64_t* dim)
{
    if (dim == nullptr || num_qubits <= 0 || num_qubits > HIPDENSITYMAT_MAX_QUBITS) {
        return false;
    }
    if (num_qubits >= std::numeric_limits<int64_t>::digits) {
        return false;
    }
    *dim = int64_t{1} << num_qubits;
    return true;
}

bool checked_density_storage_size(
    int num_qubits,
    int64_t* dim,
    int64_t* num_elements,
    size_t* size_bytes)
{
    if (num_elements == nullptr) {
        return false;
    }

    int64_t local_dim = 0;
    if (!checked_density_dimension(num_qubits, &local_dim)) {
        return false;
    }
    if (local_dim > std::numeric_limits<int64_t>::max() / local_dim) {
        return false;
    }

    const int64_t local_num_elements = local_dim * local_dim;
    if (!checked_element_count_bytes(local_num_elements, sizeof(hipComplex), size_bytes)) {
        return false;
    }

    if (dim != nullptr) {
        *dim = local_dim;
    }
    *num_elements = local_num_elements;
    return true;
}

bool density_state_size_bytes(const hipDensityMatState* internal_state, size_t* size_bytes)
{
    return internal_state != nullptr &&
           checked_element_count_bytes(internal_state->num_elements_, sizeof(hipComplex), size_bytes);
}

hipDensityMatStatus_t apply_kraus_channel(
    hipDensityMatState* internal_state,
    int target_qubit,
    const hipComplex* kraus_matrices_host,
    int num_kraus)
{
    if (internal_state == nullptr || kraus_matrices_host == nullptr || num_kraus <= 0) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    int64_t dim = 0;
    size_t size_bytes = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim) ||
        !density_state_size_bytes(internal_state, &size_bytes)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    hipComplex* accumulator_rho_device = nullptr;
    hipComplex* temp_rho_device = nullptr;
    hipComplex* kraus_matrix_device = nullptr;

    hipError_t hip_err = hipMalloc(&accumulator_rho_device, size_bytes);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;

    hip_err = hipMalloc(&temp_rho_device, size_bytes);
    if (hip_err != hipSuccess) {
        hipFree(accumulator_rho_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMalloc(&kraus_matrix_device, 4 * sizeof(hipComplex));
    if (hip_err != hipSuccess) {
        hipFree(accumulator_rho_device);
        hipFree(temp_rho_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMemset(accumulator_rho_device, 0, size_bytes);
    if (hip_err == hipSuccess) {
        dim3 blockDim2D(16, 16);
        dim3 gridDim2D((dim + blockDim2D.x - 1) / blockDim2D.x,
                       (dim + blockDim2D.y - 1) / blockDim2D.y);
        dim3 blockDim1D(256);
        dim3 gridDim1D((internal_state->num_elements_ + blockDim1D.x - 1) / blockDim1D.x);

        for (int k = 0; k < num_kraus && hip_err == hipSuccess; ++k) {
            hip_err = hipMemcpy(kraus_matrix_device,
                                kraus_matrices_host + (4 * k),
                                4 * sizeof(hipComplex),
                                hipMemcpyHostToDevice);
            if (hip_err != hipSuccess) break;

            hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
                temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
                kraus_matrix_device, internal_state->num_qubits_, target_qubit);
            hip_err = hipGetLastError();
            if (hip_err != hipSuccess) break;

            hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
                accumulator_rho_device, temp_rho_device, internal_state->num_elements_);
            hip_err = hipGetLastError();
        }
    }

    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) {
        hip_err = hipMemcpy(internal_state->device_data_, accumulator_rho_device, size_bytes, hipMemcpyDeviceToDevice);
    }

    hipFree(accumulator_rho_device);
    hipFree(temp_rho_device);
    hipFree(kraus_matrix_device);

    return check_hip_error(hip_err);
}

hipDensityMatStatus_t apply_multi_qubit_kraus_channel(
    hipDensityMatState* internal_state,
    const int* target_qubits_host,
    int num_targets,
    const hipComplex* kraus_matrices_host,
    int num_kraus)
{
    if (internal_state == nullptr || target_qubits_host == nullptr ||
        kraus_matrices_host == nullptr || num_targets <= 0 || num_kraus <= 0) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    if (num_targets > internal_state->num_qubits_) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    if (num_targets > HIPDENSITYMAT_MAX_KRAUS_TARGETS) {
        return HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED;
    }

    for (int i = 0; i < num_targets; ++i) {
        const int qubit = target_qubits_host[i];
        if (qubit < 0 || qubit >= internal_state->num_qubits_) {
            return HIPDENSITYMAT_STATUS_INVALID_VALUE;
        }
        for (int j = i + 1; j < num_targets; ++j) {
            if (qubit == target_qubits_host[j]) {
                return HIPDENSITYMAT_STATUS_INVALID_VALUE;
            }
        }
    }

    if (num_targets == 1) {
        return apply_kraus_channel(internal_state, target_qubits_host[0], kraus_matrices_host, num_kraus);
    }

    int64_t dim = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    const int target_dim = 1 << num_targets;
    const size_t kraus_size = static_cast<size_t>(target_dim) * static_cast<size_t>(target_dim);
    size_t size_bytes = 0;
    if (!density_state_size_bytes(internal_state, &size_bytes)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    hipComplex* accumulator_rho_device = nullptr;
    hipComplex* temp_rho_device = nullptr;
    hipComplex* kraus_matrix_device = nullptr;
    int* target_qubits_device = nullptr;

    hipError_t hip_err = hipMalloc(&accumulator_rho_device, size_bytes);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;

    hip_err = hipMalloc(&temp_rho_device, size_bytes);
    if (hip_err != hipSuccess) {
        hipFree(accumulator_rho_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMalloc(&kraus_matrix_device, kraus_size * sizeof(hipComplex));
    if (hip_err != hipSuccess) {
        hipFree(accumulator_rho_device);
        hipFree(temp_rho_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMalloc(&target_qubits_device, static_cast<size_t>(num_targets) * sizeof(int));
    if (hip_err != hipSuccess) {
        hipFree(accumulator_rho_device);
        hipFree(temp_rho_device);
        hipFree(kraus_matrix_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMemcpy(
        target_qubits_device,
        target_qubits_host,
        static_cast<size_t>(num_targets) * sizeof(int),
        hipMemcpyHostToDevice);
    if (hip_err == hipSuccess) {
        hip_err = hipMemset(accumulator_rho_device, 0, size_bytes);
    }

    if (hip_err == hipSuccess) {
        dim3 blockDim2D(16, 16);
        dim3 gridDim2D((dim + blockDim2D.x - 1) / blockDim2D.x,
                       (dim + blockDim2D.y - 1) / blockDim2D.y);
        dim3 blockDim1D(256);
        dim3 gridDim1D((internal_state->num_elements_ + blockDim1D.x - 1) / blockDim1D.x);

        for (int k = 0; k < num_kraus && hip_err == hipSuccess; ++k) {
            hip_err = hipMemcpy(
                kraus_matrix_device,
                kraus_matrices_host + (kraus_size * static_cast<size_t>(k)),
                kraus_size * sizeof(hipComplex),
                hipMemcpyHostToDevice);
            if (hip_err != hipSuccess) break;

            hipLaunchKernelGGL(apply_multi_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
                temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
                kraus_matrix_device, target_qubits_device, num_targets, internal_state->num_qubits_);
            hip_err = hipGetLastError();
            if (hip_err != hipSuccess) break;

            hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
                accumulator_rho_device, temp_rho_device, internal_state->num_elements_);
            hip_err = hipGetLastError();
        }
    }

    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) {
        hip_err = hipMemcpy(internal_state->device_data_, accumulator_rho_device, size_bytes, hipMemcpyDeviceToDevice);
    }

    hipFree(accumulator_rho_device);
    hipFree(temp_rho_device);
    hipFree(kraus_matrix_device);
    hipFree(target_qubits_device);

    return check_hip_error(hip_err);
}

hipDensityMatStatus_t validate_measured_qubits(
    const int* measured_qubits,
    int num_measured_qubits,
    int num_qubits)
{
    if (measured_qubits == nullptr || num_measured_qubits <= 0) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    if (num_measured_qubits > num_qubits) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    if (num_measured_qubits > HIPDENSITYMAT_MAX_SAMPLE_QUBITS) {
        return HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED;
    }

    for (int i = 0; i < num_measured_qubits; ++i) {
        if (measured_qubits[i] < 0 || measured_qubits[i] >= num_qubits) {
            return HIPDENSITYMAT_STATUS_INVALID_VALUE;
        }
        for (int j = i + 1; j < num_measured_qubits; ++j) {
            if (measured_qubits[i] == measured_qubits[j]) {
                return HIPDENSITYMAT_STATUS_INVALID_VALUE;
            }
        }
    }
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipDensityMatStatus_t copy_density_diagonal_to_host(
    hipDensityMatState* internal_state,
    std::vector<double>& diagonal_host)
{
    if (internal_state == nullptr) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    int64_t dim = 0;
    size_t diagonal_bytes = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim) ||
        !checked_element_count_bytes(dim, sizeof(double), &diagonal_bytes)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    diagonal_host.assign(static_cast<size_t>(dim), 0.0);

    double* diagonal_device = nullptr;
    hipError_t hip_err = hipMalloc(&diagonal_device, diagonal_bytes);
    if (hip_err != hipSuccess) {
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    const int block_size = 256;
    const int grid_size = static_cast<int>((dim + block_size - 1) / block_size);
    hipLaunchKernelGGL(
        extract_density_diagonal_kernel,
        grid_size,
        block_size,
        0,
        internal_state->stream_,
        static_cast<const hipComplex*>(internal_state->device_data_),
        diagonal_device,
        dim);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) {
        hip_err = hipStreamSynchronize(internal_state->stream_);
    }
    if (hip_err == hipSuccess) {
        hip_err = hipMemcpy(
            diagonal_host.data(),
            diagonal_device,
            diagonal_bytes,
            hipMemcpyDeviceToHost);
    }

    hipFree(diagonal_device);
    return check_hip_error(hip_err);
}

hipDensityMatStatus_t compute_density_marginal_probabilities(
    hipDensityMatState* internal_state,
    const int* measured_qubits,
    int num_measured_qubits,
    std::vector<double>& outcome_probs_host)
{
    if (internal_state == nullptr || measured_qubits == nullptr || num_measured_qubits <= 0) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    int64_t dim = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    const size_t num_outcomes = size_t{1} << num_measured_qubits;
    outcome_probs_host.assign(num_outcomes, 0.0);

    int* measured_qubits_device = nullptr;
    hipError_t hip_err = hipMalloc(
        &measured_qubits_device,
        static_cast<size_t>(num_measured_qubits) * sizeof(int));
    if (hip_err != hipSuccess) {
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    double* outcome_probs_device = nullptr;
    hip_err = hipMalloc(&outcome_probs_device, num_outcomes * sizeof(double));
    if (hip_err != hipSuccess) {
        hipFree(measured_qubits_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMemcpy(
        measured_qubits_device,
        measured_qubits,
        static_cast<size_t>(num_measured_qubits) * sizeof(int),
        hipMemcpyHostToDevice);
    if (hip_err == hipSuccess) {
        hip_err = hipMemset(outcome_probs_device, 0, num_outcomes * sizeof(double));
    }
    if (hip_err == hipSuccess) {
        const int block_size = 256;
        const int grid_size = static_cast<int>((dim + block_size - 1) / block_size);
        hipLaunchKernelGGL(
            accumulate_density_marginal_probabilities_kernel,
            grid_size,
            block_size,
            0,
            internal_state->stream_,
            static_cast<const hipComplex*>(internal_state->device_data_),
            outcome_probs_device,
            dim,
            measured_qubits_device,
            num_measured_qubits);
        hip_err = hipGetLastError();
    }
    if (hip_err == hipSuccess) {
        hip_err = hipStreamSynchronize(internal_state->stream_);
    }
    if (hip_err == hipSuccess) {
        hip_err = hipMemcpy(
            outcome_probs_host.data(),
            outcome_probs_device,
            num_outcomes * sizeof(double),
            hipMemcpyDeviceToHost);
    }

    hipFree(outcome_probs_device);
    hipFree(measured_qubits_device);
    return check_hip_error(hip_err);
}

}  // namespace

/**
 * @brief GPU kernel to compute partial sums for Tr(Oρ) for a single block.
 */
__global__ void expectation_value_kernel(
    const hipComplex* rho,
    double* partial_sums,
    int num_qubits,
    int target_qubit,
    hipDensityMatPauli_t pauli_op)
{
    extern __shared__ double sdata[];

    const int64_t dim = 1LL << num_qubits;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const int64_t i_start = blockIdx.x * block_size + tid;
    const int64_t stride = gridDim.x * block_size;

    double thread_sum = 0.0;

    for (int64_t i = i_start; i < dim; i += stride) {
        double term = 0.0;
        int64_t bit_mask = 1LL << target_qubit;
        int64_t i_flipped = i ^ bit_mask;

        switch (pauli_op) {
            case HIPDENSITYMAT_PAULI_Z: {
                double sign = ((i & bit_mask) == 0) ? 1.0 : -1.0;
                term = sign * rho[i * dim + i].x;
                break;
            }
            case HIPDENSITYMAT_PAULI_X: {
                term = rho[i_flipped * dim + i].x;
                break;
            }
            case HIPDENSITYMAT_PAULI_Y: {
                double sign = ((i & bit_mask) == 0) ? 1.0 : -1.0;
                term = -sign * rho[i_flipped * dim + i].y;
                break;
            }
        }
        thread_sum += term;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}


hipDensityMatStatus_t hipDensityMatCreateState(hipDensityMatState_t* state, int num_qubits) {
    if (state == nullptr || num_qubits <= 0 || num_qubits > HIPDENSITYMAT_MAX_QUBITS) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    
    hipDensityMatState* internal_state = nullptr;
    try {
        internal_state = new hipDensityMatState();
    } catch (const std::bad_alloc&) {
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    int64_t num_elements = 0;
    size_t size_bytes = 0;
    if (!checked_density_storage_size(num_qubits, nullptr, &num_elements, &size_bytes)) {
        delete internal_state;
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    internal_state->num_qubits_ = num_qubits;
    internal_state->num_elements_ = num_elements;

    if (hipMalloc(&internal_state->device_data_, size_bytes) != hipSuccess) {
        delete internal_state;
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }
    if (hipMemset(internal_state->device_data_, 0, size_bytes) != hipSuccess) {
        hipFree(internal_state->device_data_);
        delete internal_state;
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    hipComplex val_one = make_hipFloatComplex(1.0f, 0.0f);
    if (hipMemcpy(internal_state->device_data_, &val_one, sizeof(hipComplex), hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(internal_state->device_data_);
        delete internal_state;
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    internal_state->stream_ = 0;
    *state = internal_state;
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipDensityMatStatus_t hipDensityMatDestroyState(hipDensityMatState_t state) {
    if (state == nullptr) return HIPDENSITYMAT_STATUS_SUCCESS;
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (internal_state->device_data_ != nullptr) {
        hipFree(internal_state->device_data_);
    }
    delete internal_state;
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipDensityMatStatus_t hipDensityMatApplyKrausOperator(
    hipDensityMatState_t state,
    int target_qubit,
    const hipComplex* kraus_matrix_host)
{
    if (state == nullptr || kraus_matrix_host == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    return apply_kraus_channel(internal_state, target_qubit, kraus_matrix_host, 1);
}

hipDensityMatStatus_t hipDensityMatApplyBitFlipChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    if (probability < 0.0 || probability > 1.0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    const float p0 = sqrt(1.0 - probability);
    const float p1 = sqrt(probability);
    hipComplex kraus[8] = {
        make_hipFloatComplex(p0, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p0, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p1, 0.0f),
        make_hipFloatComplex(p1, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
    };
    return apply_kraus_channel(internal_state, target_qubit, kraus, 2);
}

hipDensityMatStatus_t hipDensityMatApplyPhaseFlipChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    if (probability < 0.0 || probability > 1.0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    const float p0 = sqrt(1.0 - probability);
    const float p1 = sqrt(probability);
    hipComplex kraus[8] = {
        make_hipFloatComplex(p0, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p0, 0.0f),
        make_hipFloatComplex(p1, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(-p1, 0.0f),
    };
    return apply_kraus_channel(internal_state, target_qubit, kraus, 2);
}

hipDensityMatStatus_t hipDensityMatApplyDepolarizingChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    if (probability < 0.0 || probability > 1.0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    const float p0 = sqrt(1.0 - probability);
    const float p = sqrt(probability / 3.0);
    hipComplex kraus[16] = {
        make_hipFloatComplex(p0, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p0, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p, 0.0f),
        make_hipFloatComplex(p, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(0.0f, -p),
        make_hipFloatComplex(0.0f, p), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(p, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(-p, 0.0f),
    };
    return apply_kraus_channel(internal_state, target_qubit, kraus, 4);
}

hipDensityMatStatus_t hipDensityMatComputeExpectation(
    hipDensityMatState_t state,
    int target_qubit,
    hipDensityMatPauli_t pauli_op,
    double* result_host)
{
    if (state == nullptr || result_host == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    int64_t dim = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    hipError_t hip_err;

    const int block_size = 256;
    const int num_blocks = std::min((int)((dim + block_size - 1) / block_size), 2048);
    
    double* partial_sums_device = nullptr;
    size_t partial_sums_size = num_blocks * sizeof(double);
    hip_err = hipMalloc(&partial_sums_device, partial_sums_size);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;

    hipLaunchKernelGGL(expectation_value_kernel,
        num_blocks,
        block_size,
        block_size * sizeof(double),
        internal_state->stream_,
        static_cast<const hipComplex*>(internal_state->device_data_),
        partial_sums_device,
        internal_state->num_qubits_,
        target_qubit,
        pauli_op);

    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err != hipSuccess) {
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    std::vector<double> partial_sums_host(num_blocks);
    hip_err = hipMemcpy(partial_sums_host.data(), partial_sums_device, partial_sums_size, hipMemcpyDeviceToHost);
    if (hip_err != hipSuccess) {
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        total_sum += partial_sums_host[i];
    }
    
    *result_host = total_sum;

    hipFree(partial_sums_device);

    return HIPDENSITYMAT_STATUS_SUCCESS;
}

/**
 * @brief GPU kernel to compute partial sums for Tr((Z_i Z_j...)ρ).
 */
__global__ void pauli_z_product_expectation_kernel(
    const hipComplex* rho,
    double* partial_sums,
    int num_qubits,
    int num_z_qubits,
    const int* z_qubit_indices)
{
    extern __shared__ double sdata[];

    const int64_t dim = 1LL << num_qubits;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const int64_t i_start = blockIdx.x * block_size + tid;
    const int64_t stride = gridDim.x * block_size;

    double thread_sum = 0.0;

    for (int64_t i = i_start; i < dim; i += stride) {
        int parity = 0;
        for (int k = 0; k < num_z_qubits; ++k) {
            if ((i >> z_qubit_indices[k]) & 1) {
                parity++;
            }
        }
        double sign = (parity % 2 == 0) ? 1.0 : -1.0;
        thread_sum += sign * rho[i * dim + i].x;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

hipDensityMatStatus_t hipDensityMatComputePauliZProductExpectation(
    hipDensityMatState_t state,
    int num_z_qubits,
    const int* z_qubit_indices_host,
    double* result_host)
{
    if (state == nullptr || result_host == nullptr || (num_z_qubits > 0 && z_qubit_indices_host == nullptr)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    for (int i = 0; i < num_z_qubits; ++i) {
        if (z_qubit_indices_host[i] < 0 || z_qubit_indices_host[i] >= internal_state->num_qubits_) {
            return HIPDENSITYMAT_STATUS_INVALID_VALUE;
        }
    }

    // Handle the identity case (no Z operators)
    if (num_z_qubits == 0) {
        *result_host = 1.0; // Trace of a density matrix is 1
        return HIPDENSITYMAT_STATUS_SUCCESS;
    }

    int64_t dim = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    hipError_t hip_err;

    int* z_qubit_indices_device = nullptr;
    size_t indices_size = num_z_qubits * sizeof(int);
    hip_err = hipMalloc(&z_qubit_indices_device, indices_size);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    hip_err = hipMemcpy(z_qubit_indices_device, z_qubit_indices_host, indices_size, hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) {
        hipFree(z_qubit_indices_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    const int block_size = 256;
    const int num_blocks = std::min((int)((dim + block_size - 1) / block_size), 2048);
    
    double* partial_sums_device = nullptr;
    size_t partial_sums_size = num_blocks * sizeof(double);
    hip_err = hipMalloc(&partial_sums_device, partial_sums_size);
    if (hip_err != hipSuccess) {
        hipFree(z_qubit_indices_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hipLaunchKernelGGL(pauli_z_product_expectation_kernel,
        num_blocks,
        block_size,
        block_size * sizeof(double),
        internal_state->stream_,
        static_cast<const hipComplex*>(internal_state->device_data_),
        partial_sums_device,
        internal_state->num_qubits_,
        num_z_qubits,
        z_qubit_indices_device);

    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        hipFree(z_qubit_indices_device);
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err != hipSuccess) {
        hipFree(z_qubit_indices_device);
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    std::vector<double> partial_sums_host(num_blocks);
    hip_err = hipMemcpy(partial_sums_host.data(), partial_sums_device, partial_sums_size, hipMemcpyDeviceToHost);
    if (hip_err != hipSuccess) {
        hipFree(z_qubit_indices_device);
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        total_sum += partial_sums_host[i];
    }
    
    *result_host = total_sum;

    hipFree(z_qubit_indices_device);
    hipFree(partial_sums_device);

    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipDensityMatStatus_t hipDensityMatComputeExpectationMatrix(
    hipDensityMatState_t state,
    const int* target_qubits_host,
    int num_target_qubits,
    const hipComplex* matrix_host,
    int matrix_dim,
    hipComplex* result_host)
{
    if (state == nullptr || target_qubits_host == nullptr || matrix_host == nullptr ||
        result_host == nullptr || num_target_qubits <= 0 || matrix_dim <= 0) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    const hipDensityMatStatus_t target_status =
        validate_measured_qubits(target_qubits_host, num_target_qubits, internal_state->num_qubits_);
    if (target_status != HIPDENSITYMAT_STATUS_SUCCESS) {
        return target_status;
    }
    if (num_target_qubits > HIPDENSITYMAT_MAX_DENSE_OBSERVABLE_TARGETS) {
        return HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED;
    }
    if (matrix_dim != (1 << num_target_qubits)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    int64_t dim = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    const size_t matrix_elements = static_cast<size_t>(matrix_dim) * static_cast<size_t>(matrix_dim);

    int* target_qubits_device = nullptr;
    hipError_t hip_err = hipMalloc(
        &target_qubits_device,
        static_cast<size_t>(num_target_qubits) * sizeof(int));
    if (hip_err != hipSuccess) {
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hipComplex* matrix_device = nullptr;
    hip_err = hipMalloc(&matrix_device, matrix_elements * sizeof(hipComplex));
    if (hip_err != hipSuccess) {
        hipFree(target_qubits_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    const int block_size = 256;
    const int num_blocks = std::min(static_cast<int>((dim + block_size - 1) / block_size), 2048);
    hipComplex* block_sums_device = nullptr;
    hip_err = hipMalloc(&block_sums_device, static_cast<size_t>(num_blocks) * sizeof(hipComplex));
    if (hip_err != hipSuccess) {
        hipFree(matrix_device);
        hipFree(target_qubits_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMemcpy(
        target_qubits_device,
        target_qubits_host,
        static_cast<size_t>(num_target_qubits) * sizeof(int),
        hipMemcpyHostToDevice);
    if (hip_err == hipSuccess) {
        hip_err = hipMemcpy(
            matrix_device,
            matrix_host,
            matrix_elements * sizeof(hipComplex),
            hipMemcpyHostToDevice);
    }
    if (hip_err == hipSuccess) {
        hipLaunchKernelGGL(
            density_matrix_expectation_matrix_kernel,
            num_blocks,
            block_size,
            2 * block_size * sizeof(double),
            internal_state->stream_,
            static_cast<const hipComplex*>(internal_state->device_data_),
            matrix_device,
            target_qubits_device,
            num_target_qubits,
            dim,
            matrix_dim,
            block_sums_device);
        hip_err = hipGetLastError();
    }
    if (hip_err == hipSuccess) {
        hip_err = hipStreamSynchronize(internal_state->stream_);
    }

    std::vector<hipComplex> block_sums_host(static_cast<size_t>(num_blocks));
    if (hip_err == hipSuccess) {
        hip_err = hipMemcpy(
            block_sums_host.data(),
            block_sums_device,
            static_cast<size_t>(num_blocks) * sizeof(hipComplex),
            hipMemcpyDeviceToHost);
    }

    hipFree(block_sums_device);
    hipFree(matrix_device);
    hipFree(target_qubits_device);
    if (hip_err != hipSuccess) {
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    double total_re = 0.0;
    double total_im = 0.0;
    for (const hipComplex& block_sum : block_sums_host) {
        total_re += static_cast<double>(block_sum.x);
        total_im += static_cast<double>(block_sum.y);
    }
    *result_host = make_hipFloatComplex(static_cast<float>(total_re), static_cast<float>(total_im));
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipDensityMatStatus_t hipDensityMatApplyAmplitudeDampingChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double gamma)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    if (gamma < 0.0 || gamma > 1.0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    hipComplex kraus[8] = {
        make_hipFloatComplex(1.0f, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(sqrt(1.0 - gamma), 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(sqrt(gamma), 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
    };
    return apply_kraus_channel(internal_state, target_qubit, kraus, 2);
}

hipDensityMatStatus_t hipDensityMatApplyGate(
    hipDensityMatState_t state,
    int target_qubit,
    const hipComplex* gate_matrix_host)
{
    if (state == nullptr || gate_matrix_host == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    int64_t dim = 0;
    size_t size_bytes = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim) ||
        !density_state_size_bytes(internal_state, &size_bytes)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    
    hipComplex* gate_matrix_device = nullptr;
    hipComplex* rho_out_device = nullptr;
    hipError_t hip_err;

    hip_err = hipMalloc(&gate_matrix_device, 4 * sizeof(hipComplex));
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    hip_err = hipMalloc(&rho_out_device, size_bytes);
    if (hip_err != hipSuccess) {
        hipFree(gate_matrix_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMemcpy(gate_matrix_device, gate_matrix_host, 4 * sizeof(hipComplex), hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) {
        hipFree(gate_matrix_device);
        hipFree(rho_out_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((dim + blockDim.x - 1) / blockDim.x, (dim + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim, blockDim, 0, internal_state->stream_,
        rho_out_device, static_cast<const hipComplex*>(internal_state->device_data_),
        gate_matrix_device, internal_state->num_qubits_, target_qubit);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, rho_out_device, size_bytes, hipMemcpyDeviceToDevice);
    
    hipFree(gate_matrix_device);
    hipFree(rho_out_device);
    return check_hip_error(hip_err);
}

/**
 * @brief GPU kernel to apply a CNOT gate: ρ' = UρU†.
 */
__global__ void apply_cnot_kernel(
    hipComplex* rho_out,
    const hipComplex* rho_in,
    int num_qubits,
    int control_qubit,
    int target_qubit)
{
    const int64_t dim = 1LL << num_qubits;
    const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    const int64_t control_mask = 1LL << control_qubit;
    const int64_t target_mask = 1LL << target_qubit;

    // Source indices are calculated by applying the CNOT transformation
    // to the destination indices (row, col), since CNOT is its own inverse.
    int64_t source_row = row;
    if ((row & control_mask) != 0) {
        source_row ^= target_mask;
    }

    int64_t source_col = col;
    if ((col & control_mask) != 0) {
        source_col ^= target_mask;
    }

    rho_out[row * dim + col] = rho_in[source_row * dim + source_col];
}

hipDensityMatStatus_t hipDensityMatApplyCNOT(
    hipDensityMatState_t state,
    int control_qubit,
    int target_qubit)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (control_qubit < 0 || control_qubit >= internal_state->num_qubits_ ||
        target_qubit < 0 || target_qubit >= internal_state->num_qubits_ ||
        control_qubit == target_qubit) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    int64_t dim = 0;
    size_t size_bytes = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim) ||
        !density_state_size_bytes(internal_state, &size_bytes)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    
    hipComplex* rho_out_device = nullptr;
    hipError_t hip_err;

    hip_err = hipMalloc(&rho_out_device, size_bytes);
    if (hip_err != hipSuccess) {
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((dim + blockDim.x - 1) / blockDim.x, (dim + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(apply_cnot_kernel, gridDim, blockDim, 0, internal_state->stream_,
        rho_out_device, static_cast<const hipComplex*>(internal_state->device_data_),
        internal_state->num_qubits_, control_qubit, target_qubit);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, rho_out_device, size_bytes, hipMemcpyDeviceToDevice);
    
    hipFree(rho_out_device);
    return check_hip_error(hip_err);
}

/**
 * @brief GPU kernel to apply a generic controlled single-qubit gate: ρ' = UρU†.
 */
__global__ void apply_controlled_gate_kernel(
    hipComplex* rho_out,
    const hipComplex* rho_in,
    const hipComplex* U, // 2x2 gate matrix for target qubit
    int num_qubits,
    int control_qubit,
    int target_qubit)
{
    const int64_t dim = 1LL << num_qubits;
    const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    const int64_t control_mask = 1LL << control_qubit;
    const bool is_control_in_row = ((row & control_mask) != 0);
    const bool is_control_in_col = ((col & control_mask) != 0);

    const int64_t target_low_mask = (1LL << target_qubit) - 1;
    const int64_t target_high_mask = ~((1LL << (target_qubit + 1)) - 1);

    const int64_t row_low = row & target_low_mask;
    const int64_t row_high = row & target_high_mask;
    const int row_t = (row >> target_qubit) & 1;

    const int64_t col_low = col & target_low_mask;
    const int64_t col_high = col & target_high_mask;
    const int col_t = (col >> target_qubit) & 1;

    hipComplex result = make_hipFloatComplex(0.0f, 0.0f);

    for (int kt = 0; kt < 2; ++kt) {
        for (int lt = 0; lt < 2; ++lt) {
            const int64_t k_base = (row_high & ~control_mask) | (kt << target_qubit) | row_low;
            const int64_t l_base = (col_high & ~control_mask) | (lt << target_qubit) | col_low;

            hipComplex term = make_hipFloatComplex(0.0f, 0.0f);

            // Term 1: Control bit is 0 for both k and l
            int64_t k0 = k_base;
            int64_t l0 = l_base;
            hipComplex rho_kl0 = rho_in[k0 * dim + l0];
            if (row_t == kt && col_t == lt) { // Identity applied
                term = hipCaddf(term, rho_kl0);
            }

            // Term 2: Control bit is 1 for both k and l
            int64_t k1 = k_base | control_mask;
            int64_t l1 = l_base | control_mask;
            hipComplex rho_kl1 = rho_in[k1 * dim + l1];
            hipComplex U_rowt_kt = U[row_t * 2 + kt];
            hipComplex U_colt_lt_conj = hipConjf(U[col_t * 2 + lt]);
            hipComplex u_rho_u = hipCmulf(U_rowt_kt, rho_kl1);
            u_rho_u = hipCmulf(u_rho_u, U_colt_lt_conj);
            term = hipCaddf(term, u_rho_u);
            
            // This simplified logic is only correct if control qubit is not target qubit
            // and row/col high/low masks don't overlap with control bit, which is true.
            // The full transformation involves 4 terms from U_full = |0><0|⊗I + |1><1|⊗U_target
            // but cross-terms like <0|ρ|1> are zero if control bit is separated.
            // Here we assume the control qubit state is not mixed with target operations.
            // A more general kernel would handle all 4 blocks of the full density matrix.
            // This implementation correctly handles cases where the control qubit subspace is diagonal.
            if (is_control_in_row == is_control_in_col) {
                 if (is_control_in_row) { // Control is 1
                    hipComplex U_rowt_kt = U[row_t * 2 + kt];
                    hipComplex U_colt_lt_conj = hipConjf(U[col_t * 2 + lt]);
                    int64_t k = k_base | control_mask;
                    int64_t l = l_base | control_mask;
                    hipComplex u_rho_u = hipCmulf(U_rowt_kt, rho_in[k * dim + l]);
                    u_rho_u = hipCmulf(u_rho_u, U_colt_lt_conj);
                    result = hipCaddf(result, u_rho_u);
                 } else { // Control is 0
                    if (row_t == kt && col_t == lt) {
                        int64_t k = k_base;
                        int64_t l = l_base;
                        result = hipCaddf(result, rho_in[k * dim + l]);
                    }
                 }
            } else { // Off-diagonal blocks related to the control qubit
                // U_full = |0><0|I + |1><1|U_target
                // For <0|ρ'|1>, the term is I * <0|ρ|1> * U_target^dagger
                // For <1|ρ'|0>, the term is U_target * <1|ρ|0> * I
                if (is_control_in_row) { // <1|ρ'|0> block
                    hipComplex U_rowt_kt = U[row_t * 2 + kt];
                    int64_t k = k_base | control_mask; // k has control bit 1
                    int64_t l = l_base;               // l has control bit 0
                    if (col_t == lt) { // I_colt_lt is delta
                        hipComplex u_rho = hipCmulf(U_rowt_kt, rho_in[k * dim + l]);
                        result = hipCaddf(result, u_rho);
                    }
                } else { // <0|ρ'|1> block
                    hipComplex U_colt_lt_conj = hipConjf(U[col_t * 2 + lt]);
                    int64_t k = k_base;               // k has control bit 0
                    int64_t l = l_base | control_mask; // l has control bit 1
                    if (row_t == kt) { // I_rowt_kt is delta
                        hipComplex rho_u_dag = hipCmulf(rho_in[k * dim + l], U_colt_lt_conj);
                        result = hipCaddf(result, rho_u_dag);
                    }
                }
            }
        }
    }
     rho_out[row * dim + col] = result;
}


hipDensityMatStatus_t hipDensityMatApplyControlledGate(
    hipDensityMatState_t state,
    int control_qubit,
    int target_qubit,
    const hipComplex* gate_matrix_device)
{
    if (state == nullptr || gate_matrix_device == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (control_qubit < 0 || control_qubit >= internal_state->num_qubits_ ||
        target_qubit < 0 || target_qubit >= internal_state->num_qubits_ ||
        control_qubit == target_qubit) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    int64_t dim = 0;
    size_t size_bytes = 0;
    if (!checked_density_dimension(internal_state->num_qubits_, &dim) ||
        !density_state_size_bytes(internal_state, &size_bytes)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    
    hipComplex* rho_out_device = nullptr;
    hipError_t hip_err;

    hip_err = hipMalloc(&rho_out_device, size_bytes);
    if (hip_err != hipSuccess) {
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((dim + blockDim.x - 1) / blockDim.x, (dim + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(apply_controlled_gate_kernel, gridDim, blockDim, 0, internal_state->stream_,
        rho_out_device, static_cast<const hipComplex*>(internal_state->device_data_),
        gate_matrix_device, internal_state->num_qubits_, control_qubit, target_qubit);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, rho_out_device, size_bytes, hipMemcpyDeviceToDevice);
    
    hipFree(rho_out_device);
    return check_hip_error(hip_err);
}

hipDensityMatStatus_t hipDensityMatApplyChannel(hipDensityMatState_t state, int target_qubit, const void* channel_params) {
    if (state == nullptr || channel_params == nullptr) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    const hipDensityMatChannel_t* channel = static_cast<const hipDensityMatChannel_t*>(channel_params);
    if (channel->num_kraus <= 0 || channel->kraus_matrices_host == nullptr) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (channel->num_targets > 0 && channel->target_qubits_host != nullptr) {
        return apply_multi_qubit_kraus_channel(
            internal_state,
            channel->target_qubits_host,
            channel->num_targets,
            channel->kraus_matrices_host,
            channel->num_kraus);
    }

    return apply_kraus_channel(internal_state, target_qubit, channel->kraus_matrices_host, channel->num_kraus);
}

hipDensityMatStatus_t hipDensityMatSample(
    hipDensityMatState_t state,
    const int* measured_qubits,
    int num_measured_qubits,
    int num_shots,
    uint64_t* results_host)
{
    if (state == nullptr || num_shots < 0 || (num_shots > 0 && results_host == nullptr)) {
        return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    }
    if (num_shots == 0) {
        return HIPDENSITYMAT_STATUS_SUCCESS;
    }

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    hipDensityMatStatus_t status = validate_measured_qubits(
        measured_qubits,
        num_measured_qubits,
        internal_state->num_qubits_);
    if (status != HIPDENSITYMAT_STATUS_SUCCESS) {
        return status;
    }

    std::vector<double> outcome_probs;
    status = compute_density_marginal_probabilities(
        internal_state,
        measured_qubits,
        num_measured_qubits,
        outcome_probs);
    if (status != HIPDENSITYMAT_STATUS_SUCCESS) {
        return status;
    }

    double total_prob = 0.0;
    for (double prob : outcome_probs) {
        total_prob += prob;
    }
    if (total_prob <= std::numeric_limits<double>::epsilon()) {
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::discrete_distribution<uint64_t> dist(outcome_probs.begin(), outcome_probs.end());
    for (int shot = 0; shot < num_shots; ++shot) {
        results_host[shot] = dist(rng);
    }

    return HIPDENSITYMAT_STATUS_SUCCESS;
}
