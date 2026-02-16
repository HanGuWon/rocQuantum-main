#include "rocquantum/hipTensorNet.h"
#include "rocquantum/hipTensorNet_api.h"
#include "rocquantum/rocTensorUtil.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool nearly_equal(const rocComplex& a, const rocComplex& b, float tol = 1e-4f) {
    return std::fabs(a.x - b.x) < tol && std::fabs(a.y - b.y) < tol;
}

bool check_status(rocqStatus_t status, const char* what) {
    if (status != ROCQ_STATUS_SUCCESS) {
        std::cerr << what << " failed with status " << static_cast<int>(status) << '\n';
        return false;
    }
    return true;
}

bool check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::cerr << what << " failed: " << hipGetErrorString(err) << '\n';
        return false;
    }
    return true;
}

bool check_rocblas(rocblas_status status, const char* what) {
    if (status != rocblas_status_success) {
        std::cerr << what << " failed: " << static_cast<int>(status) << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    hipStream_t stream = nullptr;
    rocblas_handle blas = nullptr;
    rocTensorNetworkHandle_t tn = nullptr;

    if (!check_hip(hipStreamCreate(&stream), "hipStreamCreate")) {
        return 1;
    }
    if (!check_rocblas(rocblas_create_handle(&blas), "rocblas_create_handle")) {
        hipStreamDestroy(stream);
        return 1;
    }
    if (!check_rocblas(rocblas_set_stream(blas, stream), "rocblas_set_stream")) {
        rocblas_destroy_handle(blas);
        hipStreamDestroy(stream);
        return 1;
    }
    if (!check_status(rocTensorNetworkCreate(&tn, ROC_DATATYPE_C64), "rocTensorNetworkCreate")) {
        rocblas_destroy_handle(blas);
        hipStreamDestroy(stream);
        return 1;
    }

    rocquantum::util::rocTensor A;
    rocquantum::util::rocTensor B;
    rocquantum::util::rocTensor result;

    A.dimensions_ = {2, 3};
    A.labels_ = {"i", "k"};
    A.calculate_strides();
    B.dimensions_ = {3, 2};
    B.labels_ = {"k", "j"};
    B.calculate_strides();

    if (!check_status(rocquantum::util::rocTensorAllocate(&A), "rocTensorAllocate(A)") ||
        !check_status(rocquantum::util::rocTensorAllocate(&B), "rocTensorAllocate(B)")) {
        rocTensorNetworkDestroy(tn);
        rocblas_destroy_handle(blas);
        hipStreamDestroy(stream);
        return 1;
    }

    // Column-major storage.
    const std::vector<rocComplex> hA = {
        {1.0f, 0.0f}, {4.0f, 0.0f},
        {2.0f, 0.0f}, {5.0f, 0.0f},
        {3.0f, 0.0f}, {6.0f, 0.0f}};
    const std::vector<rocComplex> hB = {
        {7.0f, 0.0f}, {9.0f, 0.0f}, {11.0f, 0.0f},
        {8.0f, 0.0f}, {10.0f, 0.0f}, {12.0f, 0.0f}};

    if (!check_hip(hipMemcpyAsync(A.data_,
                                  hA.data(),
                                  hA.size() * sizeof(rocComplex),
                                  hipMemcpyHostToDevice,
                                  stream),
                   "hipMemcpyAsync(A)") ||
        !check_hip(hipMemcpyAsync(B.data_,
                                  hB.data(),
                                  hB.size() * sizeof(rocComplex),
                                  hipMemcpyHostToDevice,
                                  stream),
                   "hipMemcpyAsync(B)") ||
        !check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize")) {
        rocquantum::util::rocTensorFree(&A);
        rocquantum::util::rocTensorFree(&B);
        rocTensorNetworkDestroy(tn);
        rocblas_destroy_handle(blas);
        hipStreamDestroy(stream);
        return 1;
    }

    if (!check_status(rocTensorNetworkAddTensor(tn, &A), "rocTensorNetworkAddTensor(A)") ||
        !check_status(rocTensorNetworkAddTensor(tn, &B), "rocTensorNetworkAddTensor(B)")) {
        rocquantum::util::rocTensorFree(&A);
        rocquantum::util::rocTensorFree(&B);
        rocTensorNetworkDestroy(tn);
        rocblas_destroy_handle(blas);
        hipStreamDestroy(stream);
        return 1;
    }

    hipTensorNetContractionOptimizerConfig_t config{};
    config.pathfinder_algorithm = ROCTN_PATHFINDER_ALGO_GREEDY;
    config.memory_limit_bytes = 0;
    config.num_slices = 0;

    if (!check_status(rocTensorNetworkContract(tn, &config, &result, blas, stream),
                      "rocTensorNetworkContract") ||
        !check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(contract)")) {
        rocquantum::util::rocTensorFree(&A);
        rocquantum::util::rocTensorFree(&B);
        rocquantum::util::rocTensorFree(&result);
        rocTensorNetworkDestroy(tn);
        rocblas_destroy_handle(blas);
        hipStreamDestroy(stream);
        return 1;
    }

    if (result.dimensions_ != std::vector<long long>({2, 2})) {
        std::cerr << "Unexpected result shape.\n";
        rocquantum::util::rocTensorFree(&A);
        rocquantum::util::rocTensorFree(&B);
        rocquantum::util::rocTensorFree(&result);
        rocTensorNetworkDestroy(tn);
        rocblas_destroy_handle(blas);
        hipStreamDestroy(stream);
        return 1;
    }

    std::vector<rocComplex> hResult(4);
    if (!check_hip(hipMemcpy(hResult.data(),
                             result.data_,
                             hResult.size() * sizeof(rocComplex),
                             hipMemcpyDeviceToHost),
                   "hipMemcpy(result)")) {
        rocquantum::util::rocTensorFree(&A);
        rocquantum::util::rocTensorFree(&B);
        rocquantum::util::rocTensorFree(&result);
        rocTensorNetworkDestroy(tn);
        rocblas_destroy_handle(blas);
        hipStreamDestroy(stream);
        return 1;
    }

    const std::vector<rocComplex> expected = {
        {58.0f, 0.0f},
        {139.0f, 0.0f},
        {64.0f, 0.0f},
        {154.0f, 0.0f}};

    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!nearly_equal(hResult[i], expected[i])) {
            std::cerr << "Mismatch at " << i << ": got (" << hResult[i].x << ", " << hResult[i].y
                      << "), expected (" << expected[i].x << ", " << expected[i].y << ")\n";
            ok = false;
        }
    }

    rocquantum::util::rocTensorFree(&A);
    rocquantum::util::rocTensorFree(&B);
    rocquantum::util::rocTensorFree(&result);
    rocTensorNetworkDestroy(tn);
    rocblas_destroy_handle(blas);
    hipStreamDestroy(stream);

    return ok ? 0 : 1;
}
