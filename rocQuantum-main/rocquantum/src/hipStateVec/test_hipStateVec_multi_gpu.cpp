#include "rocquantum/hipStateVec.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kCtestSkipReturnCode = 77;

bool nearly_equal(const rocComplex& a, const rocComplex& b, double tol = 1e-5) {
    return std::abs(static_cast<double>(a.x - b.x)) < tol &&
           std::abs(static_cast<double>(a.y - b.y)) < tol;
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

bool expect_state(rocsvHandle_t handle, unsigned num_qubits, const std::vector<rocComplex>& expected) {
    std::vector<rocComplex> host(expected.size());
    if (!check_status(rocsvGetStateVectorFull(handle, nullptr, host.data()), "rocsvGetStateVectorFull")) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!nearly_equal(host[i], expected[i])) {
            std::cerr << "State mismatch at " << i << ": got (" << host[i].x << ", " << host[i].y
                      << "), expected (" << expected[i].x << ", " << expected[i].y << ")\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const bool require_multi_gpu =
        argc == 2 && std::string(argv[1]) == "--require-multi-gpu";
    if (argc > 2 || (argc == 2 && !require_multi_gpu)) {
        std::cerr << "Usage: " << argv[0] << " [--require-multi-gpu]\n";
        return 2;
    }

    rocsvHandle_t handle = nullptr;
    if (!check_status(rocsvCreate(&handle), "rocsvCreate")) {
        return 1;
    }

    int visible_gpus = 0;
    if (!check_status(rocsvGetNumGpus(handle, &visible_gpus), "rocsvGetNumGpus")) {
        rocsvDestroy(handle);
        return 1;
    }
    const int required_gpus = require_multi_gpu ? 2 : 1;
    if (visible_gpus < required_gpus) {
        std::cerr << "Need at least " << required_gpus << " visible GPU(s); skipping.\n";
        rocsvDestroy(handle);
        return kCtestSkipReturnCode;
    }

    int active_gpus = 1;
    while ((active_gpus << 1) <= visible_gpus) {
        active_gpus <<= 1;
    }
    unsigned global_slice_qubits = 0;
    for (int count = active_gpus; count > 1; count >>= 1) {
        ++global_slice_qubits;
    }
    const unsigned num_qubits = global_slice_qubits + 2;
    if (!check_status(rocsvAllocateDistributedState(handle, num_qubits), "rocsvAllocateDistributedState")) {
        rocsvDestroy(handle);
        return 1;
    }
    if (!check_status(rocsvInitializeDistributedState(handle), "rocsvInitializeDistributedState")) {
        rocsvDestroy(handle);
        return 1;
    }

    rocsvDistributedInfo_t info{};
    if (!check_status(rocsvGetDistributedInfo(handle, &info), "rocsvGetDistributedInfo")) {
        rocsvDestroy(handle);
        return 1;
    }
    if (!info.distributed_mode || info.global_num_qubits != num_qubits) {
        std::cerr << "Invalid distributed metadata.\n";
        rocsvDestroy(handle);
        return 1;
    }

    std::vector<rocComplex> expected(1u << num_qubits, rocComplex{0.0f, 0.0f});
    expected[0] = {1.0f, 0.0f};
    if (!expect_state(handle, num_qubits, expected)) {
        rocsvDestroy(handle);
        return 1;
    }

    if (!check_status(rocsvApplyX(handle, nullptr, num_qubits, 0), "rocsvApplyX")) {
        rocsvDestroy(handle);
        return 1;
    }
    if (!check_status(rocsvSynchronize(handle), "rocsvSynchronize(X)")) {
        rocsvDestroy(handle);
        return 1;
    }
    expected.assign(1u << num_qubits, rocComplex{0.0f, 0.0f});
    expected[1] = {1.0f, 0.0f};
    if (!expect_state(handle, num_qubits, expected)) {
        rocsvDestroy(handle);
        return 1;
    }

    if (!check_status(rocsvApplyCNOT(handle, nullptr, num_qubits, 0, 1), "rocsvApplyCNOT")) {
        rocsvDestroy(handle);
        return 1;
    }
    if (!check_status(rocsvSynchronize(handle), "rocsvSynchronize(CNOT)")) {
        rocsvDestroy(handle);
        return 1;
    }
    expected.assign(1u << num_qubits, rocComplex{0.0f, 0.0f});
    expected[3] = {1.0f, 0.0f};
    if (!expect_state(handle, num_qubits, expected)) {
        rocsvDestroy(handle);
        return 1;
    }

    if (!check_status(rocsvInitializeDistributedState(handle), "rocsvInitializeDistributedState(reset)")) {
        rocsvDestroy(handle);
        return 1;
    }

    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    const rocComplex host_h[4] = {
        {inv_sqrt2, 0.0f},
        {inv_sqrt2, 0.0f},
        {inv_sqrt2, 0.0f},
        {-inv_sqrt2, 0.0f},
    };

    rocComplex* dev_h = nullptr;
    if (!check_hip(hipMalloc(&dev_h, sizeof(host_h)), "hipMalloc(H)")) {
        rocsvDestroy(handle);
        return 1;
    }
    if (!check_hip(hipMemcpy(dev_h, host_h, sizeof(host_h), hipMemcpyHostToDevice), "hipMemcpy(H)")) {
        hipFree(dev_h);
        rocsvDestroy(handle);
        return 1;
    }

    if (!check_status(rocsvApplyFusedSingleQubitMatrix(handle, 0, dev_h), "rocsvApplyFusedSingleQubitMatrix")) {
        hipFree(dev_h);
        rocsvDestroy(handle);
        return 1;
    }
    if (!check_status(rocsvSynchronize(handle), "rocsvSynchronize(H)")) {
        hipFree(dev_h);
        rocsvDestroy(handle);
        return 1;
    }
    hipFree(dev_h);

    expected.assign(1u << num_qubits, rocComplex{0.0f, 0.0f});
    expected[0] = {inv_sqrt2, 0.0f};
    expected[1] = {inv_sqrt2, 0.0f};
    if (!expect_state(handle, num_qubits, expected)) {
        rocsvDestroy(handle);
        return 1;
    }

    if (require_multi_gpu) {
        if (info.gpu_count < 2 || info.global_slice_qubits < 1) {
            std::cerr << "Distributed allocation did not span multiple GPUs.\n";
            rocsvDestroy(handle);
            return 1;
        }

        rocsvDistributedBackend_t backend = ROCSV_DISTRIBUTED_BACKEND_NONE;
        if (!check_status(rocsvGetDistributedBackend(handle, &backend),
                          "rocsvGetDistributedBackend")) {
            rocsvDestroy(handle);
            return 1;
        }
        if (backend != ROCSV_DISTRIBUTED_BACKEND_RCCL) {
            std::cerr << "Multi-GPU inter-rank regression requires the RCCL backend; got "
                      << rocsvDistributedBackendName(backend) << ".\n";
            rocsvDestroy(handle);
            return 1;
        }

        if (!check_status(rocsvInitializeDistributedState(handle),
                          "rocsvInitializeDistributedState(inter-rank reset)")) {
            rocsvDestroy(handle);
            return 1;
        }
        const unsigned inter_rank_target = info.local_num_qubits_per_gpu;
        if (!check_status(rocsvApplyX(handle, nullptr, num_qubits, inter_rank_target),
                          "rocsvApplyX(inter-rank target)")) {
            rocsvDestroy(handle);
            return 1;
        }
        if (!check_status(rocsvSynchronize(handle), "rocsvSynchronize(inter-rank X)")) {
            rocsvDestroy(handle);
            return 1;
        }
        expected.assign(size_t{1} << num_qubits, rocComplex{0.0f, 0.0f});
        expected[size_t{1} << inter_rank_target] = {1.0f, 0.0f};
        if (!expect_state(handle, num_qubits, expected)) {
            std::cerr << "Inter-rank X did not move amplitude across the slice boundary.\n";
            rocsvDestroy(handle);
            return 1;
        }
    }

    rocsvDestroy(handle);
    return 0;
}

