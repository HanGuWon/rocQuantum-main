#include <rocquantum/QuantumSimulator.h>
#include <rocquantum/hipDensityMat.h>
#include <rocquantum/hipStateVec.h>
#include <rocquantum/hipTensorNet_api.h>

int main() {
    static_assert(ROCSV_DISTRIBUTED_BACKEND_RCCL == 2, "distributed backend enum is visible");

    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    hipTensorNetCapabilities_t tensornet_caps{};
    rocdmHandle_t density_handle = nullptr;
    rocquantum::QuantumSimulator* simulator = nullptr;

    (void)status;
    (void)tensornet_caps;
    (void)density_handle;
    return simulator == nullptr ? 0 : 1;
}
