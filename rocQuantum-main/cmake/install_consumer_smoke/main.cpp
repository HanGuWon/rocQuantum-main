#include <rocquantum/QuantumSimulator.h>
#include <rocquantum/hipDensityMat.h>
#include <rocquantum/hipStateVec.h>
#include <rocquantum/hipTensorNet.h>
#include <rocquantum/hipTensorNet_api.h>

int main() {
    static_assert(ROCSV_DISTRIBUTED_BACKEND_RCCL == 2, "distributed backend enum is visible");

    using CoreSymbol = unsigned (rocquantum::QuantumSimulator::*)() const noexcept;
    CoreSymbol volatile core_symbol = &rocquantum::QuantumSimulator::num_qubits;

    const rocqStatus_t statevec_status = rocsvDestroy(nullptr);
    const rocqStatus_t densitymat_status = rocdmDestroyState(nullptr);
    hipTensorNetCapabilities_t tensornet_caps{};
    const rocqStatus_t tensornet_status = rocTensorNetworkGetCapabilities(&tensornet_caps);
#ifdef ROCQ_PRECISION_DOUBLE
    static_assert(sizeof(rocComplex) == sizeof(rocDoubleComplex),
                  "StateVec consumer ABI must use complex128");
    const bool precision_contract =
        tensornet_caps.supports_c128 == 1 && tensornet_caps.supports_c64 == 0;
#else
    static_assert(sizeof(rocComplex) == sizeof(rocFloatComplex),
                  "StateVec consumer ABI must use complex64");
    const bool precision_contract =
        tensornet_caps.supports_c64 == 1 && tensornet_caps.supports_c128 == 0;
#endif

#ifdef ROCQ_EXPECT_METIS
    const bool metis_contract = tensornet_caps.supports_pathfinder_metis == 1;
#else
    const bool metis_contract = tensornet_caps.supports_pathfinder_metis == 0;
#endif

    const bool linked_all_components =
        core_symbol != nullptr &&
        statevec_status == ROCQ_STATUS_SUCCESS &&
        densitymat_status == ROCQ_STATUS_SUCCESS &&
        tensornet_status == ROCQ_STATUS_SUCCESS &&
        precision_contract &&
        metis_contract;
    return linked_all_components ? 0 : 1;
}
