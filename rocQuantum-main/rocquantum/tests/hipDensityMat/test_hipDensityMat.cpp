#include "rocquantum/hipDensityMat.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr int kCtestSkipReturnCode = 77;

bool check_status(rocqStatus_t status, const char* operation) {
    if (status == ROCQ_STATUS_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with status " << static_cast<int>(status) << '\n';
    return false;
}

bool expect_near(double actual, double expected, const char* label, double tolerance = 1e-5) {
    if (std::abs(actual - expected) <= tolerance) {
        return true;
    }
    std::cerr << label << " mismatch: got " << actual << ", expected " << expected << '\n';
    return false;
}

bool test_cnot_and_sampling() {
    rocdmHandle_t state = nullptr;
    if (!check_status(rocdmCreateState(&state, 2), "rocdmCreateState(2)")) {
        return false;
    }

    const hipComplex x_gate[4] = {
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(1.0f, 0.0f),
        make_hipFloatComplex(1.0f, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
    };

    bool ok = check_status(rocdmApplyGate(state, 0, x_gate), "rocdmApplyGate(X0)") &&
              check_status(rocdmApplyCNOT(state, 0, 1), "rocdmApplyCNOT(0, 1)");

    double z0 = 0.0;
    double z1 = 0.0;
    double z0z1 = 0.0;
    const int both_qubits[2] = {0, 1};
    ok = check_status(rocdmComputeExpectation(state, 0, ROCDM_PAULI_Z, &z0),
                      "rocdmComputeExpectation(Z0)") &&
         check_status(rocdmComputeExpectation(state, 1, ROCDM_PAULI_Z, &z1),
                      "rocdmComputeExpectation(Z1)") &&
         check_status(rocdmComputePauliZProductExpectation(state, 2, both_qubits, &z0z1),
                      "rocdmComputePauliZProductExpectation") &&
         ok;
    ok = expect_near(z0, -1.0, "<Z0>") &&
         expect_near(z1, -1.0, "<Z1>") &&
         expect_near(z0z1, 1.0, "<Z0 Z1>") &&
         ok;

    std::vector<std::uint64_t> outcomes(32, 0);
    ok = check_status(rocdmSample(state,
                                  both_qubits,
                                  2,
                                  static_cast<int>(outcomes.size()),
                                  outcomes.data()),
                      "rocdmSample") &&
         ok;
    for (std::uint64_t outcome : outcomes) {
        if (outcome != 3) {
            std::cerr << "Deterministic |11> sampling returned outcome " << outcome << '\n';
            ok = false;
            break;
        }
    }

    ok = check_status(rocdmDestroyState(state), "rocdmDestroyState(2)") && ok;
    return ok;
}

bool test_channels() {
    rocdmHandle_t state = nullptr;
    if (!check_status(rocdmCreateState(&state, 1), "rocdmCreateState(1)")) {
        return false;
    }

    bool ok = check_status(rocdmApplyBitFlipChannel(state, 0, 1.0),
                           "rocdmApplyBitFlipChannel(p=1)");
    double expectation = 0.0;
    ok = check_status(rocdmComputeExpectation(state, 0, ROCDM_PAULI_Z, &expectation),
                      "rocdmComputeExpectation(after bit flip)") &&
         expect_near(expectation, -1.0, "bit-flip <Z>") &&
         ok;

    ok = check_status(rocdmApplyAmplitudeDampingChannel(state, 0, 1.0),
                      "rocdmApplyAmplitudeDampingChannel(gamma=1)") &&
         ok;
    ok = check_status(rocdmComputeExpectation(state, 0, ROCDM_PAULI_Z, &expectation),
                      "rocdmComputeExpectation(after amplitude damping)") &&
         expect_near(expectation, 1.0, "amplitude-damping <Z>") &&
         ok;

    ok = check_status(rocdmDestroyState(state), "rocdmDestroyState(1)") && ok;
    return ok;
}

bool test_invalid_limits() {
    rocdmHandle_t state = nullptr;
    const rocqStatus_t status = rocdmCreateState(&state, ROCDM_MAX_QUBITS + 1);
    if (status != ROCQ_STATUS_INVALID_VALUE || state != nullptr) {
        std::cerr << "State creation beyond ROCDM_MAX_QUBITS must fail without allocating.\n";
        if (state) {
            (void)rocdmDestroyState(state);
        }
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::cerr << "No ROCm GPU is visible; skipping native hipDensityMat regression.\n";
        return kCtestSkipReturnCode;
    }

    const bool ok = test_invalid_limits() && test_cnot_and_sampling() && test_channels();
    return ok ? 0 : 1;
}
