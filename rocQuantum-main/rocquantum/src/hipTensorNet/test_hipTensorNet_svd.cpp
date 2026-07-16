#include "rocquantum/hipTensorNet.h"
#include "rocquantum/rocTensorUtil.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
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

bool check_hip(hipError_t status, const char* operation) {
    if (status == hipSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << hipGetErrorString(status) << '\n';
    return false;
}

bool near(double actual, double expected, double tolerance = 1e-5) {
    return std::abs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::cerr << "No ROCm GPU is visible; skipping native hipTensorNet SVD regression.\n";
        return kCtestSkipReturnCode;
    }

    rocTensorNetworkHandle_t handle = nullptr;
    if (!check_status(rocTensorNetworkCreate(&handle, ROC_TENSORNET_COMPILED_COMPLEX_DTYPE),
                      "rocTensorNetworkCreate")) {
        return 1;
    }

    rocquantum::util::rocTensor input;
    rocquantum::util::rocTensor u;
    rocquantum::util::rocTensor singular_values;
    rocquantum::util::rocTensor vh;
    input.dimensions_ = {2, 2};
    input.calculate_strides();

    bool ok = check_status(rocquantum::util::rocTensorAllocate(&input),
                           "rocTensorAllocate(input)");
    const std::vector<rocComplex> host_input = {
        {1.0f, 1.0f}, {0.0f, 0.0f},
        {2.0f, 0.0f}, {1.0f, -1.0f},
    };
    if (ok) {
        ok = check_hip(hipMemcpy(input.data_,
                                 host_input.data(),
                                 host_input.size() * sizeof(rocComplex),
                                 hipMemcpyHostToDevice),
                       "hipMemcpy(input)");
    }

    if (ok) {
        ok = check_status(rocTensorSVD(handle,
                                       &u,
                                       &singular_values,
                                       &vh,
                                       &input,
                                       nullptr),
                          "rocTensorSVD");
    }

    if (ok && (u.dimensions_ != std::vector<long long>{2, 2} ||
               singular_values.dimensions_ != std::vector<long long>{2} ||
               vh.dimensions_ != std::vector<long long>{2, 2})) {
        std::cerr << "rocTensorSVD returned unexpected output shapes.\n";
        ok = false;
    }

    std::vector<rocComplex> host_singular_values(2);
    if (ok) {
        ok = check_hip(hipMemcpy(host_singular_values.data(),
                                 singular_values.data_,
                                 host_singular_values.size() * sizeof(rocComplex),
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy(singular values)");
    }
    if (ok) {
        const double first = static_cast<double>(host_singular_values[0].x);
        const double second = static_cast<double>(host_singular_values[1].x);
        if (first < second || second < 0.0 ||
            !near(static_cast<double>(host_singular_values[0].y), 0.0) ||
            !near(static_cast<double>(host_singular_values[1].y), 0.0)) {
            std::cerr << "rocTensorSVD returned incorrect singular values.\n";
            ok = false;
        }
    }

    std::vector<rocComplex> host_u(4);
    std::vector<rocComplex> host_vh(4);
    if (ok) {
        ok = check_hip(hipMemcpy(host_u.data(),
                                 u.data_,
                                 host_u.size() * sizeof(rocComplex),
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy(U)") &&
             check_hip(hipMemcpy(host_vh.data(),
                                 vh.data_,
                                 host_vh.size() * sizeof(rocComplex),
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy(Vh)");
    }
    if (ok) {
        const auto as_complex = [](const rocComplex& value) {
            return std::complex<double>(static_cast<double>(value.x),
                                        static_cast<double>(value.y));
        };
        const auto at = [](const std::vector<rocComplex>& matrix, int row, int column) {
            return matrix[static_cast<std::size_t>(row + 2 * column)];
        };
        constexpr double tolerance = 5e-4;

        for (int row = 0; row < 2 && ok; ++row) {
            for (int column = 0; column < 2; ++column) {
                std::complex<double> reconstructed = 0.0;
                for (int k = 0; k < 2; ++k) {
                    reconstructed += as_complex(at(host_u, row, k)) *
                                     static_cast<double>(host_singular_values[k].x) *
                                     as_complex(at(host_vh, k, column));
                }
                const auto expected = as_complex(at(host_input, row, column));
                if (std::abs(reconstructed - expected) > tolerance) {
                    std::cerr << "rocTensorSVD reconstruction mismatch at (" << row << ", "
                              << column << ").\n";
                    ok = false;
                    break;
                }
            }
        }

        for (int first = 0; first < 2 && ok; ++first) {
            for (int second = 0; second < 2; ++second) {
                std::complex<double> u_inner = 0.0;
                std::complex<double> vh_inner = 0.0;
                for (int k = 0; k < 2; ++k) {
                    u_inner += std::conj(as_complex(at(host_u, k, first))) *
                               as_complex(at(host_u, k, second));
                    vh_inner += as_complex(at(host_vh, first, k)) *
                                std::conj(as_complex(at(host_vh, second, k)));
                }
                const std::complex<double> expected = (first == second) ? 1.0 : 0.0;
                if (std::abs(u_inner - expected) > tolerance ||
                    std::abs(vh_inner - expected) > tolerance) {
                    std::cerr << "rocTensorSVD returned non-unitary singular vectors.\n";
                    ok = false;
                    break;
                }
            }
        }
    }

    std::vector<rocComplex> input_after(host_input.size());
    if (ok) {
        ok = check_hip(hipMemcpy(input_after.data(),
                                 input.data_,
                                 input_after.size() * sizeof(rocComplex),
                                 hipMemcpyDeviceToHost),
                       "hipMemcpy(input after SVD)");
    }
    if (ok) {
        for (std::size_t index = 0; index < host_input.size(); ++index) {
            if (!near(static_cast<double>(input_after[index].x),
                      static_cast<double>(host_input[index].x)) ||
                !near(static_cast<double>(input_after[index].y),
                      static_cast<double>(host_input[index].y))) {
                std::cerr << "rocTensorSVD modified its input tensor at index " << index << ".\n";
                ok = false;
                break;
            }
        }
    }

    (void)rocquantum::util::rocTensorFree(&input);
    (void)rocquantum::util::rocTensorFree(&u);
    (void)rocquantum::util::rocTensorFree(&singular_values);
    (void)rocquantum::util::rocTensorFree(&vh);
    ok = check_status(rocTensorNetworkDestroy(handle), "rocTensorNetworkDestroy") && ok;
    return ok ? 0 : 1;
}
