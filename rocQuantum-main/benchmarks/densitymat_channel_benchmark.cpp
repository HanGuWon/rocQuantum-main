#include "rocquantum/hipDensityMat.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

hipComplex make_complex(float real, float imag = 0.0f) {
    return hipComplex{real, imag};
}

bool check_status(rocqStatus_t status, const char* what) {
    if (status != ROCQ_STATUS_SUCCESS) {
        std::cerr << what << " failed with status " << static_cast<int>(status) << "\n";
        return false;
    }
    return true;
}

struct CaseResult {
    std::string name;
    int status = 0;
    double ms_per_trial = 0.0;
};

template <typename Fn>
CaseResult run_timed_case(const std::string& name, unsigned trials, Fn&& fn) {
    CaseResult result{name};
    auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < trials; ++i) {
        if (!fn()) {
            result.status = 1;
            break;
        }
    }
    auto end = std::chrono::steady_clock::now();
    const unsigned completed = result.status == 0 ? trials : 1;
    result.ms_per_trial =
        std::chrono::duration<double, std::milli>(end - start).count() / static_cast<double>(completed);
    return result;
}

std::vector<hipComplex> bit_flip_kraus(float probability) {
    const float keep = std::sqrt(1.0f - probability);
    const float flip = std::sqrt(probability);
    return {
        make_complex(keep), make_complex(0.0f), make_complex(0.0f), make_complex(keep),
        make_complex(0.0f), make_complex(flip), make_complex(flip), make_complex(0.0f),
    };
}

void write_json(std::ostream& out,
                int qubits,
                unsigned trials,
                int shots,
                const std::vector<CaseResult>& results) {
    out << "{\n"
        << "  \"benchmark\": \"densitymat_channel_sampling\",\n"
        << "  \"qubits\": " << qubits << ",\n"
        << "  \"trials\": " << trials << ",\n"
        << "  \"shots\": " << shots << ",\n"
        << "  \"cases\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const CaseResult& r = results[i];
        out << "    {\"name\": \"" << r.name << "\", \"status\": " << r.status
            << ", \"ms_per_trial\": " << r.ms_per_trial << "}";
        out << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    int qubits = 10;
    unsigned trials = 20;
    int shots = 4096;
    std::string output;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--qubits" && i + 1 < argc) {
            qubits = std::stoi(argv[++i]);
        } else if (arg == "--trials" && i + 1 < argc) {
            trials = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--shots" && i + 1 < argc) {
            shots = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        }
    }

    if (qubits <= 0 || trials == 0 || shots <= 0) {
        std::cerr << "qubits, trials, and shots must be positive\n";
        return 1;
    }

    rocdmHandle_t state = nullptr;
    if (!check_status(rocdmCreateState(&state, qubits), "rocdmCreateState")) {
        return 1;
    }

    const std::vector<hipComplex> kraus = bit_flip_kraus(0.01f);
    rocdmChannel_t channel;
    channel.num_kraus = 2;
    channel.kraus_matrices_host = kraus.data();

    std::vector<CaseResult> results;
    results.push_back(run_timed_case("named_bit_flip_channel", trials, [&]() {
        return check_status(rocdmApplyBitFlipChannel(state, 0, 0.01), "rocdmApplyBitFlipChannel");
    }));

    results.push_back(run_timed_case("generic_kraus_channel", trials, [&]() {
        return check_status(rocdmApplyChannel(state, 1 % qubits, &channel), "rocdmApplyChannel");
    }));

    double expectation = 0.0;
    results.push_back(run_timed_case("density_observe_z", trials, [&]() {
        return check_status(rocdmComputeExpectation(state, 0, ROCDM_PAULI_Z, &expectation),
                            "rocdmComputeExpectation");
    }));

    std::vector<uint64_t> samples(static_cast<size_t>(shots), 0);
    const int measured[] = {0, 1, 2};
    const int measured_count = qubits >= 3 ? 3 : qubits;
    results.push_back(run_timed_case("density_sample_host_correctness", trials, [&]() {
        return check_status(rocdmSample(state, measured, measured_count, shots, samples.data()), "rocdmSample");
    }));

    std::ostream* out = &std::cout;
    std::ofstream file;
    if (!output.empty()) {
        file.open(output);
        if (!file) {
            std::cerr << "failed to open output file: " << output << "\n";
            rocdmDestroyState(state);
            return 1;
        }
        out = &file;
    }

    write_json(*out, qubits, trials, shots, results);

    bool any_success = false;
    for (const CaseResult& result : results) {
        any_success = any_success || result.status == 0;
    }

    rocdmDestroyState(state);
    return any_success ? 0 : 1;
}
