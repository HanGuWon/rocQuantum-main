#include "rocqCompiler/ReferenceStateVecBackend.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void requireClose(const std::complex<double>& actual,
                  const std::complex<double>& expected,
                  const std::string& label) {
    constexpr double tolerance = 1.0e-12;
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            label + " mismatch: expected " + std::to_string(expected.real()) +
            "+" + std::to_string(expected.imag()) + "i, received " +
            std::to_string(actual.real()) + "+" +
            std::to_string(actual.imag()) + "i");
    }
}

} // namespace

int main() {
    try {
        auto bell = rocq::create_reference_backend();
        bell->initialize(/*num_qubits=*/2);
        bell->apply_gate("h", {0});
        bell->apply_gate("cnot", {0, 1});
        const auto bell_state = bell->get_state_vector();
        const double amplitude = 1.0 / std::sqrt(2.0);
        if (bell_state.size() != 4) {
            throw std::runtime_error("Bell state has the wrong dimension");
        }
        requireClose(bell_state[0], {amplitude, 0.0}, "Bell |00>");
        requireClose(bell_state[1], {0.0, 0.0}, "Bell |01>");
        requireClose(bell_state[2], {0.0, 0.0}, "Bell |10>");
        requireClose(bell_state[3], {amplitude, 0.0}, "Bell |11>");

        auto rotation = rocq::create_reference_backend();
        rotation->initialize(/*num_qubits=*/2);
        rotation->apply_gate("x", {0});
        rotation->apply_parametrized_gate("cry", 3.14159265358979323846, {0, 1});
        const auto rotation_state = rotation->get_state_vector();
        requireClose(rotation_state[0], {0.0, 0.0}, "CRY |00>");
        requireClose(rotation_state[1], {0.0, 0.0}, "CRY |01>");
        requireClose(rotation_state[2], {0.0, 0.0}, "CRY |10>");
        requireClose(rotation_state[3], {1.0, 0.0}, "CRY |11>");

        auto controlled = rocq::create_reference_backend();
        controlled->initialize(/*num_qubits=*/4);
        controlled->apply_gate("x", {0});
        controlled->apply_gate("x", {1});
        controlled->apply_gate("x", {2});
        controlled->apply_gate("mcx", {0, 1, 2, 3});
        const auto controlled_state = controlled->get_state_vector();
        requireClose(controlled_state[15], {1.0, 0.0}, "MCX |1111>");
        controlled->destroy();
        controlled->initialize(/*num_qubits=*/1);
        controlled->apply_gate("x", {0});
        const auto reused_state = controlled->get_state_vector();
        if (reused_state.size() != 2) {
            throw std::runtime_error(
                "destroy/reinitialize retained the prior state dimension");
        }
        requireClose(reused_state[0], {0.0, 0.0}, "reused |0>");
        requireClose(reused_state[1], {1.0, 0.0}, "reused |1>");

        std::cout << "rocq CPU reference backend smoke tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
