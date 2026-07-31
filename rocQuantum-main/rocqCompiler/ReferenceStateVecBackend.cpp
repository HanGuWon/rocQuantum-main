#include "ReferenceStateVecBackend.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rocq {
namespace {

using Complex = std::complex<double>;

constexpr double kInverseSqrtTwo = 0.707106781186547524400844362104849039;
constexpr double kPiOverFour = 0.785398163397448309615660845819875721;

std::string lowercase(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

std::size_t qubit_mask(unsigned qubit) {
    return std::size_t{1} << qubit;
}

} // namespace

void ReferenceStateVecBackend::initialize(unsigned num_qubits) {
    destroy();
    if (num_qubits == 0) {
        throw std::invalid_argument(
            "cpu_statevec backend requires at least one qubit.");
    }
    if (num_qubits >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error(
            "cpu_statevec state dimension overflows size_t.");
    }

    const std::size_t dimension = std::size_t{1} << num_qubits;
    if (dimension > state_.max_size()) {
        throw std::length_error(
            "cpu_statevec state dimension exceeds vector capacity.");
    }

    state_.assign(dimension, Complex{0.0, 0.0});
    state_.front() = Complex{1.0, 0.0};
    num_qubits_ = num_qubits;
    initialized_ = true;
}

void ReferenceStateVecBackend::ensure_initialized() const {
    if (!initialized_) {
        throw std::runtime_error("cpu_statevec backend is not initialized.");
    }
}

void ReferenceStateVecBackend::validate_targets(
    const std::vector<unsigned>& targets,
    std::size_t minimum,
    std::size_t maximum,
    const std::string& gate_name) const {
    if (targets.size() < minimum || targets.size() > maximum) {
        throw std::invalid_argument(
            "gate '" + gate_name + "' expects " +
            (minimum == maximum
                 ? std::to_string(minimum)
                 : std::to_string(minimum) + ".." + std::to_string(maximum)) +
            " target(s), received " + std::to_string(targets.size()) + ".");
    }

    std::size_t seen = 0;
    for (unsigned target : targets) {
        if (target >= num_qubits_) {
            throw std::out_of_range(
                "gate '" + gate_name + "' references qubit " +
                std::to_string(target) + " but the backend has " +
                std::to_string(num_qubits_) + " qubits.");
        }
        const std::size_t mask = qubit_mask(target);
        if ((seen & mask) != 0) {
            throw std::invalid_argument(
                "gate '" + gate_name + "' qubit operands must be distinct.");
        }
        seen |= mask;
    }
}

void ReferenceStateVecBackend::apply_single_qubit(
    unsigned target,
    Complex m00,
    Complex m01,
    Complex m10,
    Complex m11,
    std::size_t control_mask) {
    const std::size_t target_mask = qubit_mask(target);
    for (std::size_t zero = 0; zero < state_.size(); ++zero) {
        if ((zero & target_mask) != 0 || (zero & control_mask) != control_mask) {
            continue;
        }
        const std::size_t one = zero | target_mask;
        const Complex old_zero = state_[zero];
        const Complex old_one = state_[one];
        state_[zero] = m00 * old_zero + m01 * old_one;
        state_[one] = m10 * old_zero + m11 * old_one;
    }
}

void ReferenceStateVecBackend::apply_multi_controlled_x(
    const std::vector<unsigned>& controls,
    unsigned target) {
    std::size_t control_mask = 0;
    for (unsigned control : controls) {
        control_mask |= qubit_mask(control);
    }
    apply_single_qubit(
        target,
        Complex{0.0, 0.0},
        Complex{1.0, 0.0},
        Complex{1.0, 0.0},
        Complex{0.0, 0.0},
        control_mask);
}

void ReferenceStateVecBackend::apply_swap(
    unsigned first,
    unsigned second,
    std::size_t control_mask) {
    const std::size_t first_mask = qubit_mask(first);
    const std::size_t second_mask = qubit_mask(second);
    for (std::size_t index = 0; index < state_.size(); ++index) {
        if ((index & control_mask) != control_mask ||
            (index & first_mask) != 0 ||
            (index & second_mask) == 0) {
            continue;
        }
        std::swap(state_[index], state_[index ^ first_mask ^ second_mask]);
    }
}

void ReferenceStateVecBackend::apply_gate(
    const std::string& gate_name,
    const std::vector<unsigned>& targets) {
    ensure_initialized();
    const std::string gate = lowercase(gate_name);
    const Complex zero{0.0, 0.0};
    const Complex one{1.0, 0.0};
    const Complex imaginary{0.0, 1.0};

    if (gate == "h") {
        validate_targets(targets, 1, 1, gate_name);
        apply_single_qubit(
            targets[0],
            kInverseSqrtTwo,
            kInverseSqrtTwo,
            kInverseSqrtTwo,
            -kInverseSqrtTwo);
        return;
    }
    if (gate == "x") {
        validate_targets(targets, 1, 1, gate_name);
        apply_single_qubit(targets[0], zero, one, one, zero);
        return;
    }
    if (gate == "y") {
        validate_targets(targets, 1, 1, gate_name);
        apply_single_qubit(targets[0], zero, -imaginary, imaginary, zero);
        return;
    }
    if (gate == "z") {
        validate_targets(targets, 1, 1, gate_name);
        apply_single_qubit(targets[0], one, zero, zero, -one);
        return;
    }
    if (gate == "s") {
        validate_targets(targets, 1, 1, gate_name);
        apply_single_qubit(targets[0], one, zero, zero, imaginary);
        return;
    }
    if (gate == "sdg" || gate == "s_adj") {
        validate_targets(targets, 1, 1, gate_name);
        apply_single_qubit(targets[0], one, zero, zero, -imaginary);
        return;
    }
    if (gate == "t") {
        validate_targets(targets, 1, 1, gate_name);
        apply_single_qubit(
            targets[0], one, zero, zero, std::polar(1.0, kPiOverFour));
        return;
    }
    if (gate == "tdg" || gate == "t_adj") {
        validate_targets(targets, 1, 1, gate_name);
        apply_single_qubit(
            targets[0], one, zero, zero, std::polar(1.0, -kPiOverFour));
        return;
    }
    if (gate == "cnot" || gate == "cx") {
        validate_targets(targets, 2, 2, gate_name);
        apply_multi_controlled_x({targets[0]}, targets[1]);
        return;
    }
    if (gate == "cz") {
        validate_targets(targets, 2, 2, gate_name);
        const std::size_t mask = qubit_mask(targets[0]) | qubit_mask(targets[1]);
        for (std::size_t index = 0; index < state_.size(); ++index) {
            if ((index & mask) == mask) {
                state_[index] = -state_[index];
            }
        }
        return;
    }
    if (gate == "swap") {
        validate_targets(targets, 2, 2, gate_name);
        apply_swap(targets[0], targets[1]);
        return;
    }
    if (gate == "ccx") {
        validate_targets(targets, 3, 3, gate_name);
        apply_multi_controlled_x({targets[0], targets[1]}, targets[2]);
        return;
    }
    if (gate == "mcx") {
        validate_targets(
            targets,
            2,
            std::numeric_limits<std::size_t>::max(),
            gate_name);
        apply_multi_controlled_x(
            std::vector<unsigned>(targets.begin(), targets.end() - 1),
            targets.back());
        return;
    }
    if (gate == "cswap" || gate == "fredkin") {
        validate_targets(targets, 3, 3, gate_name);
        apply_swap(targets[1], targets[2], qubit_mask(targets[0]));
        return;
    }

    throw std::invalid_argument(
        "cpu_statevec backend does not support gate '" + gate_name + "'.");
}

void ReferenceStateVecBackend::apply_parametrized_gate(
    const std::string& gate_name,
    double parameter,
    const std::vector<unsigned>& targets) {
    ensure_initialized();
    if (!std::isfinite(parameter)) {
        throw std::invalid_argument(
            "gate '" + gate_name + "' requires a finite parameter.");
    }

    const std::string gate = lowercase(gate_name);
    const bool controlled =
        gate == "crx" || gate == "cry" || gate == "crz" ||
        gate == "cp" || gate == "cphase";
    validate_targets(targets, controlled ? 2 : 1, controlled ? 2 : 1, gate_name);

    const unsigned target = targets.back();
    const std::size_t control_mask =
        controlled ? qubit_mask(targets.front()) : 0;
    const Complex zero{0.0, 0.0};
    const double half = parameter / 2.0;

    if (gate == "rx" || gate == "crx") {
        const double cosine = std::cos(half);
        const Complex off_diagonal{0.0, -std::sin(half)};
        apply_single_qubit(
            target,
            cosine,
            off_diagonal,
            off_diagonal,
            cosine,
            control_mask);
        return;
    }
    if (gate == "ry" || gate == "cry") {
        const double cosine = std::cos(half);
        const double sine = std::sin(half);
        apply_single_qubit(
            target, cosine, -sine, sine, cosine, control_mask);
        return;
    }
    if (gate == "rz" || gate == "crz") {
        apply_single_qubit(
            target,
            std::polar(1.0, -half),
            zero,
            zero,
            std::polar(1.0, half),
            control_mask);
        return;
    }
    if (gate == "p" || gate == "phase" || gate == "cp" ||
        gate == "cphase") {
        apply_single_qubit(
            target,
            Complex{1.0, 0.0},
            zero,
            zero,
            std::polar(1.0, parameter),
            control_mask);
        return;
    }

    throw std::invalid_argument(
        "cpu_statevec backend does not support parametrized gate '" +
        gate_name + "'.");
}

std::vector<std::complex<double>>
ReferenceStateVecBackend::get_state_vector() {
    ensure_initialized();
    return state_;
}

void ReferenceStateVecBackend::destroy() {
    // destroy() is the backend resource-release boundary.  Swapping, rather
    // than clear(), returns potentially very large state-vector storage before
    // the compiler object itself is destroyed.
    std::vector<Complex>{}.swap(state_);
    num_qubits_ = 0;
    initialized_ = false;
}

std::unique_ptr<QuantumBackend> create_reference_backend() {
    return std::make_unique<ReferenceStateVecBackend>();
}

} // namespace rocq
