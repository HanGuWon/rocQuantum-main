#include "rocquantum/rocWorkspaceManager.h"
#include "rocquantum/hipStateVec.h" // For checkHipError

#include <limits>
#include <string>

namespace rocquantum {
namespace util {

namespace {

size_t align_up(size_t value, size_t alignment) noexcept {
    if (alignment == 0) {
        return value;
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    const size_t padding = alignment - remainder;
    if (value > std::numeric_limits<size_t>::max() - padding) {
        return std::numeric_limits<size_t>::max();
    }
    return value + padding;
}

} // namespace

WorkspaceManager::WorkspaceManager(size_t initial_size_bytes, hipStream_t stream)
    : d_workspace_ptr_(nullptr),
      total_size_bytes_(initial_size_bytes),
      current_offset_bytes_(0),
      stream_(stream) {
    if (total_size_bytes_ > 0) {
        hipError_t err = hipMalloc(&d_workspace_ptr_, total_size_bytes_);
        if (err != hipSuccess) {
            // In a real scenario, might throw or log. For now, ptr remains null.
            // checkHipError(err, "WorkspaceManager hipMalloc"); // Assuming checkHipError exists and handles logging/throwing
            d_workspace_ptr_ = nullptr; // Ensure it's null on failure
            total_size_bytes_ = 0;      // and size is 0
            // Consider throwing std::runtime_error here
            throw std::runtime_error("WorkspaceManager: hipMalloc failed to allocate workspace of size " + std::to_string(initial_size_bytes));
        }
    }
}

WorkspaceManager::~WorkspaceManager() {
    if (d_workspace_ptr_) {
        hipFree(d_workspace_ptr_);
        d_workspace_ptr_ = nullptr;
    }
}

rocComplex* WorkspaceManager::allocate(size_t num_elements) {
    if (!d_workspace_ptr_ || num_elements == 0) {
        return nullptr;
    }

    if (num_elements > std::numeric_limits<size_t>::max() / sizeof(rocComplex)) {
        return nullptr;
    }
    const size_t requested_bytes = num_elements * sizeof(rocComplex);
    if (requested_bytes == 0) return nullptr; // Should not happen if num_elements > 0

    // Align the current offset
    const size_t aligned_offset = align_up(current_offset_bytes_, alignment_);

    if (aligned_offset <= total_size_bytes_ &&
        requested_bytes <= total_size_bytes_ - aligned_offset) {
        rocComplex* ptr = reinterpret_cast<rocComplex*>(
            reinterpret_cast<char*>(d_workspace_ptr_) + aligned_offset
        );
        current_offset_bytes_ = aligned_offset + requested_bytes;
        return ptr;
    } else {
        // Not enough space with simple bump allocator
        // A more sophisticated manager might try to grow the pool or find free blocks.
        return nullptr;
    }
}

void WorkspaceManager::reset() {
    current_offset_bytes_ = 0;
}

size_t WorkspaceManager::get_total_size_bytes() const {
    return total_size_bytes_;
}

size_t WorkspaceManager::get_used_size_bytes() const {
    // Return the aligned offset as used, as that's the next allocation start
    return align_up(current_offset_bytes_, alignment_);
}

hipStream_t WorkspaceManager::get_stream() const {
    return stream_;
}

} // namespace util
} // namespace rocquantum
