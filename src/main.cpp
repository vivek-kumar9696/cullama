#include <iostream>
#include <vector>
#include "cuda_utils.h"

// Forward declare a dummy kernel to test compilation
void launchTestKernel();

int main(int argc, char** argv) {
    std::cout << "[cuLlama] Initializing Inference Engine..." << std::endl;

#ifdef USE_ROCM
    std::cout << "[System] Backend: AMD ROCm (HIP)" << std::endl;
#else
    std::cout << "[System] Backend: NVIDIA CUDA" << std::endl;
#endif

    // 1. Test Memory Allocation (The "Hello World" of GPU Systems)
    size_t free_mem, total_mem;
    // Note: cudaMemGetInfo is mapped to hipMemGetInfo in compatible headers usually,
    // but strict mapping might require adding it to hip_compat.h.
    // For now, let's just do a malloc.
    
    float* d_buffer;
    size_t alloc_size = 1024 * 1024 * sizeof(float); // 1MB
    
    std::cout << "[System] Allocating 1MB on Device..." << std::endl;
    CUDA_CHECK(cudaMalloc((void**)&d_buffer, alloc_size));

    std::cout << "[System] Allocation Successful. Pointer: " << d_buffer << std::endl;

    // 2. Free it
    CUDA_CHECK(cudaFree(d_buffer));
    std::cout << "[System] Memory Freed." << std::endl;

    return 0;
}