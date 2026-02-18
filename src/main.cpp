#include <iostream>
#include <vector>
#include "cuda_utils.h"
#include "memory_manager.h" // <--- NEW
#include "config.h"         // <--- NEW

int main(int argc, char** argv) {
    std::cout << "[cuLlama] Initializing Inference Engine..." << std::endl;

#ifdef USE_ROCM
    std::cout << "[System] Backend: AMD ROCm (HIP)" << std::endl;
#else
    std::cout << "[System] Backend: NVIDIA CUDA" << std::endl;
#endif

    // 1. Setup Model Config
    ModelConfig config;
    std::cout << "[Config] Llama-2-7b (Dim: " << config.dim << ")" << std::endl;

    // 2. Initialize Memory Manager (Allocate 500MB scratchpad)
    size_t scratch_size = 500 * 1024 * 1024; // 500 MB
    MemoryManager allocator(scratch_size);

    // 3. Simulate a Forward Pass (e.g., Layer 1)
    std::cout << "\n--- Simulating Inference Step 1 ---" << std::endl;
    
    // Allocate space for Hidden States (Batch Size 1, Seq Len 128)
    int tokens = 128;
    float* d_input = allocator.allocate<float>(tokens * config.dim);
    std::cout << "Allocated Input Tensor: " << d_input << std::endl;

    float* d_output = allocator.allocate<float>(tokens * config.dim);
    std::cout << "Allocated Output Tensor: " << d_output << std::endl;

    // 4. Reset for next step
    std::cout << "Resetting Arena..." << std::endl;
    allocator.reset();

    // 5. Simulate Step 2 (Pointers should be the SAME as Step 1)
    std::cout << "\n--- Simulating Inference Step 2 ---" << std::endl;
    
    float* d_input_2 = allocator.allocate<float>(tokens * config.dim);
    std::cout << "Allocated Input Tensor: " << d_input_2 << std::endl;

    if (d_input == d_input_2) {
        std::cout << "[SUCCESS] Memory Reuse Verified! Zero overhead allocation." << std::endl;
    } else {
        std::cerr << "[FAILURE] Pointers do not match!" << std::endl;
    }

    return 0;
}