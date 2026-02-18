#include <iostream>
#include <vector>
#include <cmath>       // For expf, abs
#include <cuda_runtime.h>

// Project Headers
#include "cuda_utils.h"
#include "memory_manager.h"
#include "layers.h"
#include "config.h"

int main() {
    std::cout << "[cuLlama] Phase 2.5: SwiGLU Activation" << std::endl;
    
    // 1. Setup
    int dim = 4096; // Standard hidden dim
    MemoryManager allocator(10 * 1024 * 1024); // 10MB Arena

    float* d_gate = allocator.allocate<float>(dim);
    float* d_up   = allocator.allocate<float>(dim);

    // 2. Initialize Host Data
    // Gate = 10.0, Up = 0.5
    // Result should be SiLU(10) * 0.5
    // SiLU(10) ~= 10.0 (since sigmoid(10) ~= 1)
    // Result ~= 5.0
    
    std::vector<float> h_gate(dim, 10.0f);
    std::vector<float> h_up(dim, 0.5f);

    CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), dim*sizeof(float), cudaMemcpyHostToDevice));

    // 3. Launch
    std::cout << "Launching Fused SwiGLU Kernel..." << std::endl;
    launch_silu_mul(d_gate, d_up, dim);
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Verify
    std::vector<float> h_out(dim);
    // In our kernel, we wrote the result back into d_gate (in-place)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_gate, dim*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Input Gate: 10.0, Input Up: 0.5" << std::endl;
    std::cout << "Output: " << h_out[0] << std::endl;

    // Check Math: 10 * (1 / (1 + exp(-10))) * 0.5
    float sigmoid = 1.0f / (1.0f + expf(-10.0f));
    float expected = (10.0f * sigmoid) * 0.5f;
    
    std::cout << "Expected: " << expected << std::endl;

    if (std::abs(h_out[0] - expected) < 1e-4) {
        std::cout << "[SUCCESS] SwiGLU Logic Verified!" << std::endl;
    } else {
        std::cerr << "[FAILURE] Math Mismatch." << std::endl;
    }

    return 0;
}