#include <iostream>
#include <vector>
#include <cmath>
#include "cuda_utils.h"
#include "memory_manager.h"
#include "layers.h"
#include "config.h"

int main() {
    std::cout << "[cuLlama] Phase 2: Kernel Testing (RoPE)" << std::endl;
    
    // 1. Setup
    int head_dim = 128; // Standard Llama size
    int n_heads = 4;
    int n_kv_heads = 4;
    int pos = 1; // Position 1

    MemoryManager allocator(10 * 1024 * 1024); // 10MB

    // 2. Allocate Q (Batch 1, Heads 4, Dim 128)
    size_t q_size = n_heads * head_dim;
    float* d_q = allocator.allocate<float>(q_size);
    float* d_k = allocator.allocate<float>(q_size); // Dummy K for now

    // 3. Initialize Host Data: [1, 0, 1, 0...]
    // A vector (1, 0) rotated by 90 degrees should become (0, 1) approx.
    std::vector<float> h_q(q_size);
    for (int i = 0; i < q_size; i+=2) {
        h_q[i] = 1.0f;   // x
        h_q[i+1] = 0.0f; // y
    }

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Launch RoPE
    // Llama theta base is 10000. 
    // For index 0: theta = 1.0 / 10000^0 = 1.0.
    // If pos = 1, rotation angle is 1 radian (~57 degrees).
    // Expected Result: x = cos(1) - 0*sin(1) = 0.54
    //                  y = 1*sin(1) + 0*cos(1) = 0.84
    
    std::cout << "Launching RoPE Kernel..." << std::endl;
    launch_rope(d_q, d_k, pos, head_dim, n_heads, n_kv_heads);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Verify
    std::vector<float> h_q_out(q_size);
    CUDA_CHECK(cudaMemcpy(h_q_out.data(), d_q, q_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Input: (1.0, 0.0)" << std::endl;
    std::cout << "Output (Index 0): (" << h_q_out[0] << ", " << h_q_out[1] << ")" << std::endl;

    // Check against CPU math
    float expected_x = cosf(1.0f);
    float expected_y = sinf(1.0f);

    std::cout << "Expected: (" << expected_x << ", " << expected_y << ")" << std::endl;

    if (abs(h_q_out[0] - expected_x) < 1e-4) {
        std::cout << "[SUCCESS] RoPE Rotation Verified!" << std::endl;
    } else {
        std::cerr << "[FAILURE] Math Mismatch." << std::endl;
    }

    return 0;
}