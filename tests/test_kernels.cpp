#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "memory_manager.h" // Needed to allocate memory for tests
#include "layers.h"
#include "cuda_utils.h"

// --- Helper for approximate float comparison ---
bool is_close(float a, float b, float tol = 1e-3) {
    return std::abs(a - b) < tol;
}

// ==========================================
// TEST 1: RMSNorm Kernel
// ==========================================
TEST(Kernels, RMSNorm) {
    int dim = 128; // Small dim for testing
    float eps = 1e-5f;
    MemoryManager mm(1024 * 1024);

    float* d_in = mm.allocate<float>(dim);
    float* d_w  = mm.allocate<float>(dim);
    float* d_out = mm.allocate<float>(dim);

    // Prepare Host Data (All 1s)
    std::vector<float> h_in(dim, 1.0f);   
    std::vector<float> h_w(dim, 1.0f);    // Gamma = 1
    
    cudaMemcpy(d_in, h_in.data(), dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), dim*sizeof(float), cudaMemcpyHostToDevice);

    // Launch
    launch_rmsnorm(d_out, d_in, d_w, 1, dim, eps);
    cudaDeviceSynchronize();

    // Verify
    // RMS of [1, 1, 1...] is 1.0. 
    // Output = x / RMS = 1.0.
    std::vector<float> h_out(dim);
    cudaMemcpy(h_out.data(), d_out, dim*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<dim; i++) {
        ASSERT_TRUE(is_close(h_out[i], 1.0f));
    }
}

// ==========================================
// TEST 2: RoPE (Rotation)
// ==========================================
TEST(Kernels, RoPE) {
    int head_dim = 128;
    int n_heads = 1;
    MemoryManager mm(1024 * 1024);

    float* d_q = mm.allocate<float>(head_dim);
    float* d_k = mm.allocate<float>(head_dim);

    // Input: (1, 0) pairs
    std::vector<float> h_q(head_dim);
    for(int i=0; i<head_dim; i+=2) {
        h_q[i] = 1.0f; h_q[i+1] = 0.0f;
    }
    cudaMemcpy(d_q, h_q.data(), head_dim*sizeof(float), cudaMemcpyHostToDevice);

    // Rotate at Pos 1
    launch_rope(d_q, d_k, 1, head_dim, n_heads, n_heads);
    cudaDeviceSynchronize();

    std::vector<float> h_out(head_dim);
    cudaMemcpy(h_out.data(), d_q, head_dim*sizeof(float), cudaMemcpyDeviceToHost);

    // Check first pair (Index 0, theta = 1.0)
    // Expected: cos(1) ~= 0.54, sin(1) ~= 0.84
    EXPECT_NEAR(h_out[0], cosf(1.0f), 1e-3);
    EXPECT_NEAR(h_out[1], sinf(1.0f), 1e-3);
}

// ==========================================
// TEST 3: SwiGLU (Activation)
// ==========================================
TEST(Kernels, SwiGLU) {
    int dim = 128;
    MemoryManager mm(1024 * 1024);
    float* d_gate = mm.allocate<float>(dim);
    float* d_up = mm.allocate<float>(dim);

    // Gate = 10, Up = 0.5
    std::vector<float> h_val(dim, 10.0f);
    std::vector<float> h_up(dim, 0.5f);
    
    cudaMemcpy(d_gate, h_val.data(), dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up.data(), dim*sizeof(float), cudaMemcpyHostToDevice);

    launch_silu_mul(d_gate, d_up, dim);
    cudaDeviceSynchronize();

    std::vector<float> h_out(dim);
    cudaMemcpy(h_out.data(), d_gate, dim*sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: 10 * sigmoid(10) * 0.5 ~= 5.0
    EXPECT_NEAR(h_out[0], 5.0f, 0.1f);
}

// ==========================================
// TEST 4: FlashAttention
// ==========================================
TEST(Kernels, FlashAttention) {
    int head_dim = 128;
    int seq_len = 2;
    int max_seq = 10;
    MemoryManager mm(1024 * 1024);

    float* d_q = mm.allocate<float>(head_dim);
    float* d_k = mm.allocate<float>(max_seq * head_dim);
    float* d_v = mm.allocate<float>(max_seq * head_dim);
    float* d_out = mm.allocate<float>(head_dim);

    // Set Q = 1
    std::vector<float> ones(head_dim, 1.0f);
    cudaMemcpy(d_q, ones.data(), head_dim*4, cudaMemcpyHostToDevice);

    // Set K = 1, V = 2 for Token 0
    cudaMemcpy(d_k, ones.data(), head_dim*4, cudaMemcpyHostToDevice);
    std::vector<float> twos(head_dim, 2.0f);
    cudaMemcpy(d_v, twos.data(), head_dim*4, cudaMemcpyHostToDevice);

    // Set K = 1, V = 3 for Token 1 (Offset by head_dim)
    cudaMemcpy(d_k + head_dim, ones.data(), head_dim*4, cudaMemcpyHostToDevice);
    std::vector<float> threes(head_dim, 3.0f);
    cudaMemcpy(d_v + head_dim, threes.data(), head_dim*4, cudaMemcpyHostToDevice);

    launch_flash_attention(d_out, d_q, d_k, d_v, 1, 1, head_dim, max_seq, seq_len);
    cudaDeviceSynchronize();

    std::vector<float> res(head_dim);
    cudaMemcpy(res.data(), d_out, head_dim*4, cudaMemcpyDeviceToHost);

    // Expected: Avg(2, 3) = 2.5
    EXPECT_NEAR(res[0], 2.5f, 1e-3);
}