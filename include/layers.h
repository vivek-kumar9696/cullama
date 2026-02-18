#pragma once
#include <cuda_runtime.h>

// RMSNorm
void launch_rmsnorm(float* d_out, 
                    const float* d_in, 
                    const float* d_weight, 
                    int batch_size, 
                    int hidden_dim, 
                    float eps, 
                    cudaStream_t stream = 0);

// RoPE (Rotary Embeddings)
void launch_rope(float* d_q, 
                 float* d_k, 
                 int pos, 
                 int head_dim, 
                 int n_heads, 
                 int n_kv_heads, 
                 cudaStream_t stream = 0);

// SwiGLU (SiLU * Multiply)
void launch_silu_mul(float* d_gate, 
                    const float* d_up, 
                    int size, 
                    cudaStream_t stream = 0);

// Flash Attention (Decoding / Query Length = 1)
void launch_flash_attention(
    float* d_out,
    const float* d_q,
    const float* d_k_cache,
    const float* d_v_cache,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    int current_seq_len,
    cudaStream_t stream = 0
);