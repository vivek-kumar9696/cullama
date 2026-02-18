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