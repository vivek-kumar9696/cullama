#include <cuda_runtime.h>
#include "common/hip_compat.h"
#include "cuda_utils.h"

// ================================================================
// DEVICE CODE (HuggingFace Llama Layout)
// ================================================================

__global__ void rope_kernel(float* __restrict__ q,   
                            float* __restrict__ k,   
                            int pos,                  
                            int head_dim,             
                            int n_heads,
                            int n_kv_heads)              
{
    // Each thread processes ONE pair: (i) and (i + head_dim/2)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int half_dim = head_dim / 2;
    int total_q_pairs = n_heads * half_dim;
    int total_k_pairs = n_kv_heads * half_dim;

    if (idx >= total_q_pairs && idx >= total_k_pairs) return;

    // Identify which head and which pair index within the head we are
    int head_idx = idx / half_dim;
    int pair_idx = idx % half_dim;
    
    // Calculate Frequency
    float theta = 1.0f / powf(10000.0f, (2.0f * pair_idx) / (float)head_dim);
    float m_theta = pos * theta;
    float cos_val = cosf(m_theta);
    float sin_val = sinf(m_theta);

    // Rotate Q (HuggingFace Layout: pairs are separated by half_dim)
    if (idx < total_q_pairs) {
        int q_idx = head_idx * head_dim + pair_idx;
        
        float q0 = q[q_idx];
        float q1 = q[q_idx + half_dim];

        q[q_idx]            = q0 * cos_val - q1 * sin_val;
        q[q_idx + half_dim] = q0 * sin_val + q1 * cos_val;
    }

    // Rotate K
    if (idx < total_k_pairs) {
        int k_idx = head_idx * head_dim + pair_idx;
        
        float k0 = k[k_idx];
        float k1 = k[k_idx + half_dim];

        k[k_idx]            = k0 * cos_val - k1 * sin_val;
        k[k_idx + half_dim] = k0 * sin_val + k1 * cos_val;
    }
}

// ================================================================
// HOST CODE
// ================================================================

void launch_rope(float* d_q, 
                 float* d_k, 
                 int pos, 
                 int head_dim, 
                 int n_heads, 
                 int n_kv_heads, 
                 cudaStream_t stream = 0) 
{
    // We launch threads for the HALF dim
    int total_threads = n_heads * (head_dim / 2);
    
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    rope_kernel<<<grid_size, block_size, 0, stream>>>(
        d_q, d_k, pos, head_dim, n_heads, n_kv_heads
    );
    
    CUDA_CHECK_KERNEL();
}