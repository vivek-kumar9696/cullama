#include <cuda_runtime.h>
#include "common/hip_compat.h" // Logic for AMD/NVIDIA
#include "cuda_utils.h"

// ================================================================
// DEVICE CODE
// ================================================================

// We process data in float2 chunks (64 bits) for bandwidth efficiency
__global__ void rope_kernel(float2* __restrict__ q,   // Query: [batch, heads, seq_len, head_dim/2]
                            float2* __restrict__ k,   // Key:   [batch, kv_heads, seq_len, head_dim/2]
                            int pos,                  // Current Token Position (Time Step)
                            int head_dim,             // e.g., 128 (so 64 float2s)
                            int rotary_dim,           // Usually same as head_dim
                            int n_heads,
                            int n_kv_heads,
                            int seq_len)              
{
    // 1. Calculate our coordinates
    // We launch threads over (Batch, Heads, HeadDim/2)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of pairs to process per token
    int total_pairs_q = n_heads * (head_dim / 2);
    int total_pairs_k = n_kv_heads * (head_dim / 2);

    if (idx >= total_pairs_q && idx >= total_pairs_k) return;

    // 2. Identify which head and which rotation pair we are
    // This logic assumes we are processing the *current* token only (seq_len=1 for inference)
    // If we were processing a prompt (prefill), we'd need a loop over seq_len.
    // Let's assume Inference Mode (1 token) for this kernel for simplicity.
    
    int pair_idx = idx % (head_dim / 2);
    
    // 3. Calculate Frequency (The RoPE Math)
    // theta = 10000^(-2i/d)
    float theta = 1.0f / powf(10000.0f, (2.0f * pair_idx) / (float)head_dim);
    
    // 4. Apply Rotation to Position
    float m_theta = pos * theta;
    float cos_val = cosf(m_theta);
    float sin_val = sinf(m_theta);

    // 5. Rotate Q
    if (idx < total_pairs_q) {
        float2 q_vec = q[idx];
        float2 q_out;
        
        // Rotation Matrix multiplication
        q_out.x = q_vec.x * cos_val - q_vec.y * sin_val;
        q_out.y = q_vec.x * sin_val + q_vec.y * cos_val;
        
        q[idx] = q_out; // Write back
    }

    // 6. Rotate K (If K heads < Q heads, this handles GQA)
    // We need to map the global index to the K-buffer range
    if (idx < total_pairs_k) {
        float2 k_vec = k[idx];
        float2 k_out;

        k_out.x = k_vec.x * cos_val - k_vec.y * sin_val;
        k_out.y = k_vec.x * sin_val + k_vec.y * cos_val;
        
        k[idx] = k_out;
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
    // Total pairs to process (Vectorized / 2)
    // We launch enough threads to cover the larger of Q or K
    int total_threads = n_heads * (head_dim / 2);
    
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    rope_kernel<<<grid_size, block_size, 0, stream>>>(
        (float2*)d_q, 
        (float2*)d_k, 
        pos, 
        head_dim, 
        head_dim, // rotary_dim usually == head_dim
        n_heads, 
        n_kv_heads, 
        1 // seq_len = 1 for token generation
    );
    
    CUDA_CHECK_KERNEL();
}