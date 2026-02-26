#include <cuda_runtime.h>
#include <float.h>
#include "common/hip_compat.h"
#include "cuda_utils.h"

// ================================================================
// DEVICE CODE (FlashAttention Decoding Kernel)
// ================================================================

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void flash_decoding_kernel(
    float* __restrict__ output,      //[batch, heads, head_dim]
    const float* __restrict__ q,     //[batch, heads, head_dim]
    const float* __restrict__ k_cache, //[layers, batch, heads, max_seq, head_dim]
    const float* __restrict__ v_cache, 
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    int current_seq_len,
    float scale)                     
{
    // 1. Setup Grid & Thread Identifiers
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    // GQA (Grouped Query Attention) Mapping
    int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    // Calculate memory offsets
    int q_offset = batch_idx * n_heads * head_dim + head_idx * head_dim;
    int kv_offset = kv_head_idx * max_seq_len * head_dim;

    // 2. Load Query for this thread
    float q_val = q[q_offset + tid];

    // 3. Online Softmax State Registers
    float m_prev = -FLT_MAX; // max
    float d_prev = 0.0f;     // denominator
    float acc_o = 0.0f;      // output accumulator

    // Determine how many warps are running dynamically 
    // (For TinyLlama head_dim=64, this will be 2 warps)
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // 4. Iterate over the KV Cache (Token by Token)
    for (int t = 0; t < current_seq_len; t++) {
        
        // A. Load K
        int k_idx = kv_offset + t * head_dim + tid;
        float k_val = k_cache[k_idx];

        // B. Compute Partial Score
        float score_part = q_val * k_val;
        
        // --- WARP REDUCTION ---
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            score_part += __shfl_down_sync(0xffffffff, score_part, offset);
        }
        
        // --- BLOCK REDUCTION (Dynamic & Safe) ---
        // Max block size usually 1024 threads = 32 warps.
        __shared__ float shared_score[32]; 
        
        int lane = tid % WARP_SIZE;
        int wid = tid / WARP_SIZE;
        
        // Lane 0 of each warp writes its partial sum to shared memory
        if (lane == 0) {
            shared_score[wid] = score_part;
        }
        __syncthreads(); // Wait for all warps to write

        // Thread 0 safely sums exactly the number of active warps
        float score = 0.0f;
        if (tid == 0) {
            for(int i = 0; i < num_warps; i++) {
                score += shared_score[i];
            }
            shared_score[0] = score; // Store final sum at index 0
        }
        __syncthreads(); // Wait for Thread 0 to finish summing
        
        // Broadcast the final score to all threads in the block
        score = shared_score[0];

        // Apply sqrt(d) scaling
        score *= scale;

        // C. Online Softmax Update (Registers only, no HBM writes!)
        float m_new = fmaxf(m_prev, score);
        float exp_score = expf(score - m_new);
        float correction = expf(m_prev - m_new);

        d_prev = d_prev * correction + exp_score;

        // D. Load V and Update Output Accumulator
        int v_idx = kv_offset + t * head_dim + tid;
        float v_val = v_cache[v_idx];
        
        acc_o = acc_o * correction + v_val * exp_score;

        // Save max for the next token iteration
        m_prev = m_new;
    }

    // 5. Finalize Softmax
    float out_val = acc_o / d_prev;

    // 6. Write to Global Memory (Once per layer)
    output[q_offset + tid] = out_val;
}

// ================================================================
// HOST CODE (C++ Wrapper)
// ================================================================

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
) {
    // Grid: [Batch Size, Number of Heads]
    dim3 grid(1, n_heads); // Assuming Batch=1 for inference
    
    // Block: [Head Dim]
    // Threads align exactly with the embedding dimension (e.g., 64 or 128)
    dim3 block(head_dim); 

    float scale = 1.0f / sqrtf((float)head_dim);

    flash_decoding_kernel<<<grid, block, 0, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, 
        n_heads, n_kv_heads, head_dim, max_seq_len, current_seq_len, scale
    );
    
    CUDA_CHECK_KERNEL();
}