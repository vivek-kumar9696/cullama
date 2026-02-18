#include <cuda_runtime.h>
#include <float.h>
#include "common/hip_compat.h"
#include "cuda_utils.h"

// ================================================================
// DEVICE CODE
// ================================================================

// Hardcoded for Llama-2/3 standard (Head Dim 128)
// This allows the compiler to unroll loops aggressively.
#define HEAD_DIM 128 
#define WARP_SIZE 32

__global__ void flash_decoding_kernel(
    float* __restrict__ output,      // [batch, heads, head_dim]
    const float* __restrict__ q,     // [batch, heads, head_dim]
    const float* __restrict__ k_cache, // [layers, batch, heads, max_seq, head_dim] (Logical)
    const float* __restrict__ v_cache, // Same
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    int current_seq_len,
    float scale)                     // 1 / sqrt(dim)
{
    // 1. Setup
    // Each block handles ONE Head for ONE Batch item.
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    // GQA (Grouped Query Attention) logic:
    // If we have 32 Q heads and 8 KV heads, 
    // Q heads 0,1,2,3 all share KV head 0.
    int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    // Pointers to this head's data
    // Q is [batch, heads, head_dim]
    int q_offset = batch_idx * n_heads * head_dim + head_idx * head_dim;
    
    // KV Cache is flattened in our simplified Manager:
    // Offset = Layer_Offset + (Heads * SeqLen * Dim) ...
    // For this tutorial, we assume the pointers passed IN are already pointing 
    // to the correct Layer start.
    // We just need to jump to the correct Head.
    int kv_offset = kv_head_idx * max_seq_len * head_dim;

    // 2. Load Query into Registers
    // Each thread loads a chunk of Q.
    // We assume blockDim.x = 128 (one thread per dimension float)
    float q_val = q[q_offset + tid];

    // 3. Online Softmax State
    // m = max_score, d = denominator (sum of exp)
    float m_prev = -FLT_MAX;
    float d_prev = 0.0f;
    float acc_o = 0.0f; // Accumulator for Output

    // 4. Loop over KV Cache (The "Flash" Part)
    // We process one token at a time (inefficient, usually we tile this).
    // For a tutorial, we iterate linearly to prove the math first.
    // Optimization: In Phase 4, we will load K/V in vector chunks.
    
    for (int t = 0; t < current_seq_len; t++) {
        // A. Load K vector for token 't'
        // Ideally, we'd use shared memory here, but let's rely on L2 cache for simplicity first.
        int k_idx = kv_offset + t * head_dim + tid;
        float k_val = k_cache[k_idx];

        // B. Compute Dot Product (Q * K^T)
        // We need to sum across all threads in this block (Reductions).
        // Each thread has q_val[i] * k_val[i]. We sum them up.
        float score_part = q_val * k_val;
        
        // Warp Reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            score_part += __shfl_down_sync(0xffffffff, score_part, offset);
        }
        
        // Block Reduction (across Warps)
        __shared__ float shared_score[4]; // 128 threads / 32 = 4 warps
        int lane = tid % WARP_SIZE;
        int wid = tid / WARP_SIZE;
        
        if (lane == 0) shared_score[wid] = score_part;
        __syncthreads();

        // First warp sums the shared results
        float score = 0.0f;
        if (tid < 4) score = shared_score[tid]; // Only first 4 threads load
        
        // Final Warp Reduce (only first warp active)
        if (wid == 0) {
             for (int offset = 4 / 2; offset > 0; offset /= 2) {
                score += __shfl_down_sync(0xffffffff, score, offset);
            }
        }
        // Broadcast score to all threads
        if (tid == 0) shared_score[0] = score;
        __syncthreads();
        score = shared_score[0];

        // Apply Scaling
        score *= scale;

        // C. Online Softmax Update
        // m_new = max(m_prev, score)
        // d_new = d_prev * exp(m_prev - m_new) + exp(score - m_new)
        float m_new = fmaxf(m_prev, score);
        float exp_score = expf(score - m_new);
        float correction = expf(m_prev - m_new);

        d_prev = d_prev * correction + exp_score;

        // D. Update Output Accumulator
        // O_new = O_prev * correction + V[t] * exp_score
        int v_idx = kv_offset + t * head_dim + tid;
        float v_val = v_cache[v_idx];
        
        acc_o = acc_o * correction + v_val * exp_score;

        // Update max for next iteration
        m_prev = m_new;
    }

    // 5. Final Normalize
    // Output = Acc_O / d_prev
    float out_val = acc_o / d_prev;

    // 6. Write to Global Memory
    output[q_offset + tid] = out_val;
}

// ================================================================
// HOST CODE
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
    dim3 grid(1, n_heads); // Assuming Batch=1
    
    // Block: [Head Dim]
    // Each block calculates the attention for one head.
    // Threads align with the embedding dimension (128).
    dim3 block(head_dim); 

    float scale = 1.0f / sqrtf((float)head_dim);

    flash_decoding_kernel<<<grid, block, 0, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, 
        n_heads, n_kv_heads, head_dim, max_seq_len, current_seq_len, scale
    );
    
    CUDA_CHECK_KERNEL();
}