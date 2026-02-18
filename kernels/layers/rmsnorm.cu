#include <cuda_runtime.h>
#include "common/hip_compat.h" // Logic for AMD/NVIDIA
#include "cuda_utils.h"        // Error checking

// ================================================================
// DEVICE CODE (Runs on GPU)
// ================================================================

// WARP_SIZE is 32 for NVIDIA, 64 for AMD
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// 1. Warp Reduction: Sums a value across a warp (32 threads)
// "val" is the value in *this* thread's register.
// Returns the sum of all "val"s in the warp to Lane 0.
__device__ float warpReduceSum(float val) {
    // 0xffffffff means "all threads active"
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 2. The RMSNorm Kernel
// One Block per Row (Token). Threads work together to normalize that row.
__global__ void rmsnorm_kernel(float* __restrict__ out,      // Output
                               const float* __restrict__ in, // Input
                               const float* __restrict__ weight, // Gamma (Scale)
                               int size,                     // Hidden Dim (e.g., 4096)
                               float eps)                    // Epsilon
{
    // My Thread ID within the block
    int tid = threadIdx.x; 
    
    // 1. Calculate Sum of Squares
    // Each thread sums a subset of the row (Grid-Stride Loop)
    float sum_sq = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float x = in[i];
        sum_sq += x * x;
    }

    // 2. Reduce within the Warp
    sum_sq = warpReduceSum(sum_sq);

    // 3. Reduce across Warps (using Shared Memory)
    // If we have 1024 threads, we have 32 warps.
    // Lane 0 of each warp writes its result to shared memory.
    static __shared__ float shared_sum[32]; // Max 1024 threads / 32 = 32 warps
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) {
        shared_sum[wid] = sum_sq;
    }
    
    __syncthreads(); // Wait for all warps to write

    // 4. The Last Warp sums the results of the other warps
    // Only the first warp runs this part
    sum_sq = (tid < blockDim.x / WARP_SIZE) ? shared_sum[lane] : 0.0f;
    
    if (wid == 0) {
        sum_sq = warpReduceSum(sum_sq);
    }

    // 5. Broadcast the result (Inverse Square Root) to all threads
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(sum_sq / size + eps);
    }
    __syncthreads();

    // 6. Write Output
    // Re-read input, multiply by calculated RMS and Weight (Gamma)
    for (int i = tid; i < size; i += blockDim.x) {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

// ================================================================
// HOST CODE (Callable from C++)
// ================================================================

void launch_rmsnorm(float* d_out, 
                    const float* d_in, 
                    const float* d_weight, 
                    int batch_size, 
                    int hidden_dim, 
                    float eps, 
                    cudaStream_t stream = 0) 
{
    // One block per token (row)
    dim3 grid(batch_size);
    
    // Threads per block (Optimization: Tune this!)
    // 256 is a safe sweet spot for occupancy.
    dim3 block(256); 

    // Dynamic Shared Memory size (if needed), here we used static.
    
    rmsnorm_kernel<<<grid, block, 0, stream>>>(d_out, d_in, d_weight, hidden_dim, eps);
    
    CUDA_CHECK_KERNEL();
}