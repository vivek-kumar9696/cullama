#include <cuda_runtime.h>
#include "common/hip_compat.h"
#include "cuda_utils.h"

// ================================================================
// DEVICE CODE
// ================================================================

// Helper: Calculate SiLU for a single float
__device__ __forceinline__ float silu(float x) {
    // 1 / (1 + exp(-x)) is Sigmoid.
    // SiLU = x * Sigmoid(x)
    return x / (1.0f + expf(-x));
}

// The Fused Kernel (Vectorized float4 version)
// Reads 4 floats at a time to maximize bandwidth.
__global__ void silu_mul_kernel(float* __restrict__ gate, // In/Out (Fused)
                                const float* __restrict__ up,   // Input
                                int size) 
{
    // 1. Calculate global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Vectorized path (Process 4 items at once)
    // We cast the float* to float4* to load 128 bits.
    int vectorized_size = size / 4;
    
    if (idx < vectorized_size) {
        // Reinterpret pointers as float4
        float4* gate_vec_ptr = (float4*)gate;
        const float4* up_vec_ptr = (const float4*)up;

        float4 g = gate_vec_ptr[idx]; // Load 4 floats
        float4 u = up_vec_ptr[idx];   // Load 4 floats
        float4 out;

        // Process each component
        out.x = silu(g.x) * u.x;
        out.y = silu(g.y) * u.y;
        out.z = silu(g.z) * u.z;
        out.w = silu(g.w) * u.w;

        // Store back
        gate_vec_ptr[idx] = out;
    }
    
    // Note: If size is not divisible by 4, you'd need a cleanup loop here.
    // For Llama (dim 4096 or 11008), it is always divisible by 4, 
    // so we skip the cleanup for clean interview code.
}

// ================================================================
// HOST CODE
// ================================================================

void launch_silu_mul(float* d_gate, const float* d_up, int size, cudaStream_t stream = 0) {
    // We launch threads for the VECTORIZED size
    int threads = 256;
    int vec_size = size / 4;
    int blocks = (vec_size + threads - 1) / threads;

    silu_mul_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, size);
    
    CUDA_CHECK_KERNEL();
}