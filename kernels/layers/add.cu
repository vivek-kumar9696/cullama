#include <cuda_runtime.h>
#include "common/hip_compat.h"
#include "cuda_utils.h"

// Fused Add: out = a + b
__global__ void add_kernel(float* __restrict__ out, 
                           const float* __restrict__ a, 
                           const float* __restrict__ b, 
                           int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Host Wrapper
void launch_add(float* d_out, float* d_a, float* d_b, int size, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(d_out, d_a, d_b, size);
    CUDA_CHECK_KERNEL();
}