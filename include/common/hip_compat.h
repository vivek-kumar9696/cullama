#pragma once

// ================================================================
// THE ABSTRACTION LAYER
// Maps NVIDIA CUDA calls to AMD HIP calls.
// This allows single-source compilation for both vendors.
// ================================================================

#ifdef USE_ROCM
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>

    // 1. Error Handling
    #define cudaSuccess hipSuccess
    #define cudaError_t hipError_t
    #define cudaGetErrorString hipGetErrorString

    // 2. Memory Management
    #define cudaMalloc hipMalloc
    #define cudaMemcpy hipMemcpy
    #define cudaFree hipFree
    #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
    #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define cudaMemset hipMemset

    // 3. Kernel Launch & Synchronization
    #define cudaDeviceSynchronize hipDeviceSynchronize
    #define cudaStream_t hipStream_t
    #define cudaStreamSynchronize hipStreamSynchronize

    // 4. Atomic Operations (HIP uses standard atomicAdd)
    // No mapping needed usually, but good to know.

#else
    // Native CUDA path
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
#endif