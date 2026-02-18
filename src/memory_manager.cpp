#include "memory_manager.h"
#include <iostream>

MemoryManager::MemoryManager(size_t total_size_bytes) {
    this->total_size = total_size_bytes;
    this->offset = 0;

    std::cout << "[MemoryManager] Allocating " 
              << (total_size_bytes / 1024 / 1024) 
              << " MB Scratchpad Arena..." << std::endl;

    // The ONLY time we call the driver's expensive malloc
    CUDA_CHECK(cudaMalloc(&d_base, total_size));
    
    // Zero it out just to be safe (optional, costs time)
    CUDA_CHECK(cudaMemset(d_base, 0, total_size));
}

MemoryManager::~MemoryManager() {
    if (d_base) {
        std::cout << "[MemoryManager] Teardown: Freeing Arena." << std::endl;
        CUDA_CHECK(cudaFree(d_base));
    }
}

void MemoryManager::reset() {
    // This is the magic. 
    // Instead of free()ing 100 tensors, we just set one integer to 0.
    offset = 0;
}