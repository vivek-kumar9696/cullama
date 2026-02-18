#pragma once
#include <cstddef> // for size_t
#include <iostream>
#include "cuda_utils.h" // For error checking macros

class MemoryManager {
public:
    // Constructor: Allocates the massive GPU buffer once.
    MemoryManager(size_t total_size_bytes);
    
    // Destructor: Frees the massive buffer.
    ~MemoryManager();

    // The "Fast Malloc". Returns a pointer into our pre-allocated arena.
    // Does NOT verify if we have run out of space (for speed).
    // In production, you'd add a check here.
    template <typename T>
    T* allocate(size_t num_elements) {
        size_t bytes = num_elements * sizeof(T);
        
        // 1. Align memory to 256 bytes (best for GPU memory coalescing)
        size_t aligned_bytes = (bytes + 255) & ~255; 

        // 2. Check overflow (optional, good for debugging)
        if (offset + aligned_bytes > total_size) {
            std::cerr << "OOM: Scratchpad Arena Overflow!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // 3. Calculate the pointer
        // d_base is a void*, so we cast to char* to do pointer math
        void* ptr = (char*)d_base + offset;

        // 4. Advance the offset
        offset += aligned_bytes;

        return (T*)ptr;
    }

    // Call this at the end of every token generation step.
    // It "frees" everything instantly by just resetting the counter.
    void reset();

private:
    void* d_base;      // The start of our GPU memory block
    size_t offset;     // The current "cursor" position
    size_t total_size; // Total capacity
};