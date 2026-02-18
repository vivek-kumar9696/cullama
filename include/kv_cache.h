#pragma once
#include <vector>
#include "config.h"
#include "memory_manager.h"

// The State Manager for the LLM
class KVCache {
public:
    // Initialize the cache structure (pointers are null until setup() is called)
    KVCache(ModelConfig& config);

    // Allocate the massive buffer using our MemoryManager Arena
    void setup(MemoryManager& allocator);

    // Get the pointer to the K-Cache for a specific layer
    // We use float* for now. In Phase 3 (Kernels), we switch to half/fp16.
    float* get_k_ptr(int layer, int batch_idx);
    
    // Get the pointer to the V-Cache for a specific layer
    float* get_v_ptr(int layer, int batch_idx);

    // Update the current sequence position (cursor)
    void advance_pos();
    
    // Reset cursor (for new prompt)
    void reset();

    // Get current sequence length
    int get_current_seq_len() const { return current_seq_len; }

private:
    ModelConfig config;
    
    // The massive contiguous buffers in GPU memory
    float* d_k_cache; 
    float* d_v_cache;

    int current_seq_len; // How many tokens have we generated so far?
    
    // Helper to calculate offsets
    size_t get_layer_offset(int layer);
};