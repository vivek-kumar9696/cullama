#include "kv_cache.h"
#include <iostream>

KVCache::KVCache(ModelConfig& conf) : config(conf) {
    d_k_cache = nullptr;
    d_v_cache = nullptr;
    current_seq_len = 0;
}

void KVCache::setup(MemoryManager& allocator) {
    // Calculate total elements needed
    // Shape: [layers, batch(1), heads, seq_len, dim]
    // For simplicity, we assume Batch Size = 1 for now.
    
    size_t total_elements = (size_t)config.n_layers * 
                            config.n_kv_heads * 
                            config.seq_len * 
                            config.dim; // dim is usually head_dim here

    // Actually, Llama config usually gives 'dim' as hidden_size (4096).
    // head_dim = dim / n_heads (4096 / 32 = 128).
    int head_dim = config.dim / config.n_heads;
    
    size_t layer_size = config.n_kv_heads * config.seq_len * head_dim;
    size_t total_size = layer_size * config.n_layers;

    std::cout << "[KVCache] Allocating KV Cache..." << std::endl;
    std::cout << "          Elements: " << total_size << " per cache (K & V)" << std::endl;
    std::cout << "          Size: " << (total_size * sizeof(float) * 2) / (1024*1024) << " MB" << std::endl;

    // Allocate K and V from the Arena
    d_k_cache = allocator.allocate<float>(total_size);
    d_v_cache = allocator.allocate<float>(total_size);
}

void KVCache::reset() {
    current_seq_len = 0;
}

void KVCache::advance_pos() {
    if (current_seq_len < config.seq_len) {
        current_seq_len++;
    } else {
        std::cerr << "WARNING: Context Window Full! (Ring buffer logic needed here)" << std::endl;
        // In a real ring buffer, we'd wrap around: current_seq_len = 0;
    }
}

// Pointer Math: Where does Layer L's cache start?
size_t KVCache::get_layer_offset(int layer) {
    int head_dim = config.dim / config.n_heads;
    // Offset = Layer * (Heads * SeqLen * HeadDim)
    return (size_t)layer * config.n_kv_heads * config.seq_len * head_dim;
}

float* KVCache::get_k_ptr(int layer, int batch_idx) {
    // Returns the start of the cache for this layer.
    // The Attention Kernel will handle the [head, seq_pos] indexing.
    return d_k_cache + get_layer_offset(layer);
}

float* KVCache::get_v_ptr(int layer, int batch_idx) {
    return d_v_cache + get_layer_offset(layer);
}