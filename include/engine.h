#pragma once
#include <vector>
#include "config.h"
#include "model_loader.h"
#include "memory_manager.h"
#include "kv_cache.h"
#include "layers.h"

class Engine {
public:
    // Initializes the engine with the loaded weights and config
    Engine(ModelConfig config, LlamaWeights weights);
    ~Engine();

    // The core forward pass. 
    // Takes the current token ID, returns the logits (probabilities) for the next token.
    float* forward(int token, int pos);

    // Cleans up the memory arena for the next step
    void step_cleanup();

private:
    ModelConfig config;
    LlamaWeights weights;
    
    // Core infrastructure
    MemoryManager* allocator;
    KVCache* kv_cache;

    // --- CUDA BLAS Handle (For Matrix Multiplications) ---
    // We use void* here so we don't have to include cublas headers in the engine.h
    // which keeps compile times fast and prevents bleeding NVIDIA specific headers
    // into the rest of the C++ codebase.
    void* cublas_handle; 

    // --- Helper Functions for the Forward Pass ---
    // Performs: out = in * weight
    void matmul(float* out, float* in, float* weight, int in_dim, int out_dim);
    
    // Performs: out = a + b (Residual Connection)
    void add(float* out, float* a, float* b, int size);
};