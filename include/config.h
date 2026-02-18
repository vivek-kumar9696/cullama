#pragma once

// The Blueprint of the Model
struct ModelConfig {
    int dim = 4096;        // Transformer dimension (Llama-2-7b)
    int hidden_dim = 11008; // SwiGLU hidden dimension
    int n_layers = 32;     // Number of layers
    int n_heads = 32;      // Number of attention heads
    int n_kv_heads = 32;   // Number of KV heads (GQA)
    int vocab_size = 32000;
    int seq_len = 2048;    // Max context length
    
    // For now, we hardcode Llama-2-7b values.
    // In Phase 3, we will load these from JSON.
};