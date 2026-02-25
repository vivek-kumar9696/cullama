#include "engine.h"
#include "cuda_utils.h"
#include <iostream>
#include <cublas_v2.h>

Engine::Engine(ModelConfig conf, LlamaWeights w) : config(conf), weights(w) {
    std::cout << "[Engine] Initializing Execution Environment..." << std::endl;

    // 1. Initialize the Scratchpad Memory Arena
    // 500MB is more than enough for inference batch size 1
    size_t arena_size = 500ULL * 1024 * 1024; 
    allocator = new MemoryManager(arena_size);

    // 2. Initialize the KV Cache
    kv_cache = new KVCache(config);
    kv_cache->setup(*allocator);

    // 3. Initialize High-Performance Matrix Math (cuBLAS)
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublas_handle = (void*)handle; // Hide it as void* in header
}

Engine::~Engine() {
    cublasDestroy((cublasHandle_t)cublas_handle);
    delete kv_cache;
    delete allocator;
}

// Helper: Matrix Multiplication out = in * weight
// PyTorch Linear Weights are stored as [out_features, in_features] in row-major.
void Engine::matmul(float* out, float* in, float* weight, int in_dim, int out_dim) {
    cublasHandle_t handle = (cublasHandle_t)cublas_handle;
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS is Column-Major. C++ is Row-Major.
    // To compute: C = A * W^T
    // We tell cuBLAS to do: C^T = W * A^T
    // CUBLAS_OP_T means "Transpose this matrix before multiplying"
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N, 
                out_dim, 1, in_dim, // m, n, k
                &alpha, 
                weight, in_dim,     // A (Weights)
                in, in_dim,         // B (Input)
                &beta, 
                out, out_dim);      // C (Output)
}

void Engine::step_cleanup() {
    // Instantaneous free of all tensors generated in the forward pass
    allocator->reset();
    kv_cache->advance_pos();
}

float* Engine::forward(int token_id, int pos) {
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int n_heads = config.n_heads;
    int n_kv_heads = config.n_kv_heads;
    int head_dim = dim / n_heads;
    float eps = 1e-5f;

    // 1. Token Embedding (Lookup)
    // We grab the specific row from the embedding table.
    float* x = allocator->allocate<float>(dim);
    float* embed_row = weights.token_embedding_table + (token_id * dim);
    CUDA_CHECK(cudaMemcpy(x, embed_row, dim * sizeof(float), cudaMemcpyDeviceToDevice));

    // 2. Transformer Layers Loop
    for (int l = 0; l < config.n_layers; l++) {
        
        // --- A. Attention Block ---
        // 1. Pre-Norm
        float* x_norm = allocator->allocate<float>(dim);
        launch_rmsnorm(x_norm, x, weights.rms_att_weight[l], 1, dim, eps);

        // 2. Q, K, V Projections (MatMuls)
        float* q = allocator->allocate<float>(dim);
        float* k = allocator->allocate<float>(n_kv_heads * head_dim);
        float* v = allocator->allocate<float>(n_kv_heads * head_dim);

        matmul(q, x_norm, weights.wq[l], dim, dim);
        matmul(k, x_norm, weights.wk[l], dim, n_kv_heads * head_dim);
        matmul(v, x_norm, weights.wv[l], dim, n_kv_heads * head_dim);

        // 3. Rotary Positional Embeddings (RoPE)
        launch_rope(q, k, pos, head_dim, n_heads, n_kv_heads);

        // 4. Update KV Cache (Copy K and V into the pre-allocated Arena state)
        // Offset for this specific token's position
        float* layer_k_cache = kv_cache->get_k_ptr(l, 0); 
        float* layer_v_cache = kv_cache->get_v_ptr(l, 0);
        int cache_offset = pos * head_dim; // Simplified for single token

        // (In a highly optimized engine, we'd write directly to the cache, 
        // but explicit copy is fine for this architecture proof).
        for(int h = 0; h < n_kv_heads; h++) {
            float* dest_k = layer_k_cache + (h * config.seq_len * head_dim) + cache_offset;
            float* dest_v = layer_v_cache + (h * config.seq_len * head_dim) + cache_offset;
            
            float* src_k = k + (h * head_dim);
            float* src_v = v + (h * head_dim);

            CUDA_CHECK(cudaMemcpy(dest_k, src_k, head_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(dest_v, src_v, head_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // 5. Flash Attention
        float* attn_out = allocator->allocate<float>(dim);
        launch_flash_attention(attn_out, q, layer_k_cache, layer_v_cache, 
                               n_heads, n_kv_heads, head_dim, config.seq_len, pos + 1);

        // 6. Output Projection & Residual Add
        float* proj_out = allocator->allocate<float>(dim);
        matmul(proj_out, attn_out, weights.wo[l], dim, dim);
        
        launch_add(x, x, proj_out, dim); // x = x + proj_out

        
        // --- B. Feed Forward Block (MLP) ---
        // 1. Post-Attention Norm
        float* ffn_norm = allocator->allocate<float>(dim);
        launch_rmsnorm(ffn_norm, x, weights.rms_ffn_weight[l], 1, dim, eps);

        // 2. SwiGLU (Gate & Up)
        float* gate = allocator->allocate<float>(hidden_dim);
        float* up = allocator->allocate<float>(hidden_dim);
        
        matmul(gate, ffn_norm, weights.w_gate[l], dim, hidden_dim);
        matmul(up, ffn_norm, weights.w_up[l], dim, hidden_dim);

        // 3. Fused Activation (SwiGLU)
        launch_silu_mul(gate, up, hidden_dim); // In-place modifies 'gate'

        // 4. Down Projection & Residual Add
        float* down = allocator->allocate<float>(dim);
        matmul(down, gate, weights.w_down[l], hidden_dim, dim);

        launch_add(x, x, down, dim); // x = x + down
    }

    // 3. Final Norm
    float* final_norm = allocator->allocate<float>(dim);
    launch_rmsnorm(final_norm, x, weights.rms_final_weight, 1, dim, eps);

    // 4. LM Head (Logits)
    float* logits = allocator->allocate<float>(config.vocab_size);
    matmul(logits, final_norm, weights.w_cls, dim, config.vocab_size);

    return logits;
}