#include "model_loader.h"
#include <iostream>
#include <fcntl.h>    // open
#include <unistd.h>   // close, lseek
#include <sys/mman.h> // mmap
#include <sys/stat.h> // fstat

ModelLoader::ModelLoader(const std::string& p) : path(p), fd(-1), data(nullptr) {}

ModelLoader::~ModelLoader() {
    if (data != MAP_FAILED && data != nullptr) {
        munmap(data, file_size);
    }
    if (fd != -1) {
        close(fd);
    }
}

bool ModelLoader::load() {
    std::cout << "[Loader] Opening " << path << "..." << std::endl;
    
    // 1. Open File
    fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error opening file!" << std::endl;
        return false;
    }

    // 2. Get File Size
    struct stat sb;
    if (fstat(fd, &sb) == -1) return false;
    file_size = sb.st_size;

    // 3. MMAP (The Magic)
    // We map the file as "Read Only" directly into virtual address space.
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        std::cerr << "mmap failed!" << std::endl;
        return false;
    }
    
    // 4. Parse Header
    int* ptr = (int*)data;
    if (*ptr != 0x4C4C414D) { // Magic Check
    }
    ptr++; // Skip Magic

    config.dim = *ptr++;
    config.hidden_dim = *ptr++;
    config.n_layers = *ptr++;
    config.n_heads = *ptr++;
    config.n_kv_heads = *ptr++;
    config.vocab_size = *ptr++;
    config.seq_len = *ptr++;

    std::cout << "  - Config: Layers=" << config.n_layers 
              << ", Dim=" << config.dim 
              << ", Heads=" << config.n_heads << std::endl;

    // 5. Map Pointers
    float* f_ptr = (float*)ptr; // Start of weights
    
    // Helper to advance pointer
    auto get_tensor = [&](size_t size) -> float* {
        float* t = f_ptr;
        f_ptr += size;
        return t;
    };

    // A. Embeddings
    weights.token_embedding_table = get_tensor(config.vocab_size * config.dim);

    // B. Layers
    weights.rms_att_weight.resize(config.n_layers);
    weights.wq.resize(config.n_layers);
    weights.wk.resize(config.n_layers);
    weights.wv.resize(config.n_layers);
    weights.wo.resize(config.n_layers);
    weights.rms_ffn_weight.resize(config.n_layers);
    weights.w_gate.resize(config.n_layers);
    weights.w_up.resize(config.n_layers);
    weights.w_down.resize(config.n_layers);

    for(int i=0; i<config.n_layers; i++) {
        // Attention
        int head_dim = config.dim / config.n_heads;
        // Shapes differ for GQA, but TinyLlama usually has n_heads == n_kv_heads?
        // Actually TinyLlama 1.1B uses GQA (Grouped Query Attention).
        // WQ: [dim, dim]
        // WK: [dim, n_kv_heads * head_dim]
        
        long long layer_dim = config.dim;
        long long kv_dim = config.n_kv_heads * head_dim;

        weights.rms_att_weight[i] = get_tensor(layer_dim);
        weights.wq[i] = get_tensor(layer_dim * layer_dim);
        weights.wk[i] = get_tensor(layer_dim * kv_dim);
        weights.wv[i] = get_tensor(layer_dim * kv_dim);
        weights.wo[i] = get_tensor(layer_dim * layer_dim);

        // MLP
        weights.rms_ffn_weight[i] = get_tensor(layer_dim);
        weights.w_gate[i] = get_tensor(layer_dim * config.hidden_dim);
        weights.w_up[i]   = get_tensor(layer_dim * config.hidden_dim);
        weights.w_down[i] = get_tensor(layer_dim * config.hidden_dim);
    }

    // C. Final
    weights.rms_final_weight = get_tensor(config.dim);
    weights.w_cls = get_tensor(config.vocab_size * config.dim);

    std::cout << "[Loader] Weights Mapped Successfully." << std::endl;
    return true;
}