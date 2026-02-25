#pragma once
#include <string>
#include <vector>
#include "config.h"

struct LlamaWeights {
    // Pointers into the mmap-ed file
    float* token_embedding_table; // [vocab, dim]
    
    // Arrays of pointers (one per layer)
    std::vector<float*> rms_att_weight; // [layer][dim]
    std::vector<float*> wq;             // [layer][dim, dim]
    std::vector<float*> wk;
    std::vector<float*> wv;
    std::vector<float*> wo;
    
    std::vector<float*> rms_ffn_weight;
    std::vector<float*> w_gate;
    std::vector<float*> w_up;
    std::vector<float*> w_down;

    float* rms_final_weight;
    float* w_cls; // Classifier / LM Head
};

class ModelLoader {
public:
    ModelLoader(const std::string& path);
    ~ModelLoader();

    // Loads the file via mmap
    bool load();

    // Returns the populated pointers
    LlamaWeights& get_weights() { return weights; }
    ModelConfig& get_config() { return config; }

private:
    std::string path;
    int fd;         // File Descriptor
    void* data;     // Pointer to mmap-ed data
    size_t file_size;
    
    ModelConfig config;
    LlamaWeights weights;
};