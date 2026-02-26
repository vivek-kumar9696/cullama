#include <iostream>
#include <vector>
#include "model_loader.h"
#include "engine.h"
#include "tokenizer.h"
#include "cuda_utils.h"

int argmax(float* d_logits, int vocab_size) {
    std::vector<float> h_logits(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    int best_id = 0;
    float max_val = h_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (h_logits[i] > max_val) {
            max_val = h_logits[i];
            best_id = i;
        }
    }
    return best_id;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./cuLlama <model.bin> <tokenizer.bin> <token1> <token2> ..." << std::endl;
        return 1;
    }

    ModelLoader loader(argv[1]);
    if (!loader.load()) return 1;

    Tokenizer tokenizer(argv[2]);
    Engine engine(loader.get_config(), loader.get_weights());
    
    std::cout << "\n[App] Engine Ready. Generating...\n" << std::endl;

    // --- READ PROMPT FROM COMMAND LINE ---
    std::vector<int> prompt;
    for (int i = 3; i < argc; i++) {
        prompt.push_back(std::stoi(argv[i]));
    }

    int pos = 0;

    // 1. PREFILL PHASE
    for (size_t i = 0; i < prompt.size() - 1; i++) {
        int token = prompt[i];
        
        // Print the prompt cleanly
        if (token != 1 && token != 2) std::cout << tokenizer.decode(token) << std::flush;
        
        engine.forward(token, pos);
        engine.step_cleanup();
        pos++;
    }

    // 2. GENERATION PHASE
    int current_token = prompt.back();
    std::cout << tokenizer.decode(current_token) << std::flush;

    for (int step = 0; step < 30; step++) { 
        float* d_logits = engine.forward(current_token, pos);
        CUDA_CHECK(cudaDeviceSynchronize());

        int next_token = argmax(d_logits, loader.get_config().vocab_size);
        
        std::cout << tokenizer.decode(next_token) << std::flush;

        engine.step_cleanup();
        current_token = next_token;
        pos++;
    }
    
    std::cout << "\n\n[App] Complete." << std::endl;
    return 0;
}