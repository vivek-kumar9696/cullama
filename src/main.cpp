#include <iostream>
#include <vector>
#include <algorithm>
#include "model_loader.h"
#include "engine.h"
#include "cuda_utils.h"

// Temporary naive Argmax sampler on CPU
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
    if (argc < 2) {
        std::cerr << "Usage: ./cuLlama <path_to_model.bin>" << std::endl;
        return 1;
    }

    // 1. Mount the Weights (Zero-Copy)
    ModelLoader loader(argv[1]);
    if (!loader.load()) return 1;

    // 2. Initialize the Engine
    Engine engine(loader.get_config(), loader.get_weights());
    std::cout << "[App] Engine Initialized. Ready for inference." << std::endl;

    // 3. The Inference Loop
    int prompt_token = 1; // Assume token 1 is <s> (Begin of Sequence)
    int current_token = prompt_token;
    
    std::cout << "\n--- Generating Tokens ---" << std::endl;
    std::cout << "Token IDs: " << current_token << " ";

    for (int pos = 0; pos < 10; pos++) { // Generate 10 tokens
        // A. Forward Pass
        float* d_logits = engine.forward(current_token, pos);
        
        // Wait for GPU to finish computing
        CUDA_CHECK(cudaDeviceSynchronize());

        // B. Sample next token
        int next_token = argmax(d_logits, loader.get_config().vocab_size);
        std::cout << next_token << " " << std::flush;

        // C. Clean up Arena for the next step
        engine.step_cleanup();
        
        // D. Update state
        current_token = next_token;
    }
    
    std::cout << "\n\n[App] Generation Complete." << std::endl;

    return 0;
}