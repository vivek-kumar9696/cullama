#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class Tokenizer {
public:
    Tokenizer(const std::string& vocab_file) {
        std::ifstream file(vocab_file, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open tokenizer: " << vocab_file << std::endl;
            exit(1);
        }

        int vocab_size;
        file.read((char*)&vocab_size, sizeof(int));
        vocab.resize(vocab_size);

        for (int i = 0; i < vocab_size; i++) {
            int len;
            file.read((char*)&len, sizeof(int));
            
            std::string word(len, ' ');
            file.read(&word[0], len);
            
            vocab[i] = word;
        }
        std::cout << "[Tokenizer] Loaded " << vocab_size << " tokens." << std::endl;
    }

 std::string decode(int token_id) {
        if (token_id >= 0 && token_id < vocab.size()) {
            std::string text = vocab[token_id];
            
            // SentencePiece space is Unicode U+2581 (e2 96 81 in hex)
            std::string block = "\xe2\x96\x81";
            size_t pos;
            while ((pos = text.find(block)) != std::string::npos) {
                text.replace(pos, block.length(), " ");
            }
            
            // Handle newlines natively
            if (text == "<0x0A>") return "\n";
            
            return text;
        }
        return "";
    }

private:
    std::vector<std::string> vocab;
};