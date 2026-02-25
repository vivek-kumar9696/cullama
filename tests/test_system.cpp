#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <cstdio> // for std::remove

#include "memory_manager.h"
#include "model_loader.h"

// ==========================================
// TEST 1: Memory Arena (Host/Device Allocation)
// ==========================================
TEST(System, MemoryManager_AllocationAndReset) {
    size_t size = 1024 * 1024; // 1MB
    MemoryManager allocator(size);

    float* p1 = allocator.allocate<float>(100);
    float* p2 = allocator.allocate<float>(100);

    // Pointers should be different
    ASSERT_NE(p1, p2);
    // P2 should be ahead of P1
    ASSERT_GT(p2, p1);

    // Reset
    allocator.reset();
    float* p3 = allocator.allocate<float>(100);

    // After reset, p3 should point to the start (same as p1)
    ASSERT_EQ(p1, p3);
}

// ==========================================
// TEST 2: Model Loader (Zero-Copy mmap)
// ==========================================
TEST(System, ModelLoader) {
    std::string mock_file = "mock_tiny_model.bin";
    
    // 1. Create a fake model file dynamically
    std::ofstream out(mock_file, std::ios::binary);
    ASSERT_TRUE(out.is_open()) << "Failed to create mock file.";

    // Header layout: Magic, dim, hidden_dim, layers, heads, kv_heads, vocab, seq_len
    int magic = 0x4C4C414D;
    int dim = 128;
    int hidden_dim = 256;
    int layers = 2;
    int heads = 4;
    int kv_heads = 4;
    int vocab = 1000;
    int seq_len = 2048;

    out.write((char*)&magic, sizeof(int));
    out.write((char*)&dim, sizeof(int));
    out.write((char*)&hidden_dim, sizeof(int));
    out.write((char*)&layers, sizeof(int));
    out.write((char*)&heads, sizeof(int));
    out.write((char*)&kv_heads, sizeof(int));
    out.write((char*)&vocab, sizeof(int));
    out.write((char*)&seq_len, sizeof(int));

    // Write 1MB of dummy zeroes to simulate weight data
    std::vector<char> dummy_weights(1024 * 1024, 0);
    out.write(dummy_weights.data(), dummy_weights.size());
    out.close();

    // 2. Test the Loader
    ModelLoader loader(mock_file);
    bool success = loader.load();
    
    // 3. Assertions
    EXPECT_TRUE(success) << "Loader failed to open and map the file.";
    
    ModelConfig config = loader.get_config();
    EXPECT_EQ(config.dim, 128);
    EXPECT_EQ(config.hidden_dim, 256);
    EXPECT_EQ(config.n_layers, 2);
    EXPECT_EQ(config.n_heads, 4);
    EXPECT_EQ(config.vocab_size, 1000);

    // 4. Cleanup
    std::remove(mock_file.c_str());
}