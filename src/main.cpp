#include <iostream>
#include "cuda_utils.h"

int main(int argc, char** argv) {
    std::cout << "[cuLlama] Inference Engine Ready." << std::endl;
    std::cout << "Run './unit_tests' to verify kernels." << std::endl;
    return 0;
}