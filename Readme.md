# cuLlama 🦙🚀

> **A Bare-Metal, High-Performance LLM Inference Engine in C++ and CUDA/HIP.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20%7C%20AMD-green)](https://developer.nvidia.com/cuda-zone)

**cuLlama** is a project to understand *exactly* what happens inside the GPU when an LLM generates text. It strips away the massive overhead of PyTorch and Python to implement a raw inference loop.

There are no dependencies on `torch`, `accelerate`, or `huggingface`. Just C++, CMake, and raw Kernel code.

---

## 🎯 Mission: "Memory Sovereignty"

The aim of **cuLlama** is to prove control over the hardware.

1.  **Manual Memory Management:** We do not use a Garbage Collector. We allocate a single contiguous block of GPU memory (Arena) at startup and manually manage pointers to avoid fragmentation and allocation overhead.
2.  **Kernel Fusion:** We replace PyTorch's 100+ tiny kernel launches per layer with fused kernels (RMSNorm, SwiGLU, RoPE) to keep the GPU compute-bound, not latency-bound.
3.  **Dual-Backend Compilation:** The codebase is designed to compile for **NVIDIA (CUDA)** and **AMD (ROCm/HIP)** from a single source using a custom abstraction layer.

---

## 🏗️ Architecture

The project is structured to separate the **Host (CPU)** logic from the **Device (GPU)** execution.

```text
cuLlama/
├── CMakeLists.txt                  # The Build System (Critical for C++)
├── README.md                       # "Architecture & Benchmarks"
│
├── src/                            # THE HOST CODE (C++ Logic)
│   ├── main.cpp                    # Entry point (CLI for text generation)
│   ├── engine.cpp                  # Orchestrates the generation loop (Forward -> Sample -> Cache)
│   ├── model_loader.cppcuLlama/
├── src/                  # HOST CODE (Orchestration)
│   ├── engine.cpp        # The inference loop (Forward -> Sample -> Cache)
│   ├── memory_manager.cpp# [SYSTEMS] Manual GPU Arena & Paged KV Cache
│   └── model_loader.cpp  # mmap() weights directly from disk
│
├── kernels/              # DEVICE CODE (High-Performance Math)
│   ├── attention/        # FlashAttention & PagedAttention implementations
│   ├── layers/           # Fused kernels (RMSNorm, SwiGLU)
│   └── common/           # hip_compat.h (The Magic Switch: CUDA <-> HIP)
│
└── include/              # INTERFACES
    ├── kv_cache.h        # Ring Buffer definitions
    └── config.h          # Model Hyperparameters            # mmap() weights from disk (System Call knowledge)
│   ├── memory_manager.cpp          # [SYSTEMS] Manual GPU malloc/free & KV Cache Paging
│   └── sampler.cpp                 # Top-K / Top-P sampling logic (Host side)
│
├── include/                        # HEADERS (Interface Definitions)
│   ├── config.h                    # Model Hyperparams (Llama-2-7b, TinyLlama)
│   ├── layers.h                    # Class definitions for Linear, RMSNorm, Attention
│   ├── kv_cache.h                  # [SYSTEMS] Ring Buffer / Paged Attention logic
│   └── cuda_utils.h                # Error checking macros (CUDA_CHECK)
│
├── kernels/                        # THE DEVICE CODE (CUDA/HIP)
│   ├── attention/
│   │   ├── flash_attention.cu      # [CORE] Custom Tiled Attention Kernel
│   │   ├── paged_attention.cu      # [ADVANCED] Handling non-contiguous KV blocks
│   │   └── rope.cu                 # Rotary Positional Embeddings
│   ├── layers/
│   │   ├── rmsnorm.cu              # Fused RMSNorm (Warp Shuffle Reduction)
│   │   ├── silu_mul.cu             # Fused SwiGLU Activation
│   │   └── softmax.cu              # Fast Softmax
│   ├── quantization/
│   │   ├── int8_dequant.cu         # [AMD OPTIMIZATION] W8A16 Kernel
│   │   └── fp8_utils.cu            # FP8 casting (for future proofing)
│   └── common/
│       └── hip_compat.h            # [AMD] Macros to map cudaMalloc -> hipMalloc
│
├── scripts/                        # PYTHON HELPERS
│   ├── export_weights.py           # PyTorch -> Binary format exporter
│   ├── compare_logits.py           # Debugger: Checks C++ output vs PyTorch
│   └── benchmark.py                # Plot tokens/sec
│
├── third_party/                    # EXTERNAL LIBS
│   ├── cutlass/                    # (Optional) For high-performance GEMMs
│   └── nlohmann_json/              # For config parsing
│
└── tests/                          # UNIT TESTS (GoogleTest)
    ├── test_rmsnorm.cpp            # Verifies Kernel output vs CPU reference
    └── test_kv_cache.cpp           # Verifies memory logic
```
## ⚡ Performance Features

*   **Zero-Copy Loading:** Weights are loaded via `mmap`, allowing the OS to page them in lazily. This avoids massive CPU RAM allocation and double-copying.
*   **Static Allocation (The Arena):** We allocate one contiguous block of VRAM at startup. All tensors (Linear layers, KV Cache) are views into this block. There is **zero** `malloc`/`free` overhead during the inference loop.
*   **Fused Operations:** Instead of launching 100+ small kernels per layer (standard PyTorch behavior), we fuse `RMSNorm + Residual` and `SwiGLU` to keep the GPU compute-bound, not latency-bound.
*   **Platform Agnosticism:** The codebase uses a thin abstraction layer (`hip_compat.h`) to compile native CUDA code for **NVIDIA** or native HIP code for **AMD** without changing the kernel logic.

---

## 🛠️ Build & Usage

### Prerequisites
*   **CMake** (3.18+)
*   **Compiler:** `nvcc` (NVIDIA) or `hipcc` (AMD)
*   **C++ Compiler:** `g++` or `clang`

### 1. Clone
```bash
git clone https://github.com/vivek-kumar9696/cullama.git
cd cuLlama
```

### 2. Build for NVIDIA (Default)
```bash
mkdir build && cd build
cmake ..
make
```

### 3. Build for AMD (ROCm)

To compile for AMD GPUs, we switch the backend flag. This triggers the hip_compat.h layer to remap CUDA calls to HIP.

```bash
mkdir build && cd build
cmake -DBACKEND=HIP ..
make
```
### 4. 🛠️ Python Exporter Setup

To run the C++ inference engine, you first need to download and convert the model weights from Hugging Face into our custom `.bin` format. We use a short Python script (`scripts/export_weights.py`) for this.

We recommend using [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver, to manage the dependencies.

#### A. Install `uv` (if you haven't already)
If you don't have `uv` installed on your system, you can install it via curl:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```
#### B. Set up the Environment

Navigate to your project directory and create a virtual environment using uv:
Bash

```bash
# Create a virtual environment
uv venv

# Activate the environment (Linux/macOS)
source .venv/bin/activate

# (Note: If you are on Windows, use `.venv\Scripts\activate`)
```
#### C. Install Dependencies

Once the virtual environment is activated, use uv pip to install the required machine learning packages. uv will install these significantly faster than standard pip.

```bash
uv pip install torch numpy transformers
```
#### D. Export the Weights

With the dependencies installed, you can now run the exporter script to generate the model.bin file:

```bash
python3 scripts/export_weights.py
```

### 5. Run

```bash
./cuLlama --model model.bin --prompt "The future of AI is"
```
---

## 📊 Benchmarks (WIP)

| Device | Precision | Model | Tokens/Sec |
| :--- | :---: | :---: | :---: |
| **RTX 4090** | FP16 | Llama-2-7B | *Pending* |
| **A100 80GB** | FP16 | Llama-2-7B | *Pending* |
| **MI250X** | FP16 | Llama-2-7B | *Pending* |

---

## 🗺️ Roadmap

- [x] **Phase 0:** Build System (CMake with Dual Backend support).
- [ ] **Phase 1:** Memory Arena & KV Cache Manager.
- [ ] **Phase 2:** Fused Kernels (RMSNorm, RoPE).
- [ ] **Phase 3:** FlashAttention Implementation.
- [ ] **Phase 4:** W8A16 Quantization (Int8 weights, FP16 compute).

---

## 🤝 Contributing

This is a portfolio project demonstrating Systems Engineering skills. Issues and PRs focusing on kernel optimization or hardware compatibility are welcome.

## 📄 License

MIT License.
