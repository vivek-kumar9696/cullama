cmake_minimum_required(VERSION 3.18)
project(cuLlama LANGUAGES CXX)

# --- Configuration ---
set(BACKEND "CUDA" CACHE STRING "Compute backend: CUDA or HIP")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Source Management ---
include_directories(include)
include_directories(include/common)

# Collect sources
file(GLOB_RECURSE HOST_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_KERNELS "kernels/*.cu")

# --- Backend Selection ---
if(BACKEND STREQUAL "CUDA")
    message(STATUS "Building for NVIDIA (CUDA) Backend")
    
    # 1. Enable CUDA Language
    enable_language(CUDA)
    
    # 2. Find the Toolkit (This finds headers AND libraries)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_ARCHITECTURES "native")
    add_compile_definitions(USE_CUDA)

    add_executable(cuLlama ${HOST_SOURCES} ${CUDA_KERNELS})

    # 3. Link against the Modern CMake target
    # CUDA::cudart handles the include directories for us automatically!
    target_link_libraries(cuLlama PRIVATE CUDA::cudart)

elseif(BACKEND STREQUAL "HIP")
    message(STATUS "Building for AMD (ROCm/HIP) Backend")
    
    list(APPEND CMAKE_MODULE_PATH /opt/rocm/hip/cmake) 
    find_package(HIP REQUIRED)
    
    add_compile_definitions(USE_ROCM)
    add_compile_definitions(__HIP_PLATFORM_AMD__)

    set_source_files_properties(${CUDA_KERNELS} PROPERTIES LANGUAGE HIP)
    
    hip_add_executable(cuLlama ${HOST_SOURCES} ${CUDA_KERNELS})

else()
    message(FATAL_ERROR "Unknown Backend: ${BACKEND}")
endif()