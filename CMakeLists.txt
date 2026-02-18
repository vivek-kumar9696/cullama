cmake_minimum_required(VERSION 3.18)
project(cuLlama LANGUAGES CXX)

# --- Configuration ---
set(BACKEND "CUDA" CACHE STRING "Compute backend: CUDA or HIP")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies (GoogleTest) ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# Suppress GTest installation messages
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# --- Backend Setup ---
if(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    add_compile_definitions(USE_CUDA)
    
    # Define the backend library (cudart)
    set(BACKEND_LIB CUDA::cudart)

elseif(BACKEND STREQUAL "HIP")
    list(APPEND CMAKE_MODULE_PATH /opt/rocm/hip/cmake) 
    find_package(HIP REQUIRED)
    add_compile_definitions(USE_ROCM)
    add_compile_definitions(__HIP_PLATFORM_AMD__)
    set(BACKEND_LIB hip::host)
endif()

# --- 1. The Core Library (libcuLlama) ---
# This contains all the logic, but NO main() function.
file(GLOB_RECURSE CORE_SOURCES 
    "src/memory_manager.cpp"
    "src/kv_cache.cpp"
    "kernels/*.cu"
)

# We create a LIBRARY, not an executable
add_library(cuLlama_lib STATIC ${CORE_SOURCES})

# Set include paths for the library
target_include_directories(cuLlama_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
)

# Link the backend (CUDA/HIP) to the library
target_link_libraries(cuLlama_lib PUBLIC ${BACKEND_LIB})

if(BACKEND STREQUAL "HIP")
    set_source_files_properties(${CORE_SOURCES} PROPERTIES LANGUAGE HIP)
endif()


# --- 2. The Application (cuLlama) ---
add_executable(cuLlama src/main.cpp)
target_link_libraries(cuLlama PRIVATE cuLlama_lib)


# --- 3. The Test Suite (unit_tests) ---
# We create a separate executable for testing
add_executable(unit_tests tests/test_kernels.cpp)

# Link the Core Library AND GoogleTest
target_link_libraries(unit_tests PRIVATE 
    cuLlama_lib 
    GTest::gtest_main
)

# Allow the test file to use CUDA calls directly if needed
if(BACKEND STREQUAL "CUDA")
    set_target_properties(unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Register with CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)