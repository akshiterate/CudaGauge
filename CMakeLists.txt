cmake_minimum_required(VERSION 3.18)

# CudaGauge: CPU/GPU crossover research and benchmarking library.
# RTX 2050 (laptop, 4 GB) is compute capability 8.6, hence arch 86.
project(CudaGauge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Reusable library: everything except the CLI entry point.
add_library(cudagauge STATIC
    src/timing.cpp
    src/model.cpp
    src/search.cpp
    src/benchmark.cu
    src/csv.cpp
    src/workloads_cpu.cpp
    src/workloads_gpu.cu
)
target_include_directories(cudagauge PUBLIC include PRIVATE src)

# Command-line tool built on the library.
add_executable(cudagauge_cli src/main.cpp)
target_link_libraries(cudagauge_cli PRIVATE cudagauge)

# Independent sample application (Level 2 test): consumes only the
# public headers, like an external developer would.
add_executable(sample_app examples/sample_app.cu)
target_link_libraries(sample_app PRIVATE cudagauge)

# Level 3 integration: NVIDIA cuda-samples vectorAdd behind the public
# adapter. The verbatim upstream file is NOT compiled; only the adapter is.
add_executable(vectorAdd_cudagauge examples/third_party/vectorAdd/vectorAdd_cudagauge.cu)
target_link_libraries(vectorAdd_cudagauge PRIVATE cudagauge)

# Tests: plain executables with their own main(), no test framework.
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE cudagauge)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model PRIVATE cudagauge)

add_executable(test_search tests/test_search.cpp)
target_link_libraries(test_search PRIVATE cudagauge)

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
add_test(NAME model COMMAND test_model)
add_test(NAME search COMMAND test_search)
add_test(NAME thirdparty_vectorAdd COMMAND vectorAdd_cudagauge --quick --out results)
