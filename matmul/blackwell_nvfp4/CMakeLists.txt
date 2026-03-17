cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 100a)
set(CMAKE_CXX_STANDARD 20)

set(CUDA_HOME "$ENV{CUDA_HOME}")
set(CUDACXX "${CUDA_HOME}/bin/nvcc")
set(CMAKE_CUDA_COMPILER ${CUDA_HOME}/bin/nvcc CACHE FILEPATH "CUDA compiler")
set(DCUDAToolkit_ROOT ${CUDA_HOME})
message(STATUS "Using CUDA_HOME: ${CUDA_HOME}")

project(blackwell_nvfp4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/include/cccl)

add_executable(nvfp4_gemm
    nvfp4_v1.cu
    nvfp4_v2.cu
    nvfp4_v3.cu
    nvfp4_v4.cu
    runner.cu
)
target_link_libraries(nvfp4_gemm PRIVATE cuda)