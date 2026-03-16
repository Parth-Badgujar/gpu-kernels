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

add_executable(nvfp4_v1 nvfp4_v1.cu)
target_link_libraries(nvfp4_v1 PRIVATE cuda)
add_executable(nvfp4_v2 nvfp4_v2.cu)
target_link_libraries(nvfp4_v2 PRIVATE cuda)
add_executable(nvfp4_v3 nvfp4_v3.cu)
target_link_libraries(nvfp4_v3 PRIVATE cuda)
add_executable(nvfp4_v4 nvfp4_v4.cu)
target_link_libraries(nvfp4_v4 PRIVATE cuda)