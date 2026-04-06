cmake_minimum_required(VERSION 3.20)
set(CUDA_HOME "$ENV{CUDA_HOME}")
set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)

project(SM120MATMUL LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "120a")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v -Xcudafe --diag_suppress=20054")
set(CMAKE_BUILD_TYPE "Release")
set(CUDACXX "${CUDA_HOME}/bin/nvcc")
set(DCUDAToolkit_ROOT ${CUDA_HOME})

message("CUDA_HOME: ${CUDA_HOME}")
message("CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")

set(ENV{TORCH_CUDA_ARCH_LIST} "12.0a")
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)

add_executable(sm120_nvfp4 runner.cu nvfp4_v1.cu nvfp4_v2.cu nvfp4_v3.cu nvfp4_v4.cu nvfp4_v5.cu nvfp4_v6.cu)
target_link_libraries(sm120_nvfp4 PRIVATE ${TORCH_LIBRARIES} cuda)
target_include_directories(sm120_nvfp4 PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET sm120_nvfp4 PROPERTY CUDA_ARCHITECTURES "120a")