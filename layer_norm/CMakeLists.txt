cmake_minimum_required(VERSION 3.18)
project(LAYER_NORM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 100)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20054")
set(CUDA_HOME "$ENV{CUDA_HOME}")
set(CUDACXX "${CUDA_HOME}/bin/nvcc")
set(CMAKE_CUDA_COMPILER ${CUDA_HOME}/bin/nvcc CACHE FILEPATH "CUDA compiler")
set(DCUDAToolkit_ROOT ${CUDA_HOME})


find_package(OpenMP REQUIRED)

file(GLOB LAYERNORM_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(layer_norm_objs OBJECT
	${LAYERNORM_SRC}
)

target_include_directories(layer_norm_objs PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set_target_properties(layer_norm_objs PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

add_library(layer_norm SHARED
	$<TARGET_OBJECTS:layer_norm_objs>
)

add_executable(layer_norm_test
    ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.cu
)
target_include_directories(layer_norm_test PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(layer_norm_test INTERFACE "${CMAKE_CURRENT_LIST_DIR}/../cccl/thrust"
"${CMAKE_CURRENT_LIST_DIR}/../cccl"
)
target_link_libraries(layer_norm_test PUBLIC
	layer_norm
	OpenMP::OpenMP_CXX
)
