cmake_minimum_required(VERSION 3.18)
project(LAYER_NORM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20054")

find_package(OpenMP REQUIRED)

file(GLOB ATTENTION_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(attention_objs OBJECT
	${ATTENTION_SRC}
)

target_include_directories(attention_objs PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set_target_properties(attention_objs PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

add_library(attention SHARED
	$<TARGET_OBJECTS:attention_objs>
)

add_executable(attention_test
    ${CMAKE_CURRENT_SOURCE_DIR}/attention_test.cu
)
target_include_directories(attention_test PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(attention_test PUBLIC
	attention
	OpenMP::OpenMP_CXX
)
target_compile_options(attention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)