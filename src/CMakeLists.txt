cmake_minimum_required(VERSION 3.10)
project(studentlib)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS 
    ${CUDA_NVCC_FLAGS} 
    -O3
    -gencode arch=compute_70,code=sm_70
    -Xptxas -v,-dlcm=ca,-warn-spills
    -Xcompiler -march=native
    --default-stream per-thread
    --restrict
    -lineinfo
    # --use_fast_math
    # --prec-div=false
    # --prec-sqrt=false
)

cuda_add_library(studentlib main.cu)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_options(studentlib PRIVATE -O3 -march=native)