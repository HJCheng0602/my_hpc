cmake_minimum_required(VERSION 3.20)
project(cuda_kernels CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── 自动检测 GPU 架构（需要 CMake 3.24+），否则手动指定 ────────
# 常用: 70=V100, 75=T4, 80=A100, 86=3090, 89=4090, 90=H100
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# ── 编译选项 ─────────────────────────────────────────────────────
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    --use_fast_math \
    -lineinfo \
    --ptxas-options=-v")

# Debug 模式额外选项
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

# ── 全局 include ─────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── 子目录 ───────────────────────────────────────────────────────
# add_subdirectory(kernels/reduction)
# add_subdirectory(kernels/scan)
add_subdirectory(kernels/transpose)
# add_subdirectory(kernels/gemm)
# add_subdirectory(kernels/conv)
# add_subdirectory(kernels/softmax)
# add_subdirectory(kernels/layernorm)
# add_subdirectory(kernels/flash_attention)
# add_subdirectory(kernels/fusion)

# ── 可选：开启 ctest ─────────────────────────────────────────────
enable_testing()