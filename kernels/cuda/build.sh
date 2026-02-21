#!/bin/bash
# build.sh — 快速构建脚本
# 用法:
#   ./build.sh           # Release 构建
#   ./build.sh debug     # Debug 构建（带 -G 符号，可用 cuda-gdb）
#   ./build.sh clean     # 清理构建目录
#   ./build.sh test      # 构建 + 跑所有测试

set -e

BUILD_TYPE="Release"
BUILD_DIR="build"

if [ "$1" == "debug" ]; then
  BUILD_TYPE="Debug"
  BUILD_DIR="build_debug"
elif [ "$1" == "clean" ]; then
  rm -rf build build_debug
  echo "Cleaned."
  exit 0
fi

# 自动检测 GPU 架构（需要 nvidia-smi）
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | head -1 | tr -d '.' || echo "80")
echo ">>> Detected GPU arch: sm_${ARCH}"

cmake -S . -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_CUDA_ARCHITECTURES=${ARCH}

cmake --build ${BUILD_DIR} -j$(nproc)

if [ "$1" == "test" ]; then
  cd ${BUILD_DIR} && ctest --output-on-failure
fi

echo ">>> Build complete: ./${BUILD_DIR}/"