add_executable(gemm gemm.cu)
target_include_directories(gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME gemm COMMAND gemm)


find_package(CUDAToolkit REQUIRED)
target_link_libraries(gemm PRIVATE CUDA::cublas)