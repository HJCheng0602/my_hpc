add_executable(scan scan.cu)
target_include_directories(scan PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME scan COMMAND scan)
