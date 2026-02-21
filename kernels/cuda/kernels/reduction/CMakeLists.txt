add_executable(reduction reduction.cu)
target_include_directories(reduction PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(reduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME reduction COMMAND reduction)
