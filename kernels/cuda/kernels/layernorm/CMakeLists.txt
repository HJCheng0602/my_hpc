add_executable(layernorm layernorm.cu)
target_include_directories(layernorm PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(layernorm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME layernorm COMMAND layernorm)
