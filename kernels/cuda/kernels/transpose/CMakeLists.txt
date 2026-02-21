add_executable(transpose transpose.cu)
target_include_directories(transpose PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME transpose COMMAND transpose)
