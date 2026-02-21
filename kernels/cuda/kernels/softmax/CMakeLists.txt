add_executable(softmax softmax.cu)
target_include_directories(softmax PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(softmax PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME softmax COMMAND softmax)
