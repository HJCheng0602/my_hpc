add_executable(conv conv.cu)
target_include_directories(conv PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(conv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME conv COMMAND conv)
