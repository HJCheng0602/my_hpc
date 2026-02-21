add_executable(fusion fusion.cu)
target_include_directories(fusion PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(fusion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME fusion COMMAND fusion)
