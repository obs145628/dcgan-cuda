cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(dcgan-cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}; -std=c++11)

set(CMAKE_CXX_FLAGS "-pthread -O0 -g3 -Wall -Wextra -Werror -std=c++11 -fPIC")
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -std=c++11 -O3 -DNDEBUG")

include_directories(ext/tocha/include)
find_package(ZLIB REQUIRED)

add_subdirectory(src/api)
add_subdirectory(src/cpu)
add_subdirectory(src/datasets)
add_subdirectory(src/gpu)
add_subdirectory(src/memory)
add_subdirectory(src/ops)
add_subdirectory(src/runtime)
add_subdirectory(src/utils)

set(SRC
  $<TARGET_OBJECTS:api_obj>
  $<TARGET_OBJECTS:cpu_obj>
  $<TARGET_OBJECTS:datasets_obj>
  $<TARGET_OBJECTS:gpu_obj>
  $<TARGET_OBJECTS:memory_obj>
  $<TARGET_OBJECTS:ops_obj>
  $<TARGET_OBJECTS:runtime_obj>
  $<TARGET_OBJECTS:utils_obj>
)



add_custom_target(gen_datasets
  COMMAND
  ${CMAKE_SOURCE_DIR}/tests/scripts/gen_datasets.sh
  ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR})


add_custom_target(check
   COMMAND
   python ${CMAKE_SOURCE_DIR}/tests/pyts/ts.py
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_BINARY_DIR}
   DEPENDS gen_datasets test_mnist1 test_softmax test_log_softmax test_softmax_cross_entropy test_conv2d test_sigmoid test_mat_mat_mul test_mat_rvect_add test_mse test_vect_relu test_vect_relu_leaky test_vect_tanh test_mse_grad test_sigmoid_grad test_mat_mul_add test_tmat_mat_mul test_mat_tmat_mul test_mat_mul_add_grad test_mat_sum0 test_mat_sum1 test_softmax_cross_entropy_grad test_mnist_grad test_conv2d_bias_add test_sigmoid_cross_entropy test_sigmoid_cross_entropy_grad test_conv2d_grad test_reshape test_dcgan_discriminator test_argmax test_update test_moment_update test_moment_update2 test_adam_update test_relu_grad test_conv2d_padding test_conv2d_padding_grad test_leaky_relu_grad test_conv2d_bias_add_grad test_tanh_grad test_conv2d_transpose test_conv2d_transpose_grad test_add test_sigmoid_50k test_conv_d0 test_conv_d1 test_conv_d2 test_conv_d3 test_conv_s1 test_conv_dx_d0 test_conv_dx_d1 test_conv_dx_d2 test_conv_dx_d3 test_conv_s2 test_conv_dk_d0 test_conv_dk_d1 test_conv_dk_d2 test_conv_dk_d3 test_conv_g0 test_conv_g1 test_conv_g2 test_conv_g3 test_conv_dx_g0 test_conv_dx_g1 test_conv_dx_g2 test_conv_dx_g3 test_conv_dk_g0 test_conv_dk_g1 test_conv_dk_g2 test_conv_dk_g3 test_dcgan_generator
   test_log_softmax_bm test_mat_mat_mul_bm test_mat_rvect_add_bm test_moment_update_bm test_moment_update2_bm test_mse_grad_bm test_mse_bm test_relu_grad_bm test_sigmoid_cross_entropy_grad_bm test_sigmoid_cross_entropy_bm test_sigmoid_grad_bm test_sigmoid_bm test_softmax_cross_entropy_grad_bm test_softmax_cross_entropy_bm test_softmax_bm test_tanh_grad_bm test_update_bm test_vect_relu_leaky_bm test_vect_relu_bm test_vect_tanh_bm)

set(TEST_SRC
  ${SRC}
  tests/test.cu
)
add_executable(my_test ${TEST_SRC})
target_link_libraries(my_test ${ZLIB_LIBRARIES} pthread jpeg)



set(NN_MNIST_SRC
  ${SRC}
  examples/nn_mnist.cu
)
add_executable(nn_mnist ${NN_MNIST_SRC})
target_link_libraries(nn_mnist ${ZLIB_LIBRARIES} jpeg)

set(DCGAN_SRC
  ${SRC}
  dcgan/main.cu
)
add_executable(dcgan ${DCGAN_SRC})
target_link_libraries(dcgan ${ZLIB_LIBRARIES} pthread jpeg)




### TESTS LIST ###

set(TEST_MNIST1_SRC
  ${SRC}
  tests/test_mnist1.cu
)
add_executable(test_mnist1 EXCLUDE_FROM_ALL ${TEST_MNIST1_SRC})
target_link_libraries(test_mnist1 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SOFTMAX_SRC
  ${SRC}
  tests/test_softmax.cu
)
add_executable(test_softmax EXCLUDE_FROM_ALL ${TEST_SOFTMAX_SRC})
target_link_libraries(test_softmax ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_LOG_SOFTMAX_SRC
  ${SRC}
  tests/test_log_softmax.cu
)
add_executable(test_log_softmax EXCLUDE_FROM_ALL ${TEST_LOG_SOFTMAX_SRC})
target_link_libraries(test_log_softmax ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SOFTMAX_CROSS_ENTROPY_SRC
  ${SRC}
  tests/test_softmax_cross_entropy.cu
)
add_executable(test_softmax_cross_entropy EXCLUDE_FROM_ALL ${TEST_SOFTMAX_CROSS_ENTROPY_SRC})
target_link_libraries(test_softmax_cross_entropy ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV2D_SRC
  ${SRC}
  tests/test_conv2d.cu
)
add_executable(test_conv2d EXCLUDE_FROM_ALL ${TEST_CONV2D_SRC})
target_link_libraries(test_conv2d ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV2D_PADDING_SRC
  ${SRC}
  tests/test_conv2d_padding.cu
)
add_executable(test_conv2d_padding EXCLUDE_FROM_ALL ${TEST_CONV2D_PADDING_SRC})
target_link_libraries(test_conv2d_padding ${ZLIB_LIBRARIES} pthread jpeg)


set(TEST_CONV2D_BIAS_ADD_SRC
  ${SRC}
  tests/test_conv2d_bias_add.cu
)
add_executable(test_conv2d_bias_add EXCLUDE_FROM_ALL ${TEST_CONV2D_BIAS_ADD_SRC})
target_link_libraries(test_conv2d_bias_add ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV2D_BIAS_ADD_GRAD_SRC
  ${SRC}
  tests/test_conv2d_bias_add_grad.cu
)
add_executable(test_conv2d_bias_add_grad EXCLUDE_FROM_ALL ${TEST_CONV2D_BIAS_ADD_GRAD_SRC})
target_link_libraries(test_conv2d_bias_add_grad ${ZLIB_LIBRARIES} pthread jpeg)


set(TEST_CONV2D_GRAD_SRC
  ${SRC}
  tests/test_conv2d_grad.cu
)
add_executable(test_conv2d_grad EXCLUDE_FROM_ALL ${TEST_CONV2D_GRAD_SRC})
target_link_libraries(test_conv2d_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV2D_PADDING_GRAD_SRC
  ${SRC}
  tests/test_conv2d_padding_grad.cu
)
add_executable(test_conv2d_padding_grad EXCLUDE_FROM_ALL ${TEST_CONV2D_PADDING_GRAD_SRC})
target_link_libraries(test_conv2d_padding_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV2D_TRANSPOSE_SRC
  ${SRC}
  tests/test_conv2d_transpose.cu
)
add_executable(test_conv2d_transpose EXCLUDE_FROM_ALL ${TEST_CONV2D_TRANSPOSE_SRC})
target_link_libraries(test_conv2d_transpose ${ZLIB_LIBRARIES} jpeg)

set(TEST_CONV2D_TRANSPOSE_GRAD_SRC
  ${SRC}
  tests/test_conv2d_transpose_grad.cu
)
add_executable(test_conv2d_transpose_grad EXCLUDE_FROM_ALL ${TEST_CONV2D_TRANSPOSE_GRAD_SRC})
target_link_libraries(test_conv2d_transpose_grad ${ZLIB_LIBRARIES} jpeg)


set(TEST_VECT_RELU_SRC
  ${SRC}
  tests/test_vect_relu.cu
)
add_executable(test_vect_relu EXCLUDE_FROM_ALL ${TEST_VECT_RELU_SRC})
target_link_libraries(test_vect_relu ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_SRC
  ${SRC}
  tests/test_sigmoid.cu
)
add_executable(test_sigmoid EXCLUDE_FROM_ALL ${TEST_SIGMOID_SRC})
target_link_libraries(test_sigmoid ${ZLIB_LIBRARIES} pthread jpeg)


set(TEST_MAT_MAT_MUL_SRC
  ${SRC}
  tests/test_mat_mat_mul.cu
)
add_executable(test_mat_mat_mul EXCLUDE_FROM_ALL ${TEST_MAT_MAT_MUL_SRC})
target_link_libraries(test_mat_mat_mul ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MAT_RVECT_ADD_SRC
  ${SRC}
  tests/test_mat_rvect_add.cu
)
add_executable(test_mat_rvect_add EXCLUDE_FROM_ALL ${TEST_MAT_RVECT_ADD_SRC})
target_link_libraries(test_mat_rvect_add ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MSE_SRC
  ${SRC}
  tests/test_mse.cu
)
add_executable(test_mse EXCLUDE_FROM_ALL ${TEST_MSE_SRC})
target_link_libraries(test_mse ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_VECT_RELU_LEAKY_SRC
  ${SRC}
  tests/test_vect_relu_leaky.cu
)
add_executable(test_vect_relu_leaky EXCLUDE_FROM_ALL ${TEST_VECT_RELU_LEAKY_SRC})
target_link_libraries(test_vect_relu_leaky ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_VECT_TANH_SRC
  ${SRC}
  tests/test_vect_tanh.cu
)
add_executable(test_vect_tanh EXCLUDE_FROM_ALL ${TEST_VECT_TANH_SRC})
target_link_libraries(test_vect_tanh ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_RESHAPE_SRC
  ${SRC}
  tests/test_reshape.cu
)
add_executable(test_reshape EXCLUDE_FROM_ALL ${TEST_RESHAPE_SRC})
target_link_libraries(test_reshape ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MSE_GRAD_SRC
  ${SRC}
  tests/test_mse_grad.cu
)
add_executable(test_mse_grad EXCLUDE_FROM_ALL ${TEST_MSE_GRAD_SRC})
target_link_libraries(test_mse_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_GRAD_SRC
  ${SRC}
  tests/test_sigmoid_grad.cu
)
add_executable(test_sigmoid_grad EXCLUDE_FROM_ALL ${TEST_SIGMOID_GRAD_SRC})
target_link_libraries(test_sigmoid_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MAT_MUL_ADD_SRC
  ${SRC}
  tests/test_mat_mul_add.cu
)
add_executable(test_mat_mul_add EXCLUDE_FROM_ALL ${TEST_MAT_MUL_ADD_SRC})
target_link_libraries(test_mat_mul_add ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_TMAT_MAT_MUL_SRC
  ${SRC}
  tests/test_tmat_mat_mul.cu
)
add_executable(test_tmat_mat_mul EXCLUDE_FROM_ALL ${TEST_TMAT_MAT_MUL_SRC})
target_link_libraries(test_tmat_mat_mul ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MAT_TMAT_MUL_SRC
  ${SRC}
  tests/test_mat_tmat_mul.cu
)
add_executable(test_mat_tmat_mul EXCLUDE_FROM_ALL ${TEST_MAT_TMAT_MUL_SRC})
target_link_libraries(test_mat_tmat_mul ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MAT_MUL_ADD_GRAD_SRC
  ${SRC}
  tests/test_mat_mul_add_grad.cu
)
add_executable(test_mat_mul_add_grad EXCLUDE_FROM_ALL ${TEST_MAT_MUL_ADD_GRAD_SRC})
target_link_libraries(test_mat_mul_add_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MAT_SUM0_SRC
  ${SRC}
  tests/test_mat_sum0.cu
)
add_executable(test_mat_sum0 EXCLUDE_FROM_ALL ${TEST_MAT_SUM0_SRC})
target_link_libraries(test_mat_sum0 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MAT_SUM1_SRC
  ${SRC}
  tests/test_mat_sum1.cu
)
add_executable(test_mat_sum1 EXCLUDE_FROM_ALL ${TEST_MAT_SUM1_SRC})
target_link_libraries(test_mat_sum1 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SOFTMAX_CROSS_ENTROPY_GRAD_SRC
  ${SRC}
  tests/test_softmax_cross_entropy_grad.cu
)
add_executable(test_softmax_cross_entropy_grad EXCLUDE_FROM_ALL ${TEST_SOFTMAX_CROSS_ENTROPY_GRAD_SRC})
target_link_libraries(test_softmax_cross_entropy_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MNIST_GRAD_SRC
  ${SRC}
  tests/test_mnist_grad.cu
)
add_executable(test_mnist_grad EXCLUDE_FROM_ALL ${TEST_MNIST_GRAD_SRC})
target_link_libraries(test_mnist_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_CROSS_ENTROPY_SRC
  ${SRC}
  tests/test_sigmoid_cross_entropy.cu
)
add_executable(test_sigmoid_cross_entropy EXCLUDE_FROM_ALL ${TEST_SIGMOID_CROSS_ENTROPY_SRC})
target_link_libraries(test_sigmoid_cross_entropy ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_CROSS_ENTROPY_GRAD_SRC
  ${SRC}
  tests/test_sigmoid_cross_entropy_grad.cu
)
add_executable(test_sigmoid_cross_entropy_grad EXCLUDE_FROM_ALL ${TEST_SIGMOID_CROSS_ENTROPY_GRAD_SRC})
target_link_libraries(test_sigmoid_cross_entropy_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_DCGAN_DISCRIMINATOR_SRC
  ${SRC}
  tests/test_dcgan_discriminator.cu
)
add_executable(test_dcgan_discriminator EXCLUDE_FROM_ALL ${TEST_DCGAN_DISCRIMINATOR_SRC})
target_link_libraries(test_dcgan_discriminator ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_TANH_GRAD_SRC
  ${SRC}
  tests/test_tanh_grad.cu
)
add_executable(test_tanh_grad EXCLUDE_FROM_ALL ${TEST_TANH_GRAD_SRC})
target_link_libraries(test_tanh_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_TRAIN_MNIST_SGD_SRC
  ${SRC}
  tests/test_train_mnist_sgd.cu
)
add_executable(test_train_mnist_sgd EXCLUDE_FROM_ALL ${TEST_TRAIN_MNIST_SGD_SRC})
target_link_libraries(test_train_mnist_sgd ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_ARGMAX_SRC
  ${SRC}
  tests/test_argmax.cu
)
add_executable(test_argmax EXCLUDE_FROM_ALL ${TEST_ARGMAX_SRC})
target_link_libraries(test_argmax ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_UPDATE_SRC
  ${SRC}
  tests/test_update.cu
)
add_executable(test_update EXCLUDE_FROM_ALL ${TEST_UPDATE_SRC})
target_link_libraries(test_update ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MOMENT_UPDATE_SRC
  ${SRC}
  tests/test_moment_update.cu
)
add_executable(test_moment_update EXCLUDE_FROM_ALL ${TEST_MOMENT_UPDATE_SRC})
target_link_libraries(test_moment_update ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MOMENT_UPDATE2_SRC
  ${SRC}
  tests/test_moment_update2.cu
)
add_executable(test_moment_update2 EXCLUDE_FROM_ALL ${TEST_MOMENT_UPDATE2_SRC})
target_link_libraries(test_moment_update2 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_ADAM_UPDATE_SRC
  ${SRC}
  tests/test_adam_update.cu
)
add_executable(test_adam_update EXCLUDE_FROM_ALL ${TEST_ADAM_UPDATE_SRC})
target_link_libraries(test_adam_update ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_RELU_GRAD_SRC
  ${SRC}
  tests/test_relu_grad.cu
)
add_executable(test_relu_grad EXCLUDE_FROM_ALL ${TEST_RELU_GRAD_SRC})
target_link_libraries(test_relu_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_ADD_SRC
  ${SRC}
  tests/test_add.cu
)
add_executable(test_add EXCLUDE_FROM_ALL ${TEST_ADD_SRC})
target_link_libraries(test_add ${ZLIB_LIBRARIES} pthread jpeg)


set(TEST_LEAKY_RELU_GRAD_SRC
  ${SRC}
  tests/test_leaky_relu_grad.cu
)
add_executable(test_leaky_relu_grad EXCLUDE_FROM_ALL ${TEST_LEAKY_RELU_GRAD_SRC})
target_link_libraries(test_leaky_relu_grad ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_50K_SRC
  ${SRC}
  tests/test_sigmoid_50k.cu
)
add_executable(test_sigmoid_50k EXCLUDE_FROM_ALL ${TEST_SIGMOID_50K_SRC})
target_link_libraries(test_sigmoid_50k ${ZLIB_LIBRARIES} pthread jpeg)


set(TEST_CONV_D0_SRC
  ${SRC}
  tests/test_conv_d0.cu
)
add_executable(test_conv_d0 EXCLUDE_FROM_ALL ${TEST_CONV_D0_SRC})
target_link_libraries(test_conv_d0 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_D1_SRC
  ${SRC}
  tests/test_conv_d1.cu
)
add_executable(test_conv_d1 EXCLUDE_FROM_ALL ${TEST_CONV_D1_SRC})
target_link_libraries(test_conv_d1 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_D2_SRC
  ${SRC}
  tests/test_conv_d2.cu
)
add_executable(test_conv_d2 EXCLUDE_FROM_ALL ${TEST_CONV_D2_SRC})
target_link_libraries(test_conv_d2 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_D3_SRC
  ${SRC}
  tests/test_conv_d3.cu
)
add_executable(test_conv_d3 EXCLUDE_FROM_ALL ${TEST_CONV_D3_SRC})
target_link_libraries(test_conv_d3 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_S1_SRC
  ${SRC}
  tests/test_conv_s1.cu
)
add_executable(test_conv_s1 EXCLUDE_FROM_ALL ${TEST_CONV_S1_SRC})
target_link_libraries(test_conv_s1 ${ZLIB_LIBRARIES} pthread jpeg)


set(TEST_CONV_DX_D0_SRC
  ${SRC}
  tests/test_conv_dx_d0.cu
)
add_executable(test_conv_dx_d0 EXCLUDE_FROM_ALL ${TEST_CONV_DX_D0_SRC})
target_link_libraries(test_conv_dx_d0 ${ZLIB_LIBRARIES} pthread jpeg)


set(TEST_CONV_DX_D1_SRC
  ${SRC}
  tests/test_conv_dx_d1.cu
)
add_executable(test_conv_dx_d1 EXCLUDE_FROM_ALL ${TEST_CONV_DX_D1_SRC})
target_link_libraries(test_conv_dx_d1 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DX_D2_SRC
  ${SRC}
  tests/test_conv_dx_d2.cu
)
add_executable(test_conv_dx_d2 EXCLUDE_FROM_ALL ${TEST_CONV_DX_D2_SRC})
target_link_libraries(test_conv_dx_d2 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DX_D3_SRC
  ${SRC}
  tests/test_conv_dx_d3.cu
)
add_executable(test_conv_dx_d3 EXCLUDE_FROM_ALL ${TEST_CONV_DX_D3_SRC})
target_link_libraries(test_conv_dx_d3 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_S2_SRC
  ${SRC}
  tests/test_conv_s2.cu
)
add_executable(test_conv_s2 EXCLUDE_FROM_ALL ${TEST_CONV_S2_SRC})
target_link_libraries(test_conv_s2 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DK_D0_SRC
  ${SRC}
  tests/test_conv_dk_d0.cu
)
add_executable(test_conv_dk_d0 EXCLUDE_FROM_ALL ${TEST_CONV_DK_D0_SRC})
target_link_libraries(test_conv_dk_d0 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DK_D1_SRC
  ${SRC}
  tests/test_conv_dk_d1.cu
)
add_executable(test_conv_dk_d1 EXCLUDE_FROM_ALL ${TEST_CONV_DK_D1_SRC})
target_link_libraries(test_conv_dk_d1 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DK_D2_SRC
  ${SRC}
  tests/test_conv_dk_d2.cu
)
add_executable(test_conv_dk_d2 EXCLUDE_FROM_ALL ${TEST_CONV_DK_D2_SRC})
target_link_libraries(test_conv_dk_d2 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DK_D3_SRC
  ${SRC}
  tests/test_conv_dk_d3.cu
)
add_executable(test_conv_dk_d3 EXCLUDE_FROM_ALL ${TEST_CONV_DK_D3_SRC})
target_link_libraries(test_conv_dk_d3 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_G0_SRC
  ${SRC}
  tests/test_conv_g0.cu
)
add_executable(test_conv_g0 EXCLUDE_FROM_ALL ${TEST_CONV_G0_SRC})
target_link_libraries(test_conv_g0 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_G1_SRC
  ${SRC}
  tests/test_conv_g1.cu
)
add_executable(test_conv_g1 EXCLUDE_FROM_ALL ${TEST_CONV_G1_SRC})
target_link_libraries(test_conv_g1 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_G2_SRC
  ${SRC}
  tests/test_conv_g2.cu
)
add_executable(test_conv_g2 EXCLUDE_FROM_ALL ${TEST_CONV_G2_SRC})
target_link_libraries(test_conv_g2 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_G3_SRC
  ${SRC}
  tests/test_conv_g3.cu
)
add_executable(test_conv_g3 EXCLUDE_FROM_ALL ${TEST_CONV_G3_SRC})
target_link_libraries(test_conv_g3 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DX_G0_SRC
  ${SRC}
  tests/test_conv_dx_g0.cu
)
add_executable(test_conv_dx_g0 EXCLUDE_FROM_ALL ${TEST_CONV_DX_G0_SRC})
target_link_libraries(test_conv_dx_g0 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DX_G1_SRC
  ${SRC}
  tests/test_conv_dx_g1.cu
)
add_executable(test_conv_dx_g1 EXCLUDE_FROM_ALL ${TEST_CONV_DX_G1_SRC})
target_link_libraries(test_conv_dx_g1 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DX_G2_SRC
  ${SRC}
  tests/test_conv_dx_g2.cu
)
add_executable(test_conv_dx_g2 EXCLUDE_FROM_ALL ${TEST_CONV_DX_G2_SRC})
target_link_libraries(test_conv_dx_g2 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DX_G3_SRC
  ${SRC}
  tests/test_conv_dx_g3.cu
)
add_executable(test_conv_dx_g3 EXCLUDE_FROM_ALL ${TEST_CONV_DX_G3_SRC})
target_link_libraries(test_conv_dx_g3 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DK_G0_SRC
  ${SRC}
  tests/test_conv_dk_g0.cu
)
add_executable(test_conv_dk_g0 EXCLUDE_FROM_ALL ${TEST_CONV_DK_G0_SRC})
target_link_libraries(test_conv_dk_g0 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DK_G1_SRC
  ${SRC}
  tests/test_conv_dk_g1.cu
)
add_executable(test_conv_dk_g1 EXCLUDE_FROM_ALL ${TEST_CONV_DK_G1_SRC})
target_link_libraries(test_conv_dk_g1 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DK_G2_SRC
  ${SRC}
  tests/test_conv_dk_g2.cu
)
add_executable(test_conv_dk_g2 EXCLUDE_FROM_ALL ${TEST_CONV_DK_G2_SRC})
target_link_libraries(test_conv_dk_g2 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_CONV_DK_G3_SRC
  ${SRC}
  tests/test_conv_dk_g3.cu
)
add_executable(test_conv_dk_g3 EXCLUDE_FROM_ALL ${TEST_CONV_DK_G3_SRC})
target_link_libraries(test_conv_dk_g3 ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_DCGAN_GENERATOR_SRC
  ${SRC}
  tests/test_dcgan_generator.cu
)
add_executable(test_dcgan_generator EXCLUDE_FROM_ALL ${TEST_DCGAN_GENERATOR_SRC})
target_link_libraries(test_dcgan_generator ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_LOG_SOFTMAX_BM_SRC
  ${SRC}
  tests/test_log_softmax_bm.cu
)
add_executable(test_log_softmax_bm EXCLUDE_FROM_ALL ${TEST_LOG_SOFTMAX_BM_SRC})
target_link_libraries(test_log_softmax_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MAT_MAT_MUL_BM_SRC
  ${SRC}
  tests/test_mat_mat_mul_bm.cu
)
add_executable(test_mat_mat_mul_bm EXCLUDE_FROM_ALL ${TEST_MAT_MAT_MUL_BM_SRC})
target_link_libraries(test_mat_mat_mul_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MAT_RVECT_ADD_BM_SRC
  ${SRC}
  tests/test_mat_rvect_add_bm.cu
)
add_executable(test_mat_rvect_add_bm EXCLUDE_FROM_ALL ${TEST_MAT_RVECT_ADD_BM_SRC})
target_link_libraries(test_mat_rvect_add_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MOMENT_UPDATE_BM_SRC
  ${SRC}
  tests/test_moment_update_bm.cu
)
add_executable(test_moment_update_bm EXCLUDE_FROM_ALL ${TEST_MOMENT_UPDATE_BM_SRC})
target_link_libraries(test_moment_update_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MOMENT_UPDATE2_BM_SRC
  ${SRC}
  tests/test_moment_update2_bm.cu
)
add_executable(test_moment_update2_bm EXCLUDE_FROM_ALL ${TEST_MOMENT_UPDATE2_BM_SRC})
target_link_libraries(test_moment_update2_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MSE_BM_SRC
  ${SRC}
  tests/test_mse_bm.cu
)
add_executable(test_mse_bm EXCLUDE_FROM_ALL ${TEST_MSE_BM_SRC})
target_link_libraries(test_mse_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_MSE_GRAD_BM_SRC
  ${SRC}
  tests/test_mse_grad_bm.cu
)
add_executable(test_mse_grad_bm EXCLUDE_FROM_ALL ${TEST_MSE_GRAD_BM_SRC})
target_link_libraries(test_mse_grad_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_BM_SRC
  ${SRC}
  tests/test_sigmoid_bm.cu
)
add_executable(test_sigmoid_bm EXCLUDE_FROM_ALL ${TEST_SIGMOID_BM_SRC})
target_link_libraries(test_sigmoid_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_GRAD_BM_SRC
  ${SRC}
  tests/test_sigmoid_grad_bm.cu
)
add_executable(test_sigmoid_grad_bm EXCLUDE_FROM_ALL ${TEST_SIGMOID_GRAD_BM_SRC})
target_link_libraries(test_sigmoid_grad_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_CROSS_ENTROPY_BM_SRC
  ${SRC}
  tests/test_sigmoid_cross_entropy_bm.cu
)
add_executable(test_sigmoid_cross_entropy_bm EXCLUDE_FROM_ALL ${TEST_SIGMOID_CROSS_ENTROPY_BM_SRC})
target_link_libraries(test_sigmoid_cross_entropy_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SIGMOID_CROSS_ENTROPY_GRAD_BM_SRC
  ${SRC}
  tests/test_sigmoid_cross_entropy_grad_bm.cu
)
add_executable(test_sigmoid_cross_entropy_grad_bm EXCLUDE_FROM_ALL ${TEST_SIGMOID_CROSS_ENTROPY_GRAD_BM_SRC})
target_link_libraries(test_sigmoid_cross_entropy_grad_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SOFTMAX_BM_SRC
  ${SRC}
  tests/test_softmax_bm.cu
)
add_executable(test_softmax_bm EXCLUDE_FROM_ALL ${TEST_SOFTMAX_BM_SRC})
target_link_libraries(test_softmax_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_TANH_GRAD_BM_SRC
  ${SRC}
  tests/test_tanh_grad_bm.cu
)
add_executable(test_tanh_grad_bm EXCLUDE_FROM_ALL ${TEST_TANH_GRAD_BM_SRC})
target_link_libraries(test_tanh_grad_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_UPDATE_BM_SRC
  ${SRC}
  tests/test_update_bm.cu
)
add_executable(test_update_bm EXCLUDE_FROM_ALL ${TEST_UPDATE_BM_SRC})
target_link_libraries(test_update_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_VECT_RELU_BM_SRC
  ${SRC}
  tests/test_vect_relu_bm.cu
)
add_executable(test_vect_relu_bm EXCLUDE_FROM_ALL ${TEST_VECT_RELU_BM_SRC})
target_link_libraries(test_vect_relu_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_VECT_RELU_LEAKY_BM_SRC
  ${SRC}
  tests/test_vect_relu_leaky_bm.cu
)
add_executable(test_vect_relu_leaky_bm EXCLUDE_FROM_ALL ${TEST_VECT_RELU_LEAKY_BM_SRC})
target_link_libraries(test_vect_relu_leaky_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_VECT_TANH_BM_SRC
  ${SRC}
  tests/test_vect_tanh_bm.cu
)
add_executable(test_vect_tanh_bm EXCLUDE_FROM_ALL ${TEST_VECT_TANH_BM_SRC})
target_link_libraries(test_vect_tanh_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_RELU_GRAD_BM_SRC
  ${SRC}
  tests/test_relu_grad_bm.cu
)
add_executable(test_relu_grad_bm EXCLUDE_FROM_ALL ${TEST_RELU_GRAD_BM_SRC})
target_link_libraries(test_relu_grad_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SOFTMAX_CROSS_ENTROPY_BM_SRC
  ${SRC}
  tests/test_softmax_cross_entropy_bm.cu
)
add_executable(test_softmax_cross_entropy_bm EXCLUDE_FROM_ALL ${TEST_SOFTMAX_CROSS_ENTROPY_BM_SRC})
target_link_libraries(test_softmax_cross_entropy_bm ${ZLIB_LIBRARIES} pthread jpeg)

set(TEST_SOFTMAX_CROSS_ENTROPY_GRAD_BM_SRC
  ${SRC}
  tests/test_softmax_cross_entropy_grad_bm.cu
)
add_executable(test_softmax_cross_entropy_grad_bm EXCLUDE_FROM_ALL ${TEST_SOFTMAX_CROSS_ENTROPY_GRAD_BM_SRC})
target_link_libraries(test_softmax_cross_entropy_grad_bm ${ZLIB_LIBRARIES} pthread jpeg)
