cmake_minimum_required(VERSION 3.2)
set(PROJECT dcgan-cuda)

#set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-O0 -g3 -Wall -Wextra -Werror -std=c++17")
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -std=c++17 -O3 -DNDEBUG")


add_subdirectory(ext/tocha)

include_directories(ext/tocha/include)

add_subdirectory(src/api)
add_subdirectory(src/cpu)
add_subdirectory(src/datasets)
add_subdirectory(src/memory)
add_subdirectory(src/ops)
add_subdirectory(src/runtime)
add_subdirectory(src/utils)

set(SRC
  $<TARGET_OBJECTS:api_obj>
  $<TARGET_OBJECTS:cpu_obj>
  $<TARGET_OBJECTS:datasets_obj>
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
   DEPENDS tbin-dump tbin-diff gen_datasets test_mnist1 test_softmax test_log_softmax test_softmax_cross_entropy test_conv2d test_sigmoid test_mat_mat_mul test_mat_rvect_add test_mse test_vect_relu test_vect_relu_leaky test_vect_tanh test_mse_grad test_sigmoid_grad test_mat_mul_add test_tmat_mat_mul test_mat_tmat_mul test_mat_mul_add_grad test_mat_sum0 test_mat_sum1 test_softmax_cross_entropy_grad test_mnist_grad test_conv2d_bias_add test_sigmoid_cross_entropy test_sigmoid_cross_entropy_grad test_conv2d_grad test_reshape test_dcgan_discriminator test_argmax test_update test_moment_update test_moment_update2 test_adam_update test_relu_grad test_conv2d_padding test_conv2d_padding_grad test_leaky_relu_grad test_conv2d_bias_add_grad test_tanh_grad test_conv2d_transpose test_conv2d_transpose_grad)


set(TEST_SRC
  ${SRC}
  tests/test.cc
)
add_executable(my_test ${TEST_SRC})
target_link_libraries(my_test cnpy)



set(NN_MNIST_SRC
  ${SRC}
  examples/nn_mnist.cc
)
add_executable(nn_mnist ${NN_MNIST_SRC})





### TESTS LIST ###

set(TEST_MNIST1_SRC
  ${SRC}
  tests/test_mnist1.cc
)
add_executable(test_mnist1 EXCLUDE_FROM_ALL ${TEST_MNIST1_SRC})
target_link_libraries(test_mnist1 cnpy)

set(TEST_SOFTMAX_SRC
  ${SRC}
  tests/test_softmax.cc
)
add_executable(test_softmax EXCLUDE_FROM_ALL ${TEST_SOFTMAX_SRC})
target_link_libraries(test_softmax cnpy)

set(TEST_LOG_SOFTMAX_SRC
  ${SRC}
  tests/test_log_softmax.cc
)
add_executable(test_log_softmax EXCLUDE_FROM_ALL ${TEST_LOG_SOFTMAX_SRC})
target_link_libraries(test_log_softmax cnpy)

set(TEST_SOFTMAX_CROSS_ENTROPY_SRC
  ${SRC}
  tests/test_softmax_cross_entropy.cc
)
add_executable(test_softmax_cross_entropy EXCLUDE_FROM_ALL ${TEST_SOFTMAX_CROSS_ENTROPY_SRC})
target_link_libraries(test_softmax_cross_entropy cnpy)

set(TEST_CONV2D_SRC
  ${SRC}
  tests/test_conv2d.cc
)
add_executable(test_conv2d EXCLUDE_FROM_ALL ${TEST_CONV2D_SRC})
target_link_libraries(test_conv2d cnpy)

set(TEST_CONV2D_PADDING_SRC
  ${SRC}
  tests/test_conv2d_padding.cc
)
add_executable(test_conv2d_padding EXCLUDE_FROM_ALL ${TEST_CONV2D_PADDING_SRC})
target_link_libraries(test_conv2d_padding cnpy)


set(TEST_CONV2D_BIAS_ADD_SRC
  ${SRC}
  tests/test_conv2d_bias_add.cc
)
add_executable(test_conv2d_bias_add EXCLUDE_FROM_ALL ${TEST_CONV2D_BIAS_ADD_SRC})
target_link_libraries(test_conv2d_bias_add cnpy)

set(TEST_CONV2D_BIAS_ADD_GRAD_SRC
  ${SRC}
  tests/test_conv2d_bias_add_grad.cc
)
add_executable(test_conv2d_bias_add_grad EXCLUDE_FROM_ALL ${TEST_CONV2D_BIAS_ADD_GRAD_SRC})
target_link_libraries(test_conv2d_bias_add_grad cnpy)


set(TEST_CONV2D_GRAD_SRC
  ${SRC}
  tests/test_conv2d_grad.cc
)
add_executable(test_conv2d_grad EXCLUDE_FROM_ALL ${TEST_CONV2D_GRAD_SRC})
target_link_libraries(test_conv2d_grad cnpy)

set(TEST_CONV2D_PADDING_GRAD_SRC
  ${SRC}
  tests/test_conv2d_padding_grad.cc
)
add_executable(test_conv2d_padding_grad EXCLUDE_FROM_ALL ${TEST_CONV2D_PADDING_GRAD_SRC})
target_link_libraries(test_conv2d_padding_grad cnpy)

set(TEST_CONV2D_TRANSPOSE_SRC
  ${SRC}
  tests/test_conv2d_transpose.cc
)
add_executable(test_conv2d_transpose EXCLUDE_FROM_ALL ${TEST_CONV2D_TRANSPOSE_SRC})
target_link_libraries(test_conv2d_transpose cnpy)

set(TEST_CONV2D_TRANSPOSE_GRAD_SRC
  ${SRC}
  tests/test_conv2d_transpose_grad.cc
)
add_executable(test_conv2d_transpose_grad EXCLUDE_FROM_ALL ${TEST_CONV2D_TRANSPOSE_GRAD_SRC})
target_link_libraries(test_conv2d_transpose_grad cnpy)


set(TEST_VECT_RELU_SRC
  ${SRC}
  tests/test_vect_relu.cc
)
add_executable(test_vect_relu EXCLUDE_FROM_ALL ${TEST_VECT_RELU_SRC})
target_link_libraries(test_vect_relu cnpy)

set(TEST_SIGMOID_SRC
  ${SRC}
  tests/test_sigmoid.cc
)
add_executable(test_sigmoid EXCLUDE_FROM_ALL ${TEST_SIGMOID_SRC})
target_link_libraries(test_sigmoid cnpy)


set(TEST_MAT_MAT_MUL_SRC
  ${SRC}
  tests/test_mat_mat_mul.cc
)
add_executable(test_mat_mat_mul EXCLUDE_FROM_ALL ${TEST_MAT_MAT_MUL_SRC})
target_link_libraries(test_mat_mat_mul cnpy)

set(TEST_MAT_RVECT_ADD_SRC
  ${SRC}
  tests/test_mat_rvect_add.cc
)
add_executable(test_mat_rvect_add EXCLUDE_FROM_ALL ${TEST_MAT_RVECT_ADD_SRC})
target_link_libraries(test_mat_rvect_add cnpy)

set(TEST_MSE_SRC
  ${SRC}
  tests/test_mse.cc
)
add_executable(test_mse EXCLUDE_FROM_ALL ${TEST_MSE_SRC})
target_link_libraries(test_mse cnpy)

set(TEST_VECT_RELU_LEAKY_SRC
  ${SRC}
  tests/test_vect_relu_leaky.cc
)
add_executable(test_vect_relu_leaky EXCLUDE_FROM_ALL ${TEST_VECT_RELU_LEAKY_SRC})
target_link_libraries(test_vect_relu_leaky cnpy)

set(TEST_VECT_TANH_SRC
  ${SRC}
  tests/test_vect_tanh.cc
)
add_executable(test_vect_tanh EXCLUDE_FROM_ALL ${TEST_VECT_TANH_SRC})
target_link_libraries(test_vect_tanh cnpy)

set(TEST_RESHAPE_SRC
  ${SRC}
  tests/test_reshape.cc
)
add_executable(test_reshape ${TEST_RESHAPE_SRC})
target_link_libraries(test_reshape cnpy)

set(TEST_MSE_GRAD_SRC
  ${SRC}
  tests/test_mse_grad.cc
)
add_executable(test_mse_grad EXCLUDE_FROM_ALL ${TEST_MSE_GRAD_SRC})
target_link_libraries(test_mse_grad cnpy)

set(TEST_SIGMOID_GRAD_SRC
  ${SRC}
  tests/test_sigmoid_grad.cc
)
add_executable(test_sigmoid_grad EXCLUDE_FROM_ALL ${TEST_SIGMOID_GRAD_SRC})
target_link_libraries(test_sigmoid_grad cnpy)

set(TEST_MAT_MUL_ADD_SRC
  ${SRC}
  tests/test_mat_mul_add.cc
)
add_executable(test_mat_mul_add EXCLUDE_FROM_ALL ${TEST_MAT_MUL_ADD_SRC})
target_link_libraries(test_mat_mul_add cnpy)

set(TEST_TMAT_MAT_MUL_SRC
  ${SRC}
  tests/test_tmat_mat_mul.cc
)
add_executable(test_tmat_mat_mul EXCLUDE_FROM_ALL ${TEST_TMAT_MAT_MUL_SRC})
target_link_libraries(test_tmat_mat_mul cnpy)

set(TEST_MAT_TMAT_MUL_SRC
  ${SRC}
  tests/test_mat_tmat_mul.cc
)
add_executable(test_mat_tmat_mul EXCLUDE_FROM_ALL ${TEST_MAT_TMAT_MUL_SRC})
target_link_libraries(test_mat_tmat_mul cnpy)

set(TEST_MAT_MUL_ADD_GRAD_SRC
  ${SRC}
  tests/test_mat_mul_add_grad.cc
)
add_executable(test_mat_mul_add_grad EXCLUDE_FROM_ALL ${TEST_MAT_MUL_ADD_GRAD_SRC})
target_link_libraries(test_mat_mul_add_grad cnpy)

set(TEST_MAT_SUM0_SRC
  ${SRC}
  tests/test_mat_sum0.cc
)
add_executable(test_mat_sum0 EXCLUDE_FROM_ALL ${TEST_MAT_SUM0_SRC})
target_link_libraries(test_mat_sum0 cnpy)

set(TEST_MAT_SUM1_SRC
  ${SRC}
  tests/test_mat_sum1.cc
)
add_executable(test_mat_sum1 EXCLUDE_FROM_ALL ${TEST_MAT_SUM1_SRC})
target_link_libraries(test_mat_sum1 cnpy)

set(TEST_SOFTMAX_CROSS_ENTROPY_GRAD_SRC
  ${SRC}
  tests/test_softmax_cross_entropy_grad.cc
)
add_executable(test_softmax_cross_entropy_grad EXCLUDE_FROM_ALL ${TEST_SOFTMAX_CROSS_ENTROPY_GRAD_SRC})
target_link_libraries(test_softmax_cross_entropy_grad cnpy)

set(TEST_MNIST_GRAD_SRC
  ${SRC}
  tests/test_mnist_grad.cc
)
add_executable(test_mnist_grad EXCLUDE_FROM_ALL ${TEST_MNIST_GRAD_SRC})
target_link_libraries(test_mnist_grad cnpy)

set(TEST_SIGMOID_CROSS_ENTROPY_SRC
  ${SRC}
  tests/test_sigmoid_cross_entropy.cc
)
add_executable(test_sigmoid_cross_entropy EXCLUDE_FROM_ALL ${TEST_SIGMOID_CROSS_ENTROPY_SRC})
target_link_libraries(test_sigmoid_cross_entropy cnpy)

set(TEST_SIGMOID_CROSS_ENTROPY_GRAD_SRC
  ${SRC}
  tests/test_sigmoid_cross_entropy_grad.cc
)
add_executable(test_sigmoid_cross_entropy_grad EXCLUDE_FROM_ALL ${TEST_SIGMOID_CROSS_ENTROPY_GRAD_SRC})
target_link_libraries(test_sigmoid_cross_entropy_grad cnpy)

set(TEST_DCGAN_DISCRIMINATOR_SRC
  ${SRC}
  tests/test_dcgan_discriminator.cc
)
add_executable(test_dcgan_discriminator EXCLUDE_FROM_ALL ${TEST_DCGAN_DISCRIMINATOR_SRC})
target_link_libraries(test_dcgan_discriminator cnpy)

set(TEST_TANH_GRAD_SRC
  ${SRC}
  tests/test_tanh_grad.cc
)
add_executable(test_tanh_grad EXCLUDE_FROM_ALL ${TEST_TANH_GRAD_SRC})
target_link_libraries(test_tanh_grad cnpy)

set(TEST_TRAIN_MNIST_SGD_SRC
  ${SRC}
  tests/test_train_mnist_sgd.cc
)
add_executable(test_train_mnist_sgd EXCLUDE_FROM_ALL ${TEST_TRAIN_MNIST_SGD_SRC})
target_link_libraries(test_train_mnist_sgd cnpy)

set(TEST_ARGMAX_SRC
  ${SRC}
  tests/test_argmax.cc
)
add_executable(test_argmax EXCLUDE_FROM_ALL ${TEST_ARGMAX_SRC})
target_link_libraries(test_argmax cnpy)

set(TEST_UPDATE_SRC
  ${SRC}
  tests/test_update.cc
)
add_executable(test_update EXCLUDE_FROM_ALL ${TEST_UPDATE_SRC})
target_link_libraries(test_update cnpy)

set(TEST_MOMENT_UPDATE_SRC
  ${SRC}
  tests/test_moment_update.cc
)
add_executable(test_moment_update EXCLUDE_FROM_ALL ${TEST_MOMENT_UPDATE_SRC})
target_link_libraries(test_moment_update cnpy)

set(TEST_MOMENT_UPDATE2_SRC
  ${SRC}
  tests/test_moment_update2.cc
)
add_executable(test_moment_update2 EXCLUDE_FROM_ALL ${TEST_MOMENT_UPDATE2_SRC})
target_link_libraries(test_moment_update2 cnpy)

set(TEST_ADAM_UPDATE_SRC
  ${SRC}
  tests/test_adam_update.cc
)
add_executable(test_adam_update EXCLUDE_FROM_ALL ${TEST_ADAM_UPDATE_SRC})
target_link_libraries(test_adam_update cnpy)

set(TEST_RELU_GRAD_SRC
  ${SRC}
  tests/test_relu_grad.cc
)
add_executable(test_relu_grad EXCLUDE_FROM_ALL ${TEST_RELU_GRAD_SRC})
target_link_libraries(test_relu_grad cnpy)

set(TEST_LEAKY_RELU_GRAD_SRC
  ${SRC}
  tests/test_leaky_relu_grad.cc
)
add_executable(test_leaky_relu_grad EXCLUDE_FROM_ALL ${TEST_LEAKY_RELU_GRAD_SRC})
target_link_libraries(test_leaky_relu_grad cnpy)
