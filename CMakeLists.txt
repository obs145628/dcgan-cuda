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


set(SRC
  $<TARGET_OBJECTS:api_obj>
  $<TARGET_OBJECTS:cpu_obj>
  $<TARGET_OBJECTS:datasets_obj>
  $<TARGET_OBJECTS:memory_obj>
  $<TARGET_OBJECTS:ops_obj>
  $<TARGET_OBJECTS:runtime_obj>
)

set(MNIST_TEST_SRC
  ${SRC}
  tests/test_mnist.cc
)
add_executable(mnist_test ${MNIST_TEST_SRC})
target_link_libraries(mnist_test cnpy)

add_custom_target(check_mnist
  COMMAND
  ${CMAKE_SOURCE_DIR}/tests/mnist_test.sh ${CMAKE_SOURCE_DIR}/tests/ ${CMAKE_BINARY_DIR}
  DEPENDS mnist_test tbin-dump tbin-diff)
  


set(TEST_SRC
  ${SRC}
  tests/test.cc
)
add_executable(my_test ${TEST_SRC})
target_link_libraries(my_test cnpy)
