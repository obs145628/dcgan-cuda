cmake_minimum_required(VERSION 3.2)
set(PROJECT dcgan-cuda)

#set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-O0 -g3 -Wall -Wextra -Werror -std=c++17")
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -std=c++17 -O3 -DNDEBUG")


add_subdirectory(ext/tocha)

include_directories(ext/tocha/include)

add_subdirectory(src/datasets)
add_subdirectory(src/math_cpu)
add_subdirectory(src/memory)
add_subdirectory(src/network)


set(SRC
  $<TARGET_OBJECTS:datasets_obj>
  $<TARGET_OBJECTS:math_cpu_obj>
  $<TARGET_OBJECTS:memory_obj>
  $<TARGET_OBJECTS:network_obj>
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
  
