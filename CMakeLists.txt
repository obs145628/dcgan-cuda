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




add_custom_target(gen_datasets
  COMMAND
  ${CMAKE_SOURCE_DIR}/tests/scripts/gen_datasets.sh
  ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_BINARY_DIR})


add_custom_target(check
   COMMAND
   python ${CMAKE_SOURCE_DIR}/tests/pyts/ts.py
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_BINARY_DIR}
   DEPENDS tbin-dump tbin-diff gen_datasets test_mnist1 test_softmax test_log_softmax test_softmax_cross_entropy)


set(TEST_SRC
  ${SRC}
  tests/test.cc
)
add_executable(my_test ${TEST_SRC})
target_link_libraries(my_test cnpy)






### TESTS LIST ###

set(TEST_MNIST1_SRC
  ${SRC}
  tests/test_mnist1.cc
)
add_executable(test_mnist1 ${TEST_MNIST1_SRC})
target_link_libraries(test_mnist1 cnpy)

set(TEST_SOFTMAX_SRC
  ${SRC}
  tests/test_softmax.cc
)
add_executable(test_softmax ${TEST_SOFTMAX_SRC})
target_link_libraries(test_softmax cnpy)

set(TEST_LOG_SOFTMAX_SRC
  ${SRC}
  tests/test_log_softmax.cc
)
add_executable(test_log_softmax ${TEST_LOG_SOFTMAX_SRC})
target_link_libraries(test_log_softmax cnpy)

set(TEST_SOFTMAX_CROSS_ENTROPY_SRC
  ${SRC}
  tests/test_softmax_cross_entropy.cc
)
add_executable(test_softmax_cross_entropy ${TEST_SOFTMAX_CROSS_ENTROPY_SRC})
target_link_libraries(test_softmax_cross_entropy cnpy)
