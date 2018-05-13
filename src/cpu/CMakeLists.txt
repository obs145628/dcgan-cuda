set(SRC
  kernels.cc
  runner.cc
  thread-pool-runner.cc
)

add_library(cpu_obj OBJECT ${SRC})
