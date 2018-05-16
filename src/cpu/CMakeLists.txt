set(SRC
  kernels.cc
  runner.cc
  thread-pool-runner.cc

  conv2d.cc
)

add_library(cpu_obj OBJECT ${SRC})
