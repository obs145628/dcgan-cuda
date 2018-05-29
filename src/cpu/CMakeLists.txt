set(SRC
  kernels.cc
  runner.cc
  simd_kernels.cc
  thread-pool-runner.cc
)

add_library(cpu_obj OBJECT ${SRC})
