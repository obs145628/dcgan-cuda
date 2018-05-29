set(SRC
  kernels.cu
  runner.cu
  simd_kernels.cu
  thread-pool-runner.cu
)

add_library(cpu_obj OBJECT ${SRC})
