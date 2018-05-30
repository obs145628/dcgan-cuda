set(SRC
  kernels.cu
  matmul.cu
  runner.cu
  sigmoid.cu
  sum.cu
)

add_library(gpu_obj OBJECT ${SRC})
