set(SRC
  kernels.cu
  matmul.cu
  runner.cu
  sigmoid.cu
  softmax.cu
  sum.cu
)

add_library(gpu_obj OBJECT ${SRC})
