set(SRC
  kernels.cu
  matmul.cu
  relu.cu
  runner.cu
  sigmoid.cu
  softmax.cu
  sum.cu
  update.cu
)

add_library(gpu_obj OBJECT ${SRC})
