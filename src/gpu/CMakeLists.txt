set(SRC
  kernels.cu
  runner.cu
  sigmoid.cu
)

add_library(gpu_obj OBJECT ${SRC})
